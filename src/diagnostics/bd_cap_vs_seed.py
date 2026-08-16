"""Is the NaN a property of the bond-dim cap, or of the seed?

Background
----------
Three 3x3 runs on 2026-08-15 were compared and read as "64/96 are safe, 70/75
NaN":

    bd=96  seed 72314  clean through step 105
    bd=75  seed 89039  NaN by step 0
    bd=70  seed 93074  NaN at step 10

Those runs used *different* random seeds (seed=None draws a fresh one), so cap
and seed were fully confounded. This script replays ONE failing seed across the
"safe" caps and its own cap as control. If the NaN follows the seed, it will
reproduce at 64 and 96 too; if it follows the cap, only 75 will fail.

Must run on GPU. The failing jobs ran on an A40, and the NaN does not reproduce
on CPU at any cap -- cuSOLVER and LAPACK differ on rank-deficient SVD, which is
itself a hint that this is a conditioning problem rather than anything
arithmetic about the cap value.

Note on what "Current value: nan" means in a job log: find_gs.py reports
jnp.min(value) across trials, and the value is the energy at the CURRENT params,
so a NaN at step k means either the forward pass failed outright (k=0) or the
step k-1 gradient corrupted the params.
"""
import sys

import numpy as np
import optax

from src.find_gs import K
from src.utilities.ansatz_classes import ToricCodeAnsatz

SEED = 89039          # the bd=75 run's seed, which NaN'd at step 0
CAPS = [75, 64, 96]   # own cap first (control), then the two "safe" ones
TRIALS = 20           # match the original run: more trials, more chances to fail
STEPS = 30            # original NaN'd at step 0; bd=70's at step 10


def run(cap):
    a = ToricCodeAnsatz(
        Lx=3, Ly=3, nlayers=2, howoften_toreset=7, h=0.0,
        trials=TRIALS, maxiter=STEPS, howoften_tosave=10, unitary=True,
        sparse=False, use_mps=True, use_prob_resets_ansatz=True,
        reset_layers=[1], use_optimal_ordering=True,
        bond_dim=cap, seed=SEED,
    )
    params = a.initparams
    opt = optax.adam(learning_rate=0.01)
    opt_state = opt.init(params)

    print(f"\n### cap={cap} seed={SEED} trials={TRIALS}", flush=True)
    for i in range(STEPS):
        value, grad = a._cost_vvag(params)
        v, g = np.asarray(value), np.asarray(grad)
        bad_v = ~np.isfinite(v)
        bad_g = ~np.isfinite(g).all(axis=-1)
        if (bad_v | bad_g).any():
            print(f"VERDICT cap={cap} seed={SEED} NONFINITE at step={i} "
                  f"value_nan_trials={np.where(bad_v)[0].tolist()} "
                  f"grad_nan_trials={np.where(bad_g)[0].tolist()}", flush=True)
            return
        print(f"  step {i:3d}  min E={v.min():.8f}  |g|max={np.abs(g).max():.3e}",
              flush=True)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

    print(f"VERDICT cap={cap} seed={SEED} CLEAN through {STEPS} steps", flush=True)


if __name__ == "__main__":
    print(f"jax devices: {K.__class__.__name__}", flush=True)
    import jax
    print(f"jax backend devices: {jax.devices()}", flush=True)
    for cap in CAPS:
        run(cap)
