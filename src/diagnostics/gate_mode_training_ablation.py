"""Matched training-dynamics comparison across cartan_mode/toffoli_mode/bond_dim.

Standalone counterpart to test.py's Section 4 live-training diagnostic --
same ToricCodeAnsatz config, same optax.adam loop, same per-step
energy/grad_norm/min_gap logging (reusing test.py's own
_spectrum_degeneracy_summary for exact consistency) -- but parameterized via
CLI args instead of module-level constants, so several matched configs can
be submitted as separate jobs without editing test.py.

Context: job 3611962 (test.py Section 4, seed=22405, bond_dim=64,
cartan_mode="fused", toffoli_mode="direct", normalize_state=True) showed
energy descending smoothly through ~step 25, then oscillating/stagnating
from step ~30 on, driven by grad_norm spikes (up to 5 orders of magnitude)
correlated with shrinking MPS singular-value gaps -- never a literal NaN.
Job 3612091 (plot_training_curve_tc.py, bond_dim=96, same gate modes,
different seed) showed the same qualitative down-then-up energy pattern.
This script isolates whether cartan_mode/toffoli_mode and/or bond_dim
contribute, by rerunning the exact same seed/config with each axis flipped.

Usage: python -m src.diagnostics.gate_mode_training_ablation \
    --cartan-mode separate --toffoli-mode decomposed --bond-dim 64
"""
import argparse
import sys
import types

# TensorFlow is only needed by unrelated legacy constructors in
# generate_ansatz; stub it unconditionally -- see test_mps_gate_paths.py for
# why (some environments have a TF/sklearn build incompatible with the
# installed NumPy that crashes hard on import, not just raises cleanly).
sys.modules["tensorflow"] = types.ModuleType("tensorflow")

import numpy as np
import jax
import jax.numpy as jnp
import optax

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.generate_ansatz import get_singular_values_per_cut
from src.examples.test import _spectrum_degeneracy_summary

# Matches the LIVE_* config in src/examples/test.py's Section 4 (Lx/Ly/
# nlayers/howoften_toreset/reset_layers) so results stay comparable; seed
# and trials are CLI args since different reference runs used different
# values (job 3611962: seed=22405, trials=5; job 3612091: seed=4876, trials=3).
LX, LY = 3, 3
NLAYERS = 2
HOWOFTEN_TORESET = 7
RESET_LAYERS = [1]
NORMALIZE_STATE = True
USE_OPTIMAL_ORDERING = False
PRINT_EVERY = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cartan-mode", choices=["separate", "fused"], required=True)
    parser.add_argument("--toffoli-mode", choices=["decomposed", "direct"], required=True)
    parser.add_argument("--bond-dim", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--maxiter", type=int, default=150)
    args = parser.parse_args()

    print(f"  jax devices: {jax.devices()}")
    print(f"  config: cartan_mode={args.cartan_mode} toffoli_mode={args.toffoli_mode} "
          f"bond_dim={args.bond_dim} maxiter={args.maxiter} seed={args.seed} trials={args.trials}", flush=True)

    ansatz = ToricCodeAnsatz(
        Lx=LX, Ly=LY, nlayers=NLAYERS, howoften_toreset=HOWOFTEN_TORESET,
        reset_layers=RESET_LAYERS, trials=args.trials, bond_dim=args.bond_dim,
        use_optimal_ordering=USE_OPTIMAL_ORDERING, normalize_state=NORMALIZE_STATE,
        seed=args.seed, cartan_mode=args.cartan_mode, toffoli_mode=args.toffoli_mode,
    )
    params = jnp.array(ansatz.initparams)
    optimizer = optax.adam(learning_rate=ansatz.learning_rate)
    opt_state = optimizer.init(params)

    for step in range(args.maxiter):
        value, grad = ansatz._cost_vvag(params)
        value_np = np.asarray(value)
        grad_np = np.asarray(grad)
        energy_nan = bool(np.isnan(value_np).any())
        grad_nan = bool(np.isnan(grad_np).any())
        grad_norm = float(np.linalg.norm(grad_np))
        energy = float(np.real(value_np).mean())

        do_check = (step % PRINT_EVERY == 0) or energy_nan or grad_nan
        if do_check:
            qc = ansatz._circuit(params[0])
            if ansatz.normalize_state:
                qc.normalize()
            spectra = get_singular_values_per_cut(qc)
            _, min_gap = _spectrum_degeneracy_summary(spectra)
            min_gap_str = f"{min_gap:.3e}" if min_gap is not None and np.isfinite(min_gap) else "n/a"
            print(f"  step {step:4d}: energy={energy:.6f}  grad_norm={grad_norm:.4e}  "
                  f"energy_nan={energy_nan}  grad_nan={grad_nan}  min_gap={min_gap_str}", flush=True)

        if energy_nan or grad_nan:
            print(f"\n  *** NaN detected at step {step} ***")
            return

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

    print(f"\n  no NaN encountered in {args.maxiter} steps")


if __name__ == "__main__":
    main()
