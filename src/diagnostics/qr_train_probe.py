"""Train the failing bd=70 config from scratch with the QR SVD forward: does it
survive past the original death step, and what does QR cost per step?

fwd_svd_algorithm_test.py showed that at the captured culprit parameters the
batched-Jacobi forward NaNs trial 16 deterministically while algorithm=QR gives
0 bad trials AND drops the worst finite gradient norm from 38.7 to 2.5. This
probe runs real training (guarded Adam, same as find_gs.py now) from the same
seed-93074 init for DPQC_CAP_NITER steps under qr_forward(), reporting per-step
wall time and finiteness, so the fix's steady-state cost is measured on the
identical workload that previously died within 10 steps.

If DPQC_SVD_FWD_ALG=qr is set, the production import-time hook in
generate_ansatz.py provides the QR forward and this script adds nothing on
top, so a run with that env var validates the exact path production uses;
otherwise the qr_forward() context manager is applied here.

Env: DPQC_CAP_BOND_DIM (70), DPQC_CAP_SEED (93074), DPQC_CAP_NITER (30),
     DPQC_SVD_FWD_ALG (optional, see above), DPQC_OUTDIR (npz destination)
"""
import contextlib
import io
import os
import sys
import time
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import optax

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.fwd_svd_algorithm_test import qr_forward

BOND_DIM = int(os.environ.get("DPQC_CAP_BOND_DIM", 70))
SEED = int(os.environ.get("DPQC_CAP_SEED", 93074))
NITER = int(os.environ.get("DPQC_CAP_NITER", 30))
OUTDIR = os.environ.get("DPQC_OUTDIR", "outputs")
ENV_HOOK = bool(os.environ.get("DPQC_SVD_FWD_ALG"))


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz = ToricCodeAnsatz(
            Lx=3, Ly=3, nlayers=2, howoften_toreset=7, h=0.0,
            use_prob_resets_ansatz=True, prob_reset_direction=1, reset_layers=[1],
            unitary=True, bond_dim=BOND_DIM, use_optimal_ordering=True,
            cartan_mode="fused", toffoli_mode="direct",
            trials=20, maxiter=NITER, learning_rate=0.01,
            sparse=False, use_mps=True, normalize_state=True, seed=SEED,
        )
    print(f"qr_train_probe: bd={BOND_DIM} seed={SEED} niter={NITER} "
          f"qr_via={'env hook (DPQC_SVD_FWD_ALG)' if ENV_HOOK else 'qr_forward() context'} "
          f"devices={jax.devices()}", flush=True)

    params = jnp.array(ansatz.initparams)
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    energies = np.full((NITER, ansatz.trials), np.nan)
    grad_norms = np.full((NITER, ansatz.trials), np.nan)

    with (contextlib.nullcontext() if ENV_HOOK else qr_forward()):
        for i in range(NITER):
            t0 = time.time()
            value, gradient = ansatz._cost_vvag(params)
            v, g = np.asarray(value), np.asarray(gradient)
            bad = ~(np.isfinite(v) & np.isfinite(g).all(axis=-1))
            gn = np.linalg.norm(np.where(np.isfinite(g), g, 0), axis=-1)
            energies[i], grad_norms[i] = v, gn
            print(f"step {i:3d} [{time.time()-t0:6.1f}s] E_min={np.nanmin(v):+.9f} "
                  f"|g|max={gn.max():.3e} bad={np.where(bad)[0].tolist()}", flush=True)

            finite = jnp.isfinite(gradient).all(axis=-1) & jnp.isfinite(value)
            safe_g = jnp.where(finite[:, None], gradient, 0.0)
            updates, opt_state = optimizer.update(safe_g, opt_state)
            updates = jnp.where(finite[:, None], updates, 0.0)
            params = optax.apply_updates(params, updates)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"qr_train_bd{BOND_DIM}_seed{SEED}.npz")
    np.savez_compressed(out, energies=energies, grad_norms=grad_norms,
                        final_params=np.asarray(params))
    print(f"final per-trial energies: {np.array2string(energies[-1], precision=8)}")
    print(f"saved {out}\ndone", flush=True)


if __name__ == "__main__":
    main()
