"""Does forcing the QR (non-batched-Jacobi) SVD kernel fix the batched forward NaN?

nan_capture (bd=70, seed 93074, step 6, trial 16) + batch_eps_rescue
established: the vmapped 20-trial forward produces value=nan for trial 16 on an
A40 while the identical parameters give a finite energy unbatched (and on CPU).
Under vmap every split's ``jnp.linalg.svd`` becomes a batch-20 cuSOLVER call --
the batched Jacobi kernel -- and at that step all 21 truncating splits carry
EXACTLY degenerate singular values straddling the truncation cut (cut gap = 0.0
bit-for-bit), the classic input that breaks Jacobi convergence for one batch
element while the unbatched path survives.

jax.lax.linalg.svd exposes the kernel choice. This script rebuilds the
identical batched value_and_grad at the captured culprit parameters with:

  arm 1: stock adaware_svd (jnp.linalg.svd, DEFAULT algorithm) - expect NaN
  arm 2: forward switched to algorithm=QR (same custom VJP backward)
  arm 3: QR forward + _safe_reciprocal eps=1e-10 (candidate production combo)

The forward swap monkey-patches ``jax_ops.adaware_svd_jit``: _svd_jax re-reads
that attribute from the module on every call ("from .jax_ops import
adaware_svd_jit as adaware_svd" inside the function body), so rebinding the
module attribute redirects every MPS split, and clearing JAX caches forces the
next trace to pick it up. Timing per arm is reported because QR has no batched
cuSOLVER variant (XLA loops over the batch), so this measures the speed cost.

Env: DPQC_RESCUE_NPZ, DPQC_CAP_BOND_DIM, DPQC_CAP_SEED, DPQC_RESCUE_TRIAL
"""
import contextlib
import io
import json
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
from jax.lax.linalg import svd as lax_svd, SvdAlgorithm
import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz
from tensorcircuit.backends import jax_ops
from src.diagnostics.svd_epsilon_patch import svd_epsilon

K = tc.backend

NPZ = os.environ.get("DPQC_RESCUE_NPZ")  # required by main(); qr_forward importers don't need it
BOND_DIM = int(os.environ.get("DPQC_CAP_BOND_DIM", 70))
SEED = int(os.environ.get("DPQC_CAP_SEED", 93074))
TRIAL = int(os.environ.get("DPQC_RESCUE_TRIAL", 16))
OUTDIR = os.environ.get("DPQC_OUTDIR", "outputs")


@jax.custom_vjp
def qr_adaware_svd(A):
    u, s, vh = lax_svd(A, full_matrices=False, algorithm=SvdAlgorithm.QR)
    return (u, s, vh)


def _qr_fwd(A):
    u, s, v = qr_adaware_svd(A)
    return (u, s, v), (u, s, v)


qr_adaware_svd.defvjp(_qr_fwd, jax_ops.jaxsvd_bwd)


@contextlib.contextmanager
def qr_forward():
    original = jax_ops.adaware_svd_jit
    jax_ops.adaware_svd_jit = jax.jit(qr_adaware_svd)
    jax.clear_caches()
    try:
        yield
    finally:
        jax_ops.adaware_svd_jit = original
        jax.clear_caches()


def build_ansatz():
    with contextlib.redirect_stdout(io.StringIO()):
        return ToricCodeAnsatz(
            Lx=3, Ly=3, nlayers=2, howoften_toreset=7, h=0.0,
            use_prob_resets_ansatz=True, prob_reset_direction=1, reset_layers=[1],
            unitary=True, bond_dim=BOND_DIM, use_optimal_ordering=True,
            cartan_mode="fused", toffoli_mode="direct",
            trials=20, maxiter=10, learning_rate=0.01,
            sparse=False, use_mps=True, normalize_state=True, seed=SEED,
        )


def run_arm(ansatz, params, label, ctxs):
    t0 = time.time()
    with contextlib.ExitStack() as stack:
        for c in ctxs:
            stack.enter_context(c)
        vvag = K.jit(K.vmap(K.value_and_grad(ansatz.energy_from_params, argnums=0),
                            vectorized_argnums=0))
        value, grad = vvag(params)
        v, g = np.asarray(value), np.asarray(grad)
    res = {
        "label": label,
        "seconds": round(time.time() - t0, 1),
        "n_bad_trials": int((~(np.isfinite(v) & np.isfinite(g).all(axis=-1))).sum()),
        "bad_trials": [int(i) for i in
                       np.where(~(np.isfinite(v) & np.isfinite(g).all(axis=-1)))[0]],
        "culprit_value": float(v[TRIAL]),
        "culprit_grad_norm": float(np.linalg.norm(g[TRIAL])),
        "grad_norm_max_finite": float(np.nanmax(
            np.where(np.isfinite(g).all(axis=-1),
                     np.linalg.norm(np.where(np.isfinite(g), g, 0), axis=1), np.nan))),
    }
    print(json.dumps(res), flush=True)
    return res


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    params = jnp.asarray(np.load(NPZ)["params_history"][-1])
    print(f"fwd_svd_algorithm_test: bd={BOND_DIM} trial={TRIAL} "
          f"devices={jax.devices()}", flush=True)
    ansatz = build_ansatz()

    results = [
        run_arm(ansatz, params, "stock_default_alg", []),
        run_arm(ansatz, params, "qr_forward", [qr_forward()]),
        run_arm(ansatz, params, "qr_forward_eps1e-10", [qr_forward(), svd_epsilon(1e-10)]),
    ]
    with open(os.path.join(OUTDIR, f"fwd_svd_algorithm_bd{BOND_DIM}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
