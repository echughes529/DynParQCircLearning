"""Batch-level epsilon rescue test at captured culprit parameters.

nan_capture_bd96.py established (bd=70, seed 93074, step 6): the vmapped
20-trial value_and_grad on the A40 produces a NaN gradient for trial 16, while
the single-trial forward AND single-trial gradient at the same parameters are
finite. So the failure lives in the batched GPU backward program, and any
rescue must be demonstrated on that exact program shape.

This script loads the captured failing-step parameter batch and, for each arm
(stock eps=1e-15, then DPQC-patched eps values), builds the identical
K.jit(K.vmap(K.value_and_grad(...))) helper the training loop uses and runs it
once. The stock arm runs twice to establish whether the batch NaN is even
deterministic on this hardware. A vmap-of-one arm over just the culprit trial
separates "batched program" from "batch content" effects.

Env:
  DPQC_RESCUE_NPZ      path to nan_capture_*.npz (needs params_history)
  DPQC_CAP_BOND_DIM    ansatz bond_dim (default 70)
  DPQC_CAP_SEED        ansatz seed (default 93074)
  DPQC_RESCUE_TRIAL    culprit trial index (default 16)
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
import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.svd_epsilon_patch import svd_epsilon

K = tc.backend

NPZ = os.environ["DPQC_RESCUE_NPZ"]
BOND_DIM = int(os.environ.get("DPQC_CAP_BOND_DIM", 70))
SEED = int(os.environ.get("DPQC_CAP_SEED", 93074))
TRIAL = int(os.environ.get("DPQC_RESCUE_TRIAL", 16))
OUTDIR = os.environ.get("DPQC_OUTDIR", "outputs")


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


def batch_vvag(ansatz):
    """Exactly _make_jit_helpers' noiseless cost_vvag, built fresh."""
    return K.jit(K.vmap(K.value_and_grad(ansatz.energy_from_params, argnums=0),
                        vectorized_argnums=0))


def run_arm(ansatz, params, label, eps):
    ctx = svd_epsilon(eps) if eps is not None else contextlib.nullcontext()
    t0 = time.time()
    with ctx:
        vvag = batch_vvag(ansatz)
        value, grad = vvag(params)
        v, g = np.asarray(value), np.asarray(grad)
    res = {
        "label": label,
        "seconds": round(time.time() - t0, 1),
        "value_finite_per_trial": np.isfinite(v).tolist(),
        "grad_finite_per_trial": np.isfinite(g).all(axis=-1).tolist(),
        "culprit_value": float(v[TRIAL]) if TRIAL < v.shape[0] else None,
        "culprit_grad_norm": float(np.linalg.norm(g[TRIAL])) if TRIAL < g.shape[0] else None,
        "n_bad_trials": int((~(np.isfinite(v) & np.isfinite(g).all(axis=-1))).sum()),
    }
    print(json.dumps(res), flush=True)
    return res, g


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = np.load(NPZ)
    params = jnp.asarray(data["params_history"][-1])
    step = int(data["params_history_steps"][-1])
    print(f"batch_eps_rescue: bd={BOND_DIM} seed={SEED} trial={TRIAL} "
          f"failing step={step} params={params.shape} devices={jax.devices()}",
          flush=True)
    ansatz = build_ansatz()

    results = []
    grads = {}
    for label, eps in [("stock_run1", None), ("stock_run2", None),
                       ("eps1e-12", 1e-12), ("eps1e-10", 1e-10), ("eps1e-8", 1e-8)]:
        if label == "stock_run2":
            jax.clear_caches()  # force retrace/recompile for the determinism check
        r, g = run_arm(ansatz, params, label, eps)
        results.append(r)
        grads[label] = g

    # batched program with only the culprit trial in it
    r1, _ = run_arm(ansatz, params[TRIAL:TRIAL + 1], "stock_vmap_of_one", None)
    results.append(r1)

    np.savez_compressed(os.path.join(OUTDIR, f"batch_eps_rescue_bd{BOND_DIM}.npz"),
                        **{f"grad_{k}": v for k, v in grads.items()})
    with open(os.path.join(OUTDIR, f"batch_eps_rescue_bd{BOND_DIM}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
