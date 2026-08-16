"""Re-run the exact config of job 3613124 and capture its first non-finite step.

Run 3613124 (3x3 toric code, bond_dim=96, 20 trials, lr=0.01, seed 72314,
optimal ordering, cartan fused, toffoli direct, normalize_state=True, A40 GPU)
displayed E_min = -11.9695 at step 210 and nan at step 220, then stayed nan
forever because find_gs.optimize applies Adam updates unguarded. That job saved
nothing (track_params=False, results written only after maxiter), so the
parameters that produced the first non-finite value/gradient exist nowhere on
disk. This script recreates them: the parameter init is seeded (72314) and the
ordering search is internally seeded (_rng.seed(42)), so the trajectory is
reproducible up to GPU arithmetic.

At the first step where any trial's value or gradient is non-finite it stops
training and answers, in-process, the questions the investigation needs:

  1. FORWARD OR BACKWARD?  Recompute the forward-only energy at the culprit
     params (fresh jit, single trial). Finite energy + non-finite gradient
     means the blowup is in jaxsvd_bwd (the _safe_reciprocal amplification);
     a non-finite energy means the forward SVD itself failed, the GPU cousin
     of the CPU thread knife edge ([[project-nan-svd-thread-knife-edge]]).

  2. WHAT DID THE SPECTRA LOOK LIKE?  Eagerly rebuild the culprit circuit under
     SplitRecorder at the culprit params and at the last-good params, recording
     every split's full pre-truncation spectrum: how many kept singular values
     sit inside the amplification window around sqrt(1e-15) ~ 3.16e-8, and the
     minimal kept-pair and cut gaps that jaxsvd_bwd's F matrix inverts.

  3. DOES RAISING EPSILON RESCUE IT?  PATCH A (svd_epsilon_patch.svd_epsilon)
     at eps in {1e-12, 1e-10, 1e-8}: recompute the same single-trial gradient;
     record finiteness, norm, and pairwise cosine similarity across arms.

Everything (param history, opt state, trajectories, spectra, verdicts) is
saved to $DPQC_OUTDIR/nan_capture_bd96.npz + a JSON summary printed to stdout.

Submit with run_nan_capture_bd96.sh (needs the A40 the original ran on).
Expected time to event: ~220 steps x ~60 s = ~4 h; hard cap 400 steps.
"""
import contextlib
import io
import json
import os
import sys
import time
import types
from collections import deque

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import optax

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.normalization_study import SplitRecorder
from src.diagnostics.svd_epsilon_patch import svd_epsilon

# Defaults reproduce job 3613124 (bd=96, first bad step ~211-220). Overrides:
# DPQC_CAP_BOND_DIM=70 DPQC_CAP_SEED=93074 DPQC_CAP_NITER=60 reproduces job
# 3613161, whose first bad step was already within steps 1-10.
SEED = int(os.environ.get("DPQC_CAP_SEED", 72314))
BOND_DIM = int(os.environ.get("DPQC_CAP_BOND_DIM", 96))
NITER_MAX = int(os.environ.get("DPQC_CAP_NITER", 400))
HISTORY = 30
EPS_ARMS = (1e-12, 1e-10, 1e-8)
OUTDIR = os.environ.get("DPQC_OUTDIR", "outputs")
TAG = f"bd{BOND_DIM}_seed{SEED}"


def build_ansatz():
    with contextlib.redirect_stdout(io.StringIO()):
        return ToricCodeAnsatz(
            Lx=3, Ly=3, nlayers=2, howoften_toreset=7, h=0.0,
            use_prob_resets_ansatz=True, prob_reset_direction=1, reset_layers=[1],
            unitary=True, bond_dim=BOND_DIM, use_optimal_ordering=True,
            cartan_mode="fused", toffoli_mode="direct",
            trials=20, maxiter=NITER_MAX, learning_rate=0.01,
            sparse=False, use_mps=True, normalize_state=True, seed=SEED,
        )


def spectra_summary(ansatz, params_one_trial):
    """Per-split spectra of the eagerly rebuilt circuit, summarized."""
    rec = SplitRecorder(capture_spectra=True)
    with rec.patch():
        ansatz._circuit(jnp.asarray(params_one_trial))
    out = {"n_splits": len(rec.spectra), "splits": []}
    knee = np.sqrt(1e-15)
    for spec, kept in zip(rec.spectra, rec.n_kept):
        s = np.asarray(spec, dtype=float)
        sk, sd = s[:kept], s[kept:]
        finite = bool(np.isfinite(s).all())
        entry = {"kept": int(kept), "shape": None, "finite": finite}
        if sk.size:
            entry["s_max"] = float(sk.max())
            entry["s_min_kept"] = float(sk.min())
            entry["n_in_knee_decade"] = int(((sk > knee / 10) & (sk < knee * 10)).sum())
        if sk.size >= 2:
            sq = np.sort(sk ** 2)
            entry["min_kept_gap"] = float(np.diff(sq).min())
        if sk.size and sd.size:
            entry["cut_gap"] = float(abs(sk[-1] ** 2 - sd[0] ** 2))
            entry["s_first_discarded"] = float(sd[0])
        out["splits"].append(entry)
    return out, [np.asarray(x, dtype=float) for x in rec.spectra]


def cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"nan_capture[{TAG}]: cap={NITER_MAX} steps, "
          f"devices={jax.devices()}", flush=True)
    ansatz = build_ansatz()
    params = jnp.array(ansatz.initparams)
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)

    energies = np.full((NITER_MAX, 20), np.nan)
    grad_norms = np.full((NITER_MAX, 20), np.nan)
    history = deque(maxlen=HISTORY)

    first_bad, bad_mask, culprits = None, None, []
    t0 = time.time()
    for i in range(NITER_MAX):
        value, gradient = ansatz._cost_vvag(params)
        v, g = np.asarray(value), np.asarray(gradient)
        energies[i], grad_norms[i] = v, np.linalg.norm(g, axis=-1)
        history.append((i, np.asarray(params)))

        bad = ~(np.isfinite(v) & np.isfinite(g).all(axis=-1))
        if i % 10 == 0 or bad.any():
            print(f"step {i:4d} [{time.time()-t0:7.0f}s] E_min={np.nanmin(v):+.9f} "
                  f"|g|max={grad_norms[i].max():.3e} bad={np.where(bad)[0].tolist()}",
                  flush=True)
        if bad.any():
            first_bad, bad_mask = i, bad
            culprits = np.where(bad)[0].tolist()
            break

        updates, opt_state = optimizer.update(gradient, opt_state)
        params = optax.apply_updates(params, updates)

    payload = {
        "energies": energies, "grad_norms": grad_norms,
        "first_bad": np.array(-1 if first_bad is None else first_bad),
        "params_history_steps": np.array([s for s, _ in history]),
        "params_history": np.stack([p for _, p in history]) if history else np.zeros(0),
    }
    verdicts = {"first_bad": first_bad, "culprits": culprits, "trials": {}}

    if first_bad is None:
        print("no non-finite step within the cap; saving trajectories only", flush=True)
    else:
        payload["bad_value_mask"] = ~np.isfinite(np.asarray(value))
        payload["bad_grad_mask"] = ~np.isfinite(np.asarray(gradient)).all(axis=-1)
        payload["final_value"] = np.asarray(value)
        payload["final_gradient"] = np.asarray(gradient)
        leaves = jax.tree_util.tree_leaves(opt_state)
        for k, leaf in enumerate(leaves):
            payload[f"opt_state_leaf{k}"] = np.asarray(leaf)

        p_np = np.asarray(params)
        for t in culprits[:3]:
            tag = f"trial{t}"
            print(f"\n=== classification for {tag} ===", flush=True)
            pt = jnp.asarray(p_np[t])
            tr = {}

            fwd = jax.jit(ansatz.energy_from_params)
            e = float(np.asarray(fwd(pt)))
            tr["forward_energy"] = e
            tr["forward_finite"] = bool(np.isfinite(e))
            print(f"forward-only energy: {e}  (finite: {tr['forward_finite']})", flush=True)

            vg = jax.jit(jax.value_and_grad(ansatz.energy_from_params))
            e2, g2 = vg(pt)
            g2 = np.asarray(g2)
            tr["grad_stock_finite"] = bool(np.isfinite(g2).all())
            tr["grad_stock_norm"] = float(np.linalg.norm(g2))
            tr["grad_stock_nonfinite_count"] = int((~np.isfinite(g2)).sum())
            payload[f"{tag}_grad_stock"] = g2
            print(f"stock grad: finite={tr['grad_stock_finite']} "
                  f"|g|={tr['grad_stock_norm']:.3e}", flush=True)

            summ, spectra = spectra_summary(ansatz, p_np[t])
            tr["spectra_at_fail"] = summ
            for k, spec in enumerate(spectra):
                payload[f"{tag}_spec_fail_{k}"] = spec
            if len(history) >= 2:
                step_prev, p_prev = history[-2]
                summ_prev, _ = spectra_summary(ansatz, p_prev[t])
                tr["spectra_last_good_step"] = step_prev
                tr["spectra_last_good"] = summ_prev

            arms = {}
            for eps in EPS_ARMS:
                with svd_epsilon(eps):
                    vg_p = jax.jit(jax.value_and_grad(ansatz.energy_from_params))
                    _, gp = vg_p(pt)
                gp = np.asarray(gp)
                arms[eps] = gp
                payload[f"{tag}_grad_eps{eps:g}"] = gp
                tr[f"eps{eps:g}_finite"] = bool(np.isfinite(gp).all())
                tr[f"eps{eps:g}_norm"] = float(np.linalg.norm(gp))
                print(f"eps={eps:g}: finite={tr[f'eps{eps:g}_finite']} "
                      f"|g|={tr[f'eps{eps:g}_norm']:.3e}", flush=True)
            eps_list = list(arms)
            for a in range(len(eps_list)):
                for b in range(a + 1, len(eps_list)):
                    tr[f"cos_eps{eps_list[a]:g}_eps{eps_list[b]:g}"] = cos(
                        arms[eps_list[a]], arms[eps_list[b]])
            verdicts["trials"][tag] = tr

    out = os.path.join(OUTDIR, f"nan_capture_{TAG}.npz")
    np.savez_compressed(out, **payload)
    with open(os.path.join(OUTDIR, f"nan_capture_{TAG}_verdicts.json"), "w") as f:
        json.dump(verdicts, f, indent=2, default=str)
    print(f"\nsaved {out}")
    print(json.dumps(verdicts, indent=2, default=str))


if __name__ == "__main__":
    main()
