"""Does training with bond_dim > true Schmidt rank blow up the SVD backward pass,
and does raising TensorCircuit's _safe_reciprocal epsilon prevent it?

Motivated by run 3613124 (3x3, bond_dim=96, seed 72314), which went NaN between
steps 211 and 220 after reaching E = -11.9695 (dE ~ 0.03 from the exact -12).
At 3x3 with optimal ordering the converged state needs bond dimension 64
([[project-mps-ordering-bond-dim]]), so that run keeps 96 - 64 = 32 singular
values that must DECAY TO ZERO as training converges. jaxsvd_bwd amplifies
cotangents by up to 1/(2*sqrt(eps)) ~ 1.6e7 whenever a kept singular value (via
Sinv) or a pair gap |s_i^2 - s_j^2| (via F) transits the broadening knee at
sqrt(eps) ~ 3.16e-8, so every surplus value is guaranteed to pass through the
maximum-amplification window on its way down. Several splits transiting at once
multiply up to inf/NaN, and find_gs.py applies the Adam update unguarded, so one
bad gradient kills the trial permanently.

This script reproduces that regime at login-node scale: 2x2 toric code (true
converged rank 16, [[project-mps-lossless-bond-dims]]) trained at bond_dim=24,
so 8 surplus values decay through the knee near convergence. Arms:

  * control: stock eps = 1e-15
  * rescue arms: eps raised via PATCH A (svd_epsilon_patch.svd_epsilon), which
    caps the per-split amplification at 1/(2*sqrt(eps))

Per step it records per-trial energy and gradient norm; every --probe-every
steps it eagerly rebuilds the argmin trial's circuit under SplitRecorder and
records, per split, the kept spectrum tail and the pairwise-gap structure the
backward F matrix sees. The prediction being tested: control gradient-norm
spikes line up with kept singular values inside the [sqrt(eps)/10, 10*sqrt(eps)]
window, and the spike ceiling scales like 1/(2*sqrt(eps)) across arms.

Usage:
    python -m src.diagnostics.svd_eps_blowup_training \
        --bond-dim 24 --trials 8 --maxiter 1200 --eps-arms 1e-10 1e-8
"""
import argparse
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

import jax.numpy as jnp
import optax

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.normalization_study import SplitRecorder
from src.diagnostics.svd_epsilon_patch import svd_epsilon


def make_run_ansatz(args, seed):
    """The failing run's ansatz flags, at --Lx x --Ly and the requested cap."""
    with contextlib.redirect_stdout(io.StringIO()):
        return ToricCodeAnsatz(
            Lx=args.Lx, Ly=args.Ly, nlayers=2, howoften_toreset=7, h=0.0,
            use_prob_resets_ansatz=True, prob_reset_direction=1, reset_layers=[1],
            unitary=True, bond_dim=args.bond_dim, use_optimal_ordering=True,
            cartan_mode="fused", toffoli_mode="direct",
            trials=args.trials, maxiter=args.maxiter, learning_rate=args.lr,
            sparse=False, use_mps=True, normalize_state=True, seed=seed,
        )


def probe_spectra(ansatz, params_one_trial):
    """Eagerly rebuild one trial's circuit; summarize what jaxsvd_bwd would see.

    Returns per-probe scalars plus the per-split kept tails, all computed from
    the full pre-truncation spectra that SplitRecorder captures.
    """
    rec = SplitRecorder(capture_spectra=True)
    with rec.patch():
        ansatz._circuit(jnp.asarray(params_one_trial))

    min_kept_s = np.inf          # smallest kept singular value above the zero floor
    min_kept_gap = np.inf        # smallest |s_i^2-s_j^2| over those kept pairs
    min_cut_gap = np.inf         # smallest |s_kept^2 - s_disc^2| across a cut
    n_knee = 0                   # kept values inside [knee/10, knee*10], knee=3.16e-8
    n_trunc = 0                  # splits that actually discarded weight
    max_disc_w = 0.0             # largest discarded weight fraction at any split
    knee_lo, knee_hi = np.sqrt(1e-15) / 10, np.sqrt(1e-15) * 10
    zero_floor = 1e-14           # exact numerical zeros are harmless in x/(x^2+eps)
    tails = []
    for spec, kept in zip(rec.spectra, rec.n_kept):
        s = np.asarray(spec, dtype=float)
        sk, sd = s[:kept], s[kept:]
        sk_nz = sk[sk > zero_floor]
        if sd.size:
            n_trunc += 1
            tot = float((s ** 2).sum())
            if tot > 0:
                max_disc_w = max(max_disc_w, float((sd ** 2).sum()) / tot)
        if sk_nz.size:
            min_kept_s = min(min_kept_s, float(sk_nz.min()))
            n_knee += int(((sk_nz >= knee_lo) & (sk_nz <= knee_hi)).sum())
            tails.append(sk_nz[-min(4, sk_nz.size):])
        if sk_nz.size >= 2:
            sq = sk_nz ** 2
            gaps = np.abs(sq[:, None] - sq[None, :])
            iu = np.triu_indices(sk_nz.size, k=1)
            if iu[0].size:
                min_kept_gap = min(min_kept_gap, float(gaps[iu].min()))
        if sk.size and sd.size:
            min_cut_gap = min(min_cut_gap, float(np.abs(sk[-1] ** 2 - sd[0] ** 2)))
    return {
        "min_kept_s": min_kept_s,
        "min_kept_gap": min_kept_gap,
        "min_cut_gap": min_cut_gap,
        "n_kept_in_knee_window": n_knee,
        "n_truncating_splits": n_trunc,
        "max_discarded_weight": max_disc_w,
        "n_splits": len(rec.spectra),
        "tails": tails,
    }


def run_arm(args, eps, seed):
    """Train one arm; eps=None means the stock 1e-15 rule."""
    label = "stock(1e-15)" if eps is None else f"eps={eps:g}"
    ansatz = make_run_ansatz(args, seed)
    params = jnp.array(ansatz.initparams)
    optimizer = optax.adam(learning_rate=args.lr)
    opt_state = optimizer.init(params)

    energies = np.full((args.maxiter, args.trials), np.nan)
    grad_norms = np.full((args.maxiter, args.trials), np.nan)
    probes = {}
    first_bad = None

    ctx = svd_epsilon(eps) if eps is not None else contextlib.nullcontext()
    t0 = time.time()
    with ctx:
        for i in range(args.maxiter):
            value, gradient = ansatz._cost_vvag(params)
            v = np.asarray(value)
            g = np.asarray(gradient)
            energies[i] = v
            grad_norms[i] = np.linalg.norm(g, axis=-1)

            bad = ~(np.isfinite(v) & np.isfinite(g).all(axis=-1))
            if bad.any() and first_bad is None:
                first_bad = i
                print(f"  [{label}] step {i}: NON-FINITE trials {np.where(bad)[0].tolist()}"
                      f" | energies {v}", flush=True)
                # capture the spectra the failing trial's forward produced
                culprit = int(np.where(bad)[0][0])
                probes[i] = probe_spectra(ansatz, np.asarray(params)[culprit])
                break

            if i % args.probe_every == 0:
                tr = int(np.argmin(v))
                p = probe_spectra(ansatz, np.asarray(params)[tr])
                probes[i] = p
                print(f"  [{label}] step {i:4d} E_min={v.min():+.6f} "
                      f"|g|max={grad_norms[i].max():.3e} "
                      f"min_kept_s={p['min_kept_s']:.3e} "
                      f"min_kept_gap={p['min_kept_gap']:.3e} "
                      f"cut_gap={p['min_cut_gap']:.3e} "
                      f"in_knee={p['n_kept_in_knee_window']} "
                      f"n_trunc={p['n_truncating_splits']} "
                      f"disc_w={p['max_discarded_weight']:.2e}", flush=True)

            updates, opt_state = optimizer.update(gradient, opt_state)
            params = optax.apply_updates(params, updates)

    mins = time.time() - t0
    print(f"  [{label}] done in {mins:.0f}s; first_bad={first_bad}; "
          f"best E={np.nanmin(energies):+.8f}", flush=True)
    return {
        "label": label, "eps": eps, "first_bad": first_bad,
        "energies": energies, "grad_norms": grad_norms,
        "probes": probes, "seconds": mins,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=int, default=2)
    ap.add_argument("--Ly", type=int, default=2)
    ap.add_argument("--bond-dim", type=int, default=24)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--maxiter", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--probe-every", type=int, default=25)
    ap.add_argument("--eps-arms", type=float, nargs="*", default=[1e-10, 1e-8])
    ap.add_argument("--outdir", default=os.environ.get("DPQC_OUTDIR", "outputs"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"svd_eps_blowup_training: {args.Lx}x{args.Ly}, bond_dim={args.bond_dim}, "
          f"trials={args.trials}, maxiter={args.maxiter}, seed={args.seed}", flush=True)

    results = [run_arm(args, None, args.seed)]
    for eps in args.eps_arms:
        results.append(run_arm(args, float(eps), args.seed))

    out = os.path.join(args.outdir,
                       f"svd_eps_blowup_{args.Lx}x{args.Ly}_bd{args.bond_dim}_seed{args.seed}.npz")
    payload = {}
    summary = []
    for r in results:
        key = r["label"].replace("(", "_").replace(")", "").replace("=", "_")
        payload[f"{key}_energies"] = r["energies"]
        payload[f"{key}_grad_norms"] = r["grad_norms"]
        payload[f"{key}_probe_steps"] = np.array(sorted(r["probes"]))
        for stat in ("min_kept_s", "min_kept_gap", "min_cut_gap", "n_kept_in_knee_window",
                     "n_truncating_splits", "max_discarded_weight"):
            payload[f"{key}_{stat}"] = np.array(
                [r["probes"][s][stat] for s in sorted(r["probes"])])
        summary.append({
            "label": r["label"], "first_bad": r["first_bad"],
            "best_energy": float(np.nanmin(r["energies"])),
            "max_grad_norm": float(np.nanmax(r["grad_norms"])),
            "seconds": r["seconds"],
        })
    np.savez_compressed(out, **payload)
    print(f"\nsaved {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
