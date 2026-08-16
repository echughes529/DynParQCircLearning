"""Experiment 4: how the shrinking norm degrades the SVD *backward* pass.

tensorcircuit differentiates every MPS split with its own custom VJP
(backends/jax_ops.py::jaxsvd_bwd). That rule builds

    F_ij = _safe_reciprocal(s_i^2 - s_j^2),   _safe_reciprocal(x) = x/(x^2+eps)

with a hard-coded eps = 1e-15. Two things follow, and both are scale
dependent, which is the whole point of this experiment:

  1. |F| peaks at 1/(2*sqrt(eps)) ~ 1.58e7 when |s_i^2 - s_j^2| = sqrt(eps)
     ~ 3.16e-8, and *falls back towards zero* below that. So a split whose
     singular values are too close does not blow up -- it silently gets the
     WRONG gradient, damped towards zero.
  2. eps is absolute. Rescaling the state by lambda scales every s by lambda
     and every (s_i^2 - s_j^2) by lambda^2, so letting ||psi|| decay moves the
     entire spectrum towards the broadened regime and simultaneously inflates
     the true 1/(s_i^2-s_j^2) factors by 1/lambda^2.

This script takes the spectra of every split actually performed by the real
ansatz, under norm_mode="off" (norm allowed to decay) and norm_mode="layer"
(rescaled at each layer boundary), and reports the resulting backward-pass
conditioning. Same parameters, same circuit, same truncation -- only the
running scale differs.

Usage:
    python -m src.diagnostics.norm_backward_conditioning --Lx 3 --Ly 2 --bond-dims 2 4 8
"""
import argparse
import json
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax.numpy as jnp

from src.diagnostics.normalization_study import (
    SAFE_RECIPROCAL_EPS,
    SAFE_RECIPROCAL_KNEE,
    SplitRecorder,
    make_ansatz,
)


def safe_reciprocal(x):
    return x / (x * x + SAFE_RECIPROCAL_EPS)


def split_conditioning(spectrum):
    """Backward-pass conditioning numbers for one split's spectrum."""
    s = np.asarray(spectrum, dtype=float)
    s = s[s > 0]
    if s.size < 2:
        return None
    sq = s ** 2
    iu = np.triu_indices(s.size, k=1)
    gaps = (sq[:, None] - sq[None, :])[iu]
    f_true = 1.0 / gaps
    f_used = safe_reciprocal(gaps)
    # relative error the broadening introduces into each F entry
    rel_err = np.abs(f_used - f_true) / np.abs(f_true)
    return dict(
        total_weight=float(sq.sum()),
        max_abs_F=float(np.abs(f_used).max()),
        min_abs_sq_gap=float(np.abs(gaps).min()),
        min_sv=float(s.min()),
        frac_F_distorted_1pct=float((rel_err > 0.01).mean()),
        max_F_rel_err=float(rel_err.max()),
        n_pairs=int(gaps.size),
    )


def collect(mode, bond_dim, seed, args):
    a = make_ansatz(mode, Lx=args.Lx, Ly=args.Ly, nlayers=args.nlayers,
                    howoften_toreset=7, reset_layers=[args.nlayers - 1],
                    trials=1, bond_dim=bond_dim, seed=seed,
                    use_optimal_ordering=True, cartan_mode=args.cartan_mode,
                    toffoli_mode=args.toffoli_mode)
    params = jnp.array(a.initparams[0])
    rec = SplitRecorder(capture_spectra=True)
    with rec.patch():
        qc = a._circuit(params)
    final_norm_sq = float(np.abs(np.asarray(qc.get_norm()))) ** 2

    per_split = [c for c in (split_conditioning(s) for s in rec.spectra) if c]
    if not per_split:
        return None
    max_F = np.array([c["max_abs_F"] for c in per_split])
    weight = np.array([c["total_weight"] for c in per_split])
    return dict(
        mode=mode, bond_dim=bond_dim, seed=seed,
        n_splits=len(per_split), final_norm_sq=final_norm_sq,
        # the scale the splits actually run at, mid-circuit
        min_split_weight=float(weight.min()), median_split_weight=float(np.median(weight)),
        max_abs_F_overall=float(max_F.max()),
        median_max_abs_F=float(np.median(max_F)),
        # multiplicative amplification the reverse chain has to carry:
        # sum of logs of the worst F factor at every split
        log10_amplification=float(np.log10(max_F).sum()),
        n_splits_at_broadening_peak=int((max_F > 0.5 / (2 * SAFE_RECIPROCAL_KNEE)).sum()),
        min_abs_sq_gap=float(min(c["min_abs_sq_gap"] for c in per_split)),
        min_sv=float(min(c["min_sv"] for c in per_split)),
        max_frac_F_distorted=float(max(c["frac_F_distorted_1pct"] for c in per_split)),
        mean_frac_F_distorted=float(np.mean([c["frac_F_distorted_1pct"] for c in per_split])),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--bond-dims", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--cartan-mode", default="fused")
    p.add_argument("--toffoli-mode", default="direct")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"  broadening knee |s_i^2 - s_j^2| = {SAFE_RECIPROCAL_KNEE:.3e}, "
          f"max|F| = {1/(2*SAFE_RECIPROCAL_KNEE):.3e}")
    rows = []
    for bond_dim in args.bond_dims:
        for seed in args.seeds:
            for mode in ("off", "layer"):
                r = collect(mode, bond_dim, seed, args)
                if r is None:
                    continue
                rows.append(r)
                print(f"  bd={bond_dim:>3} seed={seed} {mode:<5}: "
                      f"||psi||^2={r['final_norm_sq']:.6f}  "
                      f"min_split_weight={r['min_split_weight']:.6f}  "
                      f"max|F|={r['max_abs_F_overall']:.3e}  "
                      f"median max|F|={r['median_max_abs_F']:.3e}  "
                      f"sum log10|F|={r['log10_amplification']:+.1f}  "
                      f"min_sq_gap={r['min_abs_sq_gap']:.2e}  "
                      f"F distorted>1%: mean {r['mean_frac_F_distorted']:.2%} "
                      f"max {r['max_frac_F_distorted']:.2%}", flush=True)
            print(flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1, default=float)
        print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
