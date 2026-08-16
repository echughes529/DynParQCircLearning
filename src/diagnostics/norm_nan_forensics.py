"""Experiment 6: locate the exact split that first produces a NaN.

The forward pass alone -- no optimizer, no autodiff -- can produce a NaN state
at 3x3 / bond_dim=64 / natural ordering, which is the configuration the
historical un-normalized runs used (logs/2026-08-13_17-01-33_not_normed_
3x3_bd64_3606421 printed "state norm before energy eval: nan" at the very
first optimization step). This script walks every SVD in that circuit and
reports:

  * the index of the first split whose OUTPUT is non-finite,
  * whether that split's INPUT was already non-finite (NaN inherited from
    earlier) or finite (the SVD itself failed to converge),
  * the scale and conditioning of that input matrix,

and then rebuilds the identical circuit from the identical parameters under
each normalization regime, so "does periodic normalization prevent this
specific NaN" is answered by direct comparison rather than by inference.

Usage:
    python -m src.diagnostics.norm_nan_forensics --Lx 3 --Ly 3 --bond-dim 64 \
        --seed 4876 --natural-ordering --cartan-mode separate --toffoli-mode decomposed
"""
import argparse
import json
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax.numpy as jnp

from src.diagnostics.normalization_study import NORM_MODES, make_ansatz


class NanTracer:
    """Wrap the backend SVD and record the first non-finite result."""

    def __init__(self):
        self.records = []
        self.first_bad = None
        self.n_calls = 0

    def patch(self):
        from tensornetwork.backends.jax import jax_backend as _tn_jax
        self._orig = _tn_jax.JaxBackend.svd
        tracer = self
        original = self._orig

        def wrapped(self_backend, tensor, *a, **kw):
            idx = tracer.n_calls
            tracer.n_calls += 1
            t = np.asarray(tensor)
            in_finite = bool(np.isfinite(t).all())
            out = original(self_backend, tensor, *a, **kw)
            u, s, vh, s_rest = out
            s_np = np.abs(np.asarray(s)).astype(float)
            rest_np = np.abs(np.asarray(s_rest)).astype(float)
            out_finite = all(bool(np.isfinite(np.asarray(x)).all())
                             for x in (u, s, vh, s_rest))
            if tracer.first_bad is None and not out_finite:
                full = np.concatenate([s_np, rest_np])
                good = full[np.isfinite(full) & (full > 0)]
                tracer.first_bad = dict(
                    split_index=idx,
                    input_was_finite=in_finite,
                    input_shape=list(t.shape),
                    input_frobenius=float(np.linalg.norm(t[np.isfinite(t)])),
                    input_max_abs=float(np.abs(t[np.isfinite(t)]).max()) if in_finite else np.nan,
                    n_singular_values=int(full.size),
                    n_finite_singular_values=int(np.isfinite(full).sum()),
                    largest_sv=float(good.max()) if good.size else np.nan,
                    smallest_positive_sv=float(good.min()) if good.size else np.nan,
                    condition_number=float(good.max() / good.min()) if good.size else np.nan,
                )
            tracer.records.append(dict(index=idx, in_finite=in_finite,
                                       out_finite=out_finite,
                                       weight=float(np.nansum(s_np ** 2) +
                                                    np.nansum(rest_np ** 2))))
            return out

        _tn_jax.JaxBackend.svd = wrapped
        return self

    def unpatch(self):
        from tensornetwork.backends.jax import jax_backend as _tn_jax
        _tn_jax.JaxBackend.svd = self._orig


def run_mode(mode, params, args, common):
    a = make_ansatz(mode, bond_dim=args.bond_dim, seed=args.seed, **common)
    tracer = NanTracer().patch()
    try:
        qc = a._circuit(jnp.asarray(params))
        norm = float(np.abs(np.asarray(qc.get_norm())))
    finally:
        tracer.unpatch()
    tensors_finite = all(bool(np.isfinite(np.asarray(t)).all()) for t in qc.get_tensors())
    return dict(mode=mode, n_svds=tracer.n_calls, final_norm=norm,
                final_norm_sq=norm ** 2, tensors_finite=tensors_finite,
                first_bad=tracer.first_bad,
                weights=[r["weight"] for r in tracer.records])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=3)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--bond-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=4876)
    p.add_argument("--natural-ordering", action="store_true")
    p.add_argument("--cartan-mode", default="separate")
    p.add_argument("--toffoli-mode", default="decomposed")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    common = dict(Lx=args.Lx, Ly=args.Ly, nlayers=args.nlayers, howoften_toreset=7,
                  reset_layers=[args.nlayers - 1], trials=1,
                  use_optimal_ordering=not args.natural_ordering,
                  cartan_mode=args.cartan_mode, toffoli_mode=args.toffoli_mode)

    seed_ansatz = make_ansatz("off", bond_dim=args.bond_dim, seed=args.seed, **common)
    params = np.asarray(seed_ansatz.initparams[0])
    print(f"  {args.Lx}x{args.Ly} bd={args.bond_dim} seed={args.seed} "
          f"{'natural' if args.natural_ordering else 'optimal'} ordering "
          f"{args.cartan_mode}/{args.toffoli_mode}: "
          f"{seed_ansatz.n_mps_qubits} MPS qubits", flush=True)

    results = []
    for mode in NORM_MODES:
        r = run_mode(mode, params, args, common)
        results.append(r)
        status = "FINITE" if r["tensors_finite"] else "*** NON-FINITE ***"
        print(f"\n  norm_mode={mode:<5}: {r['n_svds']} SVDs, "
              f"||psi||^2={r['final_norm_sq']:.6f}  state {status}")
        fb = r["first_bad"]
        if fb:
            print(f"    first non-finite SVD at split {fb['split_index']}/{r['n_svds']}")
            print(f"      input already non-finite? {fb['input_was_finite'] is False}")
            print(f"      input shape {fb['input_shape']}  ||A||_F={fb['input_frobenius']:.6e}  "
                  f"max|A_ij|={fb['input_max_abs']:.3e}")
            print(f"      singular values: {fb['n_finite_singular_values']}/"
                  f"{fb['n_singular_values']} finite, "
                  f"largest={fb['largest_sv']:.3e}, smallest>0={fb['smallest_positive_sv']:.3e}, "
                  f"cond={fb['condition_number']:.3e}")
        else:
            print("    no non-finite SVD result")

    print("\n  ---- verdict ----")
    bad = [r["mode"] for r in results if not r["tensors_finite"]]
    ok = [r["mode"] for r in results if r["tensors_finite"]]
    print(f"    NaN in forward pass:    {bad if bad else 'none'}")
    print(f"    finite forward pass:    {ok if ok else 'none'}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=1, default=float)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
