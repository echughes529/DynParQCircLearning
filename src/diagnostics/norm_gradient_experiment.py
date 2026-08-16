"""Experiment 2: what does NOT normalizing do to the *gradient*?

With normalize_state=False the loss handed to Adam is

    E_off(theta) = <psi|H|psi>            (psi NOT normalized)
                 = n(theta) * E_true(theta),      n = ||psi||^2 <= 1

so, exactly,

    grad E_off = n * grad E_true  +  E_true * grad n
                 \____________/     \______________/
                  the real signal    a spurious channel that rewards the
                                     optimizer for changing how much weight
                                     the truncation throws away

This script measures both terms directly with autodiff, and separately
measures how far each normalization regime's gradient sits from the
untruncated (exact-bond-dimension) reference gradient.

Usage:
    python -m src.diagnostics.norm_gradient_experiment --Lx 3 --Ly 2
"""
import argparse
import json
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax
import jax.numpy as jnp

from src.diagnostics.normalization_study import NORM_MODES, make_ansatz


def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--bond-dims", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--exact-bond-dim", type=int, default=None,
                   help="bond dim large enough to be lossless; default 2**(n_mps//2)")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    common = dict(Lx=args.Lx, Ly=args.Ly, nlayers=args.nlayers, howoften_toreset=7,
                  reset_layers=[args.nlayers - 1], trials=1,
                  use_optimal_ordering=True, cartan_mode="fused", toffoli_mode="direct")

    probe = make_ansatz("layer", bond_dim=8, seed=1, **common)
    n_mps = probe.n_mps_qubits
    exact_bd = args.exact_bond_dim or 2 ** (n_mps // 2)
    print(f"  {args.Lx}x{args.Ly}: {n_mps} MPS qubits, {probe.nparams} params, "
          f"lossless bond dim = {exact_bd}", flush=True)

    rows = []
    for seed in args.seeds:
        ref = make_ansatz("layer", bond_dim=exact_bd, seed=seed, **common)
        params = jnp.array(ref.initparams[0])
        e_ref = float(np.asarray(ref.energy_from_params(params)))
        g_ref = np.asarray(jax.grad(ref.energy_from_params)(params))

        for bond_dim in args.bond_dims:
            per_mode = {}
            for mode in NORM_MODES:
                a = make_ansatz(mode, bond_dim=bond_dim, seed=seed, **common)
                e = float(np.asarray(a.energy_from_params(params)))
                g = np.asarray(jax.grad(a.energy_from_params)(params))
                per_mode[mode] = dict(
                    energy=e,
                    energy_err_vs_exact=e - e_ref,
                    grad_norm=float(np.linalg.norm(g)),
                    grad_rel_err_vs_exact=float(np.linalg.norm(g - g_ref) /
                                                np.linalg.norm(g_ref)),
                    grad_cos_vs_exact=cos_sim(g, g_ref),
                )

            # Split the "off" gradient into its two exact components.
            a_off = make_ansatz("off", bond_dim=bond_dim, seed=seed, **common)
            n_val = float(np.asarray(a_off.norm_sq_from_params(params)))
            grad_n = np.asarray(jax.grad(a_off.norm_sq_from_params)(params))
            e_true = float(np.asarray(a_off.rayleigh_from_params(params)))
            grad_e_true = np.asarray(jax.grad(a_off.rayleigh_from_params)(params))
            grad_off = np.asarray(jax.grad(a_off.energy_from_params)(params))

            signal = n_val * grad_e_true
            spurious = e_true * grad_n
            row = dict(
                Lx=args.Lx, Ly=args.Ly, seed=seed, bond_dim=bond_dim,
                exact_bond_dim=exact_bd, n_mps_qubits=n_mps,
                energy_exact=e_ref, norm_sq=n_val, energy_true=e_true,
                grad_norm_signal=float(np.linalg.norm(signal)),
                grad_norm_spurious=float(np.linalg.norm(spurious)),
                spurious_fraction=float(np.linalg.norm(spurious) /
                                        np.linalg.norm(signal)) if np.linalg.norm(signal) else np.nan,
                # sanity: the identity above should hold to roundoff
                decomposition_residual=float(
                    np.linalg.norm(grad_off - signal - spurious) /
                    max(np.linalg.norm(grad_off), 1e-300)),
                cos_off_vs_true=cos_sim(grad_off, grad_e_true),
                modes=per_mode,
            )
            rows.append(row)
            print(f"  seed={seed} bd={bond_dim:>3}: ||psi||^2={n_val:.6f}  "
                  f"E_true={e_true:+.6f}  spurious/signal={row['spurious_fraction']:.3e}  "
                  f"cos(grad_off,grad_true)={row['cos_off_vs_true']:.6f}  "
                  f"resid={row['decomposition_residual']:.1e}", flush=True)
            for mode in NORM_MODES:
                m = per_mode[mode]
                print(f"      {mode:<5}: E={m['energy']:+.8f}  dE_vs_exact={m['energy_err_vs_exact']:+.3e}  "
                      f"|g|={m['grad_norm']:.4e}  grad_rel_err={m['grad_rel_err_vs_exact']:.3e}  "
                      f"cos={m['grad_cos_vs_exact']:.6f}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1, default=float)
        print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
