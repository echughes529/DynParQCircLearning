"""Experiment 1: what does NOT normalizing actually do to the MPS?

Forward-pass only (no autodiff, no training), so this runs in seconds and can
sweep lattice size x bond dimension x gate mode.

Measures, for one fixed random parameter vector per configuration:
  * ||psi||^2 after the full circuit with normalization disabled,
  * the per-split norm loss that produces it,
  * the resulting bias in the energy the optimizer is handed
    (E_unnormalized == ||psi||^2 * E_true, verified numerically),
  * how far the shrinking scale pushes the SVD backward pass towards
    tensorcircuit's absolute Lorentzian-broadening floor.

Usage:
    python -m src.diagnostics.norm_decay_experiment --out results.json
"""
import argparse
import json
import sys
import types

# generate_ansatz imports tensorflow for unrelated legacy constructors only.
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax.numpy as jnp

from src.diagnostics.normalization_study import (
    SplitRecorder,
    make_ansatz,
    spectrum_gap_stats,
)

LATTICES = [(2, 2), (3, 2), (3, 3)]
BOND_DIMS = [4, 8, 16, 32, 64, 96]
GATE_MODES = [("separate", "decomposed"), ("fused", "direct")]


def hamiltonian_energy(ansatz, qc):
    energy = 0.0
    for ops, coeff in ansatz._hamiltonian_terms():
        energy += coeff * qc.expectation(*ops)
    return float(np.real(np.asarray(energy)))


def run_one(Lx, Ly, bond_dim, cartan_mode, toffoli_mode, use_optimal_ordering,
            nlayers, seed, capture_spectra):
    """One configuration: build the circuit twice (unnormalized / normalized)
    from the same parameters and compare."""
    common = dict(Lx=Lx, Ly=Ly, nlayers=nlayers, howoften_toreset=7,
                  reset_layers=[nlayers - 1], trials=1, bond_dim=bond_dim,
                  use_optimal_ordering=use_optimal_ordering, seed=seed,
                  cartan_mode=cartan_mode, toffoli_mode=toffoli_mode)

    a_off = make_ansatz("off", **common)
    params = jnp.array(a_off.initparams[0])

    rec = SplitRecorder(capture_spectra=capture_spectra)
    with rec.patch():
        qc_off = a_off._circuit(params)
    summary = rec.summary()

    # Ground truth for the norm, read straight off the circuit rather than
    # reconstructed, so the reconstruction can be checked against it.
    norm_sq = float(np.abs(np.asarray(qc_off.get_norm()))) ** 2

    # Energy as the optimizer would see it with normalize_state=False ...
    e_off = hamiltonian_energy(a_off, qc_off)
    # ... versus the same state normalized (the physically meaningful energy).
    qc_norm = a_off._circuit(params)
    qc_norm.normalize()
    e_true = hamiltonian_energy(a_off, qc_norm)

    row = dict(
        Lx=Lx, Ly=Ly, bond_dim=bond_dim, nlayers=nlayers,
        cartan_mode=cartan_mode, toffoli_mode=toffoli_mode,
        use_optimal_ordering=use_optimal_ordering, seed=seed,
        n_mps_qubits=a_off.n_mps_qubits,
        exact_bond_dim=2 ** (a_off.n_mps_qubits // 2),
        norm_sq=norm_sq,
        norm_sq_reconstruction_err=abs(summary["final_norm_sq"] - norm_sq),
        energy_unnormalized=e_off, energy_true=e_true,
        energy_bias_abs=e_off - e_true,
        energy_bias_rel=(e_off - e_true) / e_true if e_true else np.nan,
        predicted_energy_unnormalized=norm_sq * e_true,
        **summary,
    )

    if capture_spectra:
        # Same spectra, evaluated at the scale they actually have in the
        # un-normalized run vs. the scale they would have if the state were
        # kept at unit norm throughout.
        row["gaps_unnormalized"] = spectrum_gap_stats(rec.spectra, scale=1.0)
        row["gaps_rescaled_to_unit_norm"] = spectrum_gap_stats(
            rec.spectra, scale=1.0 / max(summary["final_norm"], 1e-300))
    return row, rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--lattices", type=str, default="2x2,3x2,3x3")
    p.add_argument("--bond-dims", type=int, nargs="+", default=BOND_DIMS)
    p.add_argument("--spectra", action="store_true",
                   help="also record per-split singular-value spectra (slower)")
    p.add_argument("--natural-ordering", action="store_true",
                   help="use_optimal_ordering=False (ancillas appended at the "
                        "end of the chain), i.e. the pre-abf5c85 default")
    p.add_argument("--gate-modes", default="all", choices=["all", "fused", "separate"])
    args = p.parse_args()

    lattices = [tuple(int(v) for v in s.split("x")) for s in args.lattices.split(",")]
    gate_modes = {"all": GATE_MODES,
                  "fused": [("fused", "direct")],
                  "separate": [("separate", "decomposed")]}[args.gate_modes]
    rows = []

    for (Lx, Ly) in lattices:
        for cartan_mode, toffoli_mode in gate_modes:
            for bond_dim in args.bond_dims:
                for seed in args.seeds:
                    try:
                        row, rec = run_one(Lx, Ly, bond_dim, cartan_mode, toffoli_mode,
                                           not args.natural_ordering, args.nlayers,
                                           seed, args.spectra)
                    except Exception as e:
                        print(f"  {Lx}x{Ly} bd={bond_dim} {cartan_mode}/{toffoli_mode} "
                              f"seed={seed} FAILED: {type(e).__name__}: {e}", flush=True)
                        continue
                    rows.append(row)
                    print(f"  {Lx}x{Ly} bd={bond_dim:>3} {cartan_mode:>8}/{toffoli_mode:<10} "
                          f"seed={seed}: svds={row['n_splits']:>4} "
                          f"truncating={row['n_truncating_splits']:>4}  "
                          f"||psi||^2={row['norm_sq']:.6f}  "
                          f"E_unnorm={row['energy_unnormalized']:+.8f}  "
                          f"E_true={row['energy_true']:+.8f}  "
                          f"pred={row['predicted_energy_unnormalized']:+.8f}  "
                          f"recon_err={row['norm_sq_reconstruction_err']:.1e}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1, default=float)
        print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
