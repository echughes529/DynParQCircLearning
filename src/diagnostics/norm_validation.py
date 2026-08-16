"""Validation checks underpinning the normalization study.

Each check is a claim the report leans on, tested directly against the real
ansatz rather than assumed:

  V1  MPSCircuit.get_norm() (the quantity qc.normalize() divides by) really is
      ||psi||, i.e. the MPS is still canonical at that point -- including
      after the three-qubit direct-Toffoli path, which goes through apply_MPO
      rather than apply_adjacent_double_gate.
  V2  With normalization off, E_unnormalized == ||psi||^2 * E_true exactly.
  V3  norm_mode="end" and norm_mode="layer" compute the same function of the
      parameters (per-layer rescaling is a conditioning device, not a change
      of objective), to floating-point roundoff.
  V4  Gate-count effect of the streamlined gate application: how many
      truncating SVDs each cartan_mode/toffoli_mode combination performs.

Usage:
    python -m src.diagnostics.norm_validation --Lx 3 --Ly 2
"""
import argparse
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax.numpy as jnp

from src.diagnostics.normalization_study import SplitRecorder, make_ansatz


def energy(ansatz, qc):
    e = 0.0
    for ops, coeff in ansatz._hamiltonian_terms():
        e += coeff * qc.expectation(*ops)
    return float(np.real(np.asarray(e)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--bond-dims", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    common = dict(Lx=args.Lx, Ly=args.Ly, nlayers=args.nlayers, howoften_toreset=7,
                  reset_layers=[args.nlayers - 1], trials=1, seed=args.seed,
                  use_optimal_ordering=True)

    print("V1: does get_norm() equal the true ||psi||?")
    for toffoli_mode in ("decomposed", "direct"):
        for cartan_mode in ("separate", "fused"):
            for bd in args.bond_dims:
                a = make_ansatz("off", bond_dim=bd, cartan_mode=cartan_mode,
                                toffoli_mode=toffoli_mode, **common)
                params = jnp.array(a.initparams[0])
                qc = a._circuit(params)
                center_norm = float(np.abs(np.asarray(qc.get_norm())))
                true_norm = float(np.linalg.norm(np.asarray(qc.wavefunction())))
                rel = abs(center_norm - true_norm) / true_norm
                flag = "OK " if rel < 1e-10 else "*** MISMATCH ***"
                print(f"  {flag} {cartan_mode:>8}/{toffoli_mode:<10} bd={bd:>3}: "
                      f"get_norm={center_norm:.12f}  ||wavefunction||={true_norm:.12f}  "
                      f"rel_diff={rel:.2e}")

    print("\nV2: E_unnormalized == ||psi||^2 * E_true?")
    for bd in args.bond_dims:
        a = make_ansatz("off", bond_dim=bd, cartan_mode="fused",
                        toffoli_mode="direct", **common)
        params = jnp.array(a.initparams[0])
        qc = a._circuit(params)
        n_sq = float(np.abs(np.asarray(qc.get_norm()))) ** 2
        e_off = energy(a, qc)
        qc2 = a._circuit(params)
        qc2.normalize()
        e_true = energy(a, qc2)
        pred = n_sq * e_true
        rel = abs(e_off - pred) / max(abs(e_off), 1e-300)
        flag = "OK " if rel < 1e-10 else "*** MISMATCH ***"
        print(f"  {flag} bd={bd:>3}: ||psi||^2={n_sq:.10f}  E_unnorm={e_off:+.12f}  "
              f"||psi||^2*E_true={pred:+.12f}  rel_diff={rel:.2e}")

    print("\nV3: does per-layer normalization change the objective?")
    for bd in args.bond_dims:
        a_end = make_ansatz("end", bond_dim=bd, cartan_mode="fused",
                            toffoli_mode="direct", **common)
        a_layer = make_ansatz("layer", bond_dim=bd, cartan_mode="fused",
                              toffoli_mode="direct", **common)
        params = jnp.array(a_end.initparams[0])
        e_end = float(np.asarray(a_end.energy_from_params(params)))
        e_layer = float(np.asarray(a_layer.energy_from_params(params)))
        rel = abs(e_end - e_layer) / max(abs(e_end), 1e-300)
        flag = "OK " if rel < 1e-9 else "*** DIFFERS ***"
        print(f"  {flag} bd={bd:>3}: E_end={e_end:+.14f}  E_layer={e_layer:+.14f}  "
              f"rel_diff={rel:.2e}")

    print("\nV4: truncating-SVD count by gate mode")
    for cartan_mode in ("separate", "fused"):
        for toffoli_mode in ("decomposed", "direct"):
            for bd in args.bond_dims:
                a = make_ansatz("off", bond_dim=bd, cartan_mode=cartan_mode,
                                toffoli_mode=toffoli_mode, **common)
                params = jnp.array(a.initparams[0])
                rec = SplitRecorder(capture_spectra=False)
                with rec.patch():
                    qc = a._circuit(params)
                s = rec.summary()
                n_sq = float(np.abs(np.asarray(qc.get_norm()))) ** 2
                print(f"  {cartan_mode:>8}/{toffoli_mode:<10} bd={bd:>3}: "
                      f"svds={s['n_splits']:>4}  truncating={s['n_truncating_splits']:>4}  "
                      f"||psi||^2={n_sq:.6f}  worst_split_loss={s['max_split_weight_lost']:.3e}")


if __name__ == "__main__":
    main()
