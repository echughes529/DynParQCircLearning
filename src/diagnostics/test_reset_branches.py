# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Smoke test for src.diagnostics.reset_branches.

Builds the smallest DPQC instance with exactly one reset event (2 ancillas,
via ToricCodeAnsatz(Lx=2, Ly=2, nlayers=1, reset_layers=[0])), enumerates its
ancilla branches, and cross-checks the branch decomposition against the
reduced density matrix obtained by tracing out the ancillas directly -- the
check that actually confirms the branch decomposition reconstructs the true
mixed state, not just that probabilities happen to sum to 1.

Run with: pytest src/diagnostics/test_reset_branches.py -v
Or directly for a printed summary table: python -m src.diagnostics.test_reset_branches
"""

import numpy as np
import pytest
import tensorcircuit.quantum as qu

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.reset_branches import enumerate_branches, mps_bond_dims


def _build_branch_data():
    np.random.seed(0)
    # bond_dim=None (the new default) means qc is built with threshold-only
    # truncation, not a fixed count cap -- so no genuine entanglement is lost
    # at construction time. That matters here: a capped bond_dim would make
    # qc.wavefunction() already lossy/unnormalized before the diagnostic even
    # runs, which would masquerade as a bug in the branch decomposition below.
    ansatz = ToricCodeAnsatz(Lx=2, Ly=2, nlayers=1, reset_layers=[0], trials=1)
    params = ansatz.initparams[0]
    qc = ansatz._circuit(params)
    results, total_p = enumerate_branches(
        qc, ansatz.ancilla_mps_positions, ansatz.lattice.num_qubits
    )
    return ansatz, qc, results, total_p


def _rho_from_trace(qc, ancilla_indices):
    state = np.asarray(qc.wavefunction())
    return np.asarray(qu.reduced_density_matrix(state, cut=list(ancilla_indices)))


def _rho_from_branches(results, n_system_qubits):
    dim_s = 2 ** n_system_qubits
    rho = np.zeros((dim_s, dim_s), dtype=complex)
    for info in results.values():
        phi = info["phi"]
        rho += info["p_x"] * np.outer(phi, phi.conj())
    return rho


@pytest.fixture(scope="module")
def branch_data():
    ansatz, qc, results, total_p = _build_branch_data()
    return {"ansatz": ansatz, "qc": qc, "results": results, "total_p": total_p}


def test_probabilities_sum_to_one(branch_data):
    assert abs(branch_data["total_p"] - 1.0) < 1e-6


def test_branch_norms(branch_data):
    for x, info in branch_data["results"].items():
        norm = np.linalg.norm(info["phi"])
        assert abs(norm - 1.0) < 1e-8, f"branch {x} has norm {norm}"


def test_bond_dims_valid(branch_data):
    n_system = branch_data["ansatz"].lattice.num_qubits
    for x, info in branch_data["results"].items():
        bond_dims = info["bond_dims"]
        assert len(bond_dims) == n_system - 1
        for cut, d in enumerate(bond_dims):
            max_theoretical = 2 ** min(cut + 1, n_system - cut - 1)
            assert 1 <= d <= max_theoretical, (
                f"branch {x} cut {cut}: bond dim {d} outside [1, {max_theoretical}]"
            )


def test_truncation_error_matches_or_reduces_bond_dims(branch_data):
    """truncation_error=None (used by the branch_data fixture) reconstructs
    exactly; truncation_error=<float> should only ever match or reduce the
    per-cut bond dimension for the same branch, never increase it, since
    truncating can only discard singular values, not add them."""
    n_system = branch_data["ansatz"].lattice.num_qubits
    for x, info in branch_data["results"].items():
        exact_bond_dims = info["bond_dims"]
        truncated_bond_dims = mps_bond_dims(
            info["phi"], n_system, truncation_error=1e-4
        )
        assert len(truncated_bond_dims) == len(exact_bond_dims)
        for cut, (d_trunc, d_exact) in enumerate(zip(truncated_bond_dims, exact_bond_dims)):
            assert 1 <= d_trunc <= d_exact, (
                f"branch {x} cut {cut}: truncated bond dim {d_trunc} not in "
                f"[1, {d_exact}] (exact bond dim)"
            )


def test_mixture_reconstructs_rho_S(branch_data):
    ansatz, qc, results = branch_data["ansatz"], branch_data["qc"], branch_data["results"]
    rho_trace = _rho_from_trace(qc, ansatz.ancilla_mps_positions)
    rho_branches = _rho_from_branches(results, ansatz.lattice.num_qubits)
    max_err = np.max(np.abs(rho_trace - rho_branches))
    assert max_err < 1e-6, f"mixture reconstruction mismatch: {max_err}"


if __name__ == "__main__":
    ansatz, qc, results, total_p = _build_branch_data()

    print(f"\n{'x':<10}{'p_x':<14}{'max_bond'}")
    for x, info in sorted(results.items()):
        print(f"{str(x):<10}{info['p_x']:<14.6f}{info['max_bond']}")
    print(f"sum p_x = {total_p:.8f}")

    rho_trace = _rho_from_trace(qc, ansatz.ancilla_mps_positions)
    rho_branches = _rho_from_branches(results, ansatz.lattice.num_qubits)
    max_err = np.max(np.abs(rho_trace - rho_branches))
    print(f"cross-check max|rho_trace - rho_branches| = {max_err:.2e}")
