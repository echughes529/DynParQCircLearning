# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Per-branch conditional system state and MPS bond dimension after
projecting reset ancillas onto a computational-basis outcome.

Resets in this codebase swap the system state onto ancilla qubits, so the
final MPS is a purification of the system's mixed state: system qubits
entangled with every discarded ancilla. Standard MPS bond dimension is only
defined for a pure state, so it isn't directly meaningful on the full
purified state. Projecting the ancilla register onto one computational-basis
outcome x and renormalizing yields a pure state on the system qubits alone,
with a well-defined per-cut bond dimension; this module enumerates all such
branches for small (smoke-test-scale) circuits.

Read-only diagnostic: does not modify the training or forward-simulation
path, and is not intended to scale to production-size circuits with many
resets (see enumerate_branches).
"""

import itertools
import warnings

import numpy as np
import tensorcircuit as tc

from src.utilities.generate_ansatz import get_singular_values_per_cut


def conditional_branch(state, n_total_qubits, ancilla_indices, x, prob_floor=1e-9):
    """Project the dense purified state onto ancilla outcome x, renormalize.

    state: flat array of length 2**n_total_qubits, as returned by
        tc.MPSCircuit.wavefunction() -- qubits in chain-position bit order.
    ancilla_indices: chain positions to project out. Need not be contiguous
        or trailing: with the default interleaved MPS ordering, each reset's
        ancilla pair sits immediately after its associated system qubit, not
        at the end of the chain.
    x: tuple of 0/1 outcomes, one per entry of ancilla_indices, same order.

    Returns (phi_normalized, p_x). p_x is the squared norm of the
    un-renormalized projection (the branch probability). phi_normalized is
    None if p_x < prob_floor, since renormalizing by a near-zero norm would
    just amplify numerical noise into a meaningless state; a warning is
    raised in that case instead.
    """
    psi = np.asarray(state).reshape((2,) * n_total_qubits)
    index = [slice(None)] * n_total_qubits
    for pos, bit in zip(ancilla_indices, x):
        index[pos] = bit
    projected = psi[tuple(index)].reshape(-1)
    p_x = float(np.real(np.vdot(projected, projected)))
    if p_x < prob_floor:
        warnings.warn(
            f"conditional_branch: p_x={p_x:.3e} for outcome {x} is below "
            f"prob_floor={prob_floor:.1e}; skipping renormalization."
        )
        return None, p_x
    return projected / np.sqrt(p_x), p_x


def mps_bond_dims(phi, n_system_qubits, min_singular_value=1e-6):
    """Per-cut Schmidt rank of a system-only pure state phi.

    Re-derives an MPS from the dense state phi with NO truncation (split={},
    every singular value kept exactly -- reusing the ansatz's training
    bond_dim cap here could silently hide the very bond-dimension inflation
    this diagnostic exists to measure). The bond-dimension count is then a
    post-hoc, per-value magnitude threshold (min_singular_value) applied to
    the exact spectrum, not a truncation criterion: tensorcircuit-ng's own
    truncation-error option bounds the cumulative Frobenius norm of the
    *discarded tail*, not each value individually, which can miscount for
    near-degenerate spectra -- exactly the regime the toric code is prone
    to. Counting on the exact, untruncated spectrum avoids that failure
    mode.

    Reuses the codebase's one SVD/bond-dimension convention
    (get_singular_values_per_cut) instead of introducing a second one.
    """
    branch_qc = tc.MPSCircuit(n_system_qubits, wavefunction=phi, split={})
    singular_values = get_singular_values_per_cut(branch_qc)
    return [int(np.sum(np.asarray(s) > min_singular_value)) for s in singular_values]


def enumerate_branches(qc, ancilla_indices, n_system_qubits, max_ancillas=6,
                        prob_floor=1e-9, min_singular_value=1e-6):
    """Enumerate all 2**len(ancilla_indices) ancilla outcomes for qc's
    purified state.

    qc: a tc.MPSCircuit holding the full (system + ancilla) purified state.
    ancilla_indices: chain positions of the ancilla qubits to project over.
    n_system_qubits: number of qubits kept after projecting out the
        ancillas; must equal qc's total qubit count minus len(ancilla_indices)
        (checked below) and is used to size each branch's MPS reconstruction.

    Full enumeration is O(2**len(ancilla_indices)), so this raises if that
    exceeds max_ancillas: this tool is for small smoke tests, not
    production-size circuits with many resets (that needs an MPO-of-rho_S
    approach, out of scope here).

    Returns (results, total_p):
      results: dict x -> {"p_x", "phi", "bond_dims", "max_bond"} for every
        branch with p_x >= prob_floor.
      total_p: sum of p_x over ALL branches (including below-floor ones),
        asserted to equal 1.0 to within 1e-6 as a built-in correctness check.
    """
    m = len(ancilla_indices)
    if m > max_ancillas:
        raise ValueError(
            f"{m} ancillas requested but max_ancillas={max_ancillas}: full "
            f"enumeration is O(2**{m}) and this tool is for small smoke "
            f"tests only. For production-size circuits, use an MPO-of-rho_S "
            f"approach instead (out of scope for this diagnostic)."
        )

    n_total = qc._nqubits
    if n_total != n_system_qubits + m:
        raise ValueError(
            f"qc has {n_total} qubits but n_system_qubits={n_system_qubits} "
            f"+ len(ancilla_indices)={m} = {n_system_qubits + m}."
        )

    state = np.asarray(qc.wavefunction())
    results = {}
    total_p = 0.0
    for x in itertools.product((0, 1), repeat=m):
        phi, p_x = conditional_branch(state, n_total, ancilla_indices, x, prob_floor=prob_floor)
        total_p += p_x
        if phi is None:
            continue
        bond_dims = mps_bond_dims(phi, n_system_qubits, min_singular_value=min_singular_value)
        results[x] = {
            "p_x": p_x,
            "phi": phi,
            "bond_dims": bond_dims,
            "max_bond": max(bond_dims) if bond_dims else 1,
        }

    assert abs(total_p - 1.0) < 1e-6, (
        f"branch probabilities sum to {total_p}, expected 1.0 -- the "
        f"projection/renormalization is likely inconsistent with the given "
        f"ancilla_indices."
    )

    return results, total_p
