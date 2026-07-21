# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Backward-lightcone gate-count reporting using Qiskit's actual `LightCone`
transpiler pass.

Because Qiskit's `LightCone` pass only operates on Qiskit `QuantumCircuit`
objects (not `tc.Circuit`), this module builds an explicit, standalone Qiskit
mirror of `construct_dyn_circuit_toriccodelattice_prob_resets`
(`src/utilities/generate_ansatz.py:234-337`) rather than intercepting the real
TensorCircuit-building code.

Gotcha worth remembering when reading this file: Qiskit's gate methods take
the rotation angle FIRST, positionally, then the qubit(s) - e.g.
`qc.ry(theta, qubit)` - the reverse of TensorCircuit's
`qc.ry(qubit, theta=theta)`. Getting this backward would silently build the
wrong circuit (both are just floats/ints to Python).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import LightCone

from src.utilities.generate_toric_code_hamiltonian import ToricCode


def _qiskit_universalsingle(qc: QuantumCircuit, qubit: int, params: Sequence[float], paramindex: int) -> int:
    """Mirrors `universalsingle` in generate_ansatz.py: Ry-Rz-Ry on one qubit."""
    qc.ry(params[paramindex], qubit)
    qc.rz(params[paramindex + 1], qubit)
    qc.ry(params[paramindex + 2], qubit)
    return paramindex + 3


def _qiskit_cartan_block(qc: QuantumCircuit, q0: int, q1: int, params: Sequence[float], paramindex: int) -> int:
    """Mirrors `cartanblock` in generate_ansatz.py, applied directly to `q0`/`q1`
    instead of building a separate 2-qubit sub-circuit and appending it -
    Qiskit has no need for that indirection since we're not reusing the block
    anywhere else."""
    paramindex = _qiskit_universalsingle(qc, q0, params, paramindex)
    paramindex = _qiskit_universalsingle(qc, q1, params, paramindex)
    qc.rxx(params[paramindex], q0, q1)
    qc.ryy(params[paramindex + 1], q0, q1)
    qc.rzz(params[paramindex + 2], q0, q1)
    return paramindex + 3


def _qiskit_one_set_of_unitaries(
    qc: QuantumCircuit, claws: Sequence[Tuple[int, int]], params: Sequence[float], paramindex: int
) -> int:
    """Mirrors `onesetofunitaries` (non-measurement branch)."""
    for q0, q1 in claws:
        paramindex = _qiskit_cartan_block(qc, q0, q1, params, paramindex)
    return paramindex


def _qiskit_one_layer_of_single_unitaries(qc: QuantumCircuit, params: Sequence[float], paramindex: int, n: int) -> int:
    """Mirrors `onelayerofsingleunitaries`."""
    for i in range(n):
        paramindex = _qiskit_universalsingle(qc, i, params, paramindex)
    return paramindex


def build_qiskit_prob_resets_circuit(
    params: Sequence[float],
    Lx: int,
    Ly: int,
    nlayers: int,
    reset_qubits: Optional[List[int]] = None,
    reset_direction: int = 1,
    reset_layers: Optional[List[int]] = None,
) -> QuantumCircuit:
    """
    Qiskit mirror of `construct_dyn_circuit_toriccodelattice_prob_resets`
    (generate_ansatz.py:234-337). Same qubit numbering (reuses `ToricCode`),
    same parameter layout, same gate sequence - just emitted as Qiskit gate
    calls instead of TensorCircuit ones.
    """
    toriccode = ToricCode(Lx, Ly)
    nplaquettes = (Lx - 1) * (Ly - 1)
    nq = 2 * Lx * Ly - Lx - Ly

    if reset_qubits is None:
        reset_qubits = [toriccode.qubit_index(x, y, reset_direction) for x in range(Lx - 1) for y in range(Ly - 1)]
        reset_qubits = [q for q in reset_qubits if q is not None]
    nresets_per_layer = len(reset_qubits)

    if reset_layers is None:
        reset_layers = []
    n_reset_layers = len(reset_layers)
    total_resets = nresets_per_layer * n_reset_layers
    nancillas = 2 * total_resets

    nparams = nplaquettes * 3 * 9 * nlayers + total_resets + 3 * nq
    if len(params) != nparams:
        raise ValueError(f"Parameter vector has wrong size: got {len(params)}, expected {nparams}.")

    qc = QuantumCircuit(nq + nancillas)
    paramindex = 0

    claws = toriccode.all_claws()
    claws = [claws[i::3][j] for i in range(3) for j in range((Lx - 1) * (Ly - 1))]

    ancilla_index = 0
    for l in range(nlayers):
        paramindex = _qiskit_one_set_of_unitaries(qc, claws, params, paramindex)

        if l in reset_layers:
            for sys_qubit in reset_qubits:
                prob_ancilla = nq + 2 * ancilla_index
                purif_ancilla = nq + 2 * ancilla_index + 1

                qc.ry(params[paramindex], prob_ancilla)
                paramindex += 1

                qc.ccx(prob_ancilla, sys_qubit, purif_ancilla)
                qc.ccx(prob_ancilla, purif_ancilla, sys_qubit)

                ancilla_index += 1

    paramindex = _qiskit_one_layer_of_single_unitaries(qc, params, paramindex, nq)
    return qc


def term_to_bit_terms(ops) -> Tuple[str, List[int]]:
    """
    Convert a HamiltonianTerm's `ops` (tuple of `(tc.gates.z()/x(), [qubit])`)
    into the `(bit_terms, indices)` pair `LightCone` expects. `tc.gates.z()`
    and `tc.gates.x()` are TensorCircuit `Gate` objects carrying a `.name`
    attribute ("z"/"x") - that's all we need to read off the Pauli letter.
    """
    bit_terms = "".join(gate.name.upper() for gate, _qubits in ops)
    indices = [q for _gate, qubits in ops for q in qubits]
    return bit_terms, indices


def compute_lightcone_for_term(circuit: QuantumCircuit, ops) -> QuantumCircuit:
    """Run Qiskit's LightCone pass for one Hamiltonian term's observable."""
    bit_terms, indices = term_to_bit_terms(ops)
    pm = PassManager([LightCone(bit_terms=bit_terms, indices=indices)])
    return pm.run(circuit)


def _touched_qubits(circuit: QuantumCircuit) -> set:
    """
    The LightCone pass drops gates, not qubits - `circuit`'s qubit register
    stays the same size, but most qubits end up with zero gates on them.
    This walks the circuit's instructions to find which qubits still have at
    least one gate, which is the actual lightcone qubit count we want to
    report.
    """
    touched = set()
    for instruction in circuit.data:
        for qubit in instruction.qubits:
            touched.add(circuit.find_bit(qubit).index)
    return touched


def report_lightcones_for_ansatz(ansatz, params) -> List[Dict[str, Any]]:
    """
    Build the Qiskit mirror circuit once, then for every term in
    `ansatz._hamiltonian_terms()` compute its Qiskit LightCone reduction and
    report the qubit-count/gate-count reduction.
    """
    qiskit_params = [float(p) for p in params]
    full_circuit = build_qiskit_prob_resets_circuit(
        qiskit_params,
        Lx=ansatz.Lx,
        Ly=ansatz.Ly,
        nlayers=ansatz.nlayers,
        reset_qubits=ansatz.which_qubits_for_prob_reset,
        reset_direction=ansatz.prob_reset_direction,
        reset_layers=ansatz.active_reset_layers,
    )
    n_qubits_total = full_circuit.num_qubits
    n_gates_total = len(full_circuit.data)

    terms = ansatz._hamiltonian_terms()
    labels = ansatz.term_labels() if hasattr(ansatz, "term_labels") else [(f"term_{i}", "unknown") for i in range(len(terms))]

    results: List[Dict[str, Any]] = []
    for (ops, _coeff), (label, family) in zip(terms, labels):
        reduced_circuit = compute_lightcone_for_term(full_circuit, ops)
        lightcone_qubits = _touched_qubits(reduced_circuit)

        results.append({
            "label": label,
            "family": family,
            "n_qubits_total": n_qubits_total,
            "n_qubits_lightcone": len(lightcone_qubits),
            "n_gates_total": n_gates_total,
            "n_gates_kept": len(reduced_circuit.data),
        })

    return results
