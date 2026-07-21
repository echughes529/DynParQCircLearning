# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Backward-lightcone analysis for the (non-MPS, full-statevector) tc.Circuit ansatze.

Theory
------
An observable's expectation value <psi|O|psi> for |psi> = U|0> can be computed
by evolving O *backward* through the gates of U instead of evolving the state
forward: <0| G_1^dagger ... G_n^dagger O G_n ... G_1 |0>. Reading that chain of
conjugations from the gate closest to O outward is equivalent to walking the
circuit's gate list in reverse (last-applied gate first).

Any gate acting on qubits disjoint from the operator's current support
commutes with it trivially and can be dropped without changing the
expectation value at all - this is an exact identity, not an approximation.
A gate that *does* touch the current support must be kept, and may grow the
support to include its other qubits.

The "backward lightcone" of an observable is exactly the qubit set (and the
kept-gate subset) produced by this backward walk. This module implements a
conservative version of it: it does not check whether a touching gate
actually commutes with the running operator (which is what a fully
Pauli-algebra-aware implementation, e.g. Qiskit's LightCone transpiler pass,
would do) - it always assumes "touches -> keep and grow". This never produces
a wrong (too-small) lightcone, only a possibly-larger-than-strictly-necessary
one.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import tensorcircuit as tc

from src.utilities.lightcone_plotting import plot_lightcone

# Captured once, before `trace_circuit()` ever monkey-patches `tc.Circuit` -
# both `_GateTapeProxy.__init__` and `build_reduced_circuit` must construct
# *real* circuits through this reference, never through `tc.Circuit` itself
# (which is the very attribute `trace_circuit()` temporarily replaces; a
# lookup of `tc.Circuit` from inside `_GateTapeProxy.__init__` while patched
# would resolve back to `_GateTapeProxy` itself and recurse forever).
_RealCircuit = tc.Circuit


class _GateTapeProxy:
    """
    Transparent wrapper around a real tc.Circuit.

    Every method call is forwarded to the real circuit unchanged (so the
    circuit this proxy wraps is fully valid and usable), but calls that touch
    specific qubits are first appended to `self.tape` as
    {"name", "args", "kwargs", "qubits"}. Qubits are taken from integer
    positional arguments (covers plain gate calls like `qc.ry(3, theta=...)`
    and `qc.ccx(0, 1, 2)`) or from the `indices=` keyword (covers
    `qc.append(inst, indices=[...])`, used for the Cartan-block sub-circuits
    in generate_ansatz.py). Calls with no qubit arguments (e.g. `.state()`,
    `.expectation(...)`) are not gates and are not recorded.
    """

    def __init__(self, nqubits: int, *args: Any, **kwargs: Any) -> None:
        self._real = _RealCircuit(nqubits, *args, **kwargs)
        self.nqubits = nqubits
        self.tape: List[Dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real, name)
        if not callable(attr):
            return attr

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            indices = kwargs.get("indices")
            if indices is not None:
                qubits = tuple(int(q) for q in indices)
            else:
                qubits = tuple(a for a in args if isinstance(a, (int, np.integer)))
            if qubits:
                self.tape.append({"name": name, "args": args, "kwargs": kwargs, "qubits": qubits})
            return attr(*args, **kwargs)

        return wrapped


@contextmanager
def trace_circuit():
    """
    Context manager that temporarily makes `tc.Circuit(...)` return a
    `_GateTapeProxy` instead of a plain circuit, so that calling the *real*
    ansatz-construction function (e.g.
    `construct_dyn_circuit_toriccodelattice_prob_resets`) inside the `with`
    block gives back a circuit whose full gate tape can be inspected
    afterward - with zero duplicated circuit-building logic.

    Usage:
        with trace_circuit() as get_tape:
            qc = ansatz._circuit(params)   # qc is a _GateTapeProxy
        tape = get_tape(qc)
    """
    tc.Circuit = _GateTapeProxy
    try:
        yield lambda proxy: proxy.tape
    finally:
        tc.Circuit = _RealCircuit


def backward_lightcone(
    tape: Sequence[Dict[str, Any]], target_qubits: Sequence[int]
) -> Tuple[Set[int], List[int]]:
    """
    Walk `tape` in reverse (last-applied gate first), growing the lightcone
    qubit set every time a gate touches it. Returns the final qubit set and
    the indices (in original, forward order) of the gates that must be kept.
    """
    lightcone: Set[int] = set(target_qubits)
    kept_indices: List[int] = []
    for idx in range(len(tape) - 1, -1, -1):
        qubits = tape[idx]["qubits"]
        if lightcone.intersection(qubits):
            lightcone.update(qubits)
            kept_indices.append(idx)
    kept_indices.reverse()
    return lightcone, kept_indices


def build_reduced_circuit(
    tape: Sequence[Dict[str, Any]], kept_indices: Sequence[int], lightcone_qubits: Set[int]
):
    """
    Replay only the kept tape entries onto a fresh, smaller tc.Circuit whose
    qubits are `sorted(lightcone_qubits)` remapped to a compact 0..k-1
    indexing. Returns (reduced_circuit, qubit_map) where qubit_map sends
    original qubit indices to their compact index in the reduced circuit.
    """
    ordered_qubits = sorted(lightcone_qubits)
    qubit_map = {orig: new for new, orig in enumerate(ordered_qubits)}

    qc = _RealCircuit(len(ordered_qubits))
    for idx in kept_indices:
        entry = tape[idx]
        name, args, kwargs = entry["name"], entry["args"], entry["kwargs"]
        method = getattr(qc, name)
        if "indices" in kwargs:
            new_kwargs = dict(kwargs)
            new_kwargs["indices"] = [qubit_map[q] for q in entry["qubits"]]
            method(*args, **new_kwargs)
        else:
            new_args = tuple(qubit_map[a] if isinstance(a, (int, np.integer)) else a for a in args)
            method(*new_args, **kwargs)
    return qc, qubit_map


def _term_qubits(ops) -> List[int]:
    """Extract the flat list of qubit indices a HamiltonianTerm's `ops` acts on."""
    return [q for _gate, qubit_list in ops for q in qubit_list]


def report_lightcones_for_ansatz(ansatz, params, verify_terms: int = 5) -> List[Dict[str, Any]]:
    """
    Trace `ansatz._circuit(params)` once, then for every term in
    `ansatz._hamiltonian_terms()` compute its backward lightcone and report
    the qubit-count/gate-count reduction. For the first `verify_terms` terms,
    also builds the reduced circuit and checks its expectation value against
    the full circuit's, confirming the reduction is exact.

    Returns a list of per-term dicts: label, family, n_qubits_total,
    n_qubits_lightcone, n_gates_total, n_gates_kept, and (for verified terms)
    verified: bool.
    """
    with trace_circuit() as get_tape:
        qc = ansatz._circuit(params)
    tape = get_tape(qc)

    terms = ansatz._hamiltonian_terms()
    labels = ansatz.term_labels() if hasattr(ansatz, "term_labels") else [(f"term_{i}", "unknown") for i in range(len(terms))]

    results: List[Dict[str, Any]] = []
    for i, ((ops, _coeff), (label, family)) in enumerate(zip(terms, labels)):
        target_qubits = _term_qubits(ops)
        lightcone_qubits, kept_indices = backward_lightcone(tape, target_qubits)

        record: Dict[str, Any] = {
            "label": label,
            "family": family,
            "n_qubits_total": qc.nqubits,
            "n_qubits_lightcone": len(lightcone_qubits),
            "n_gates_total": len(tape),
            "n_gates_kept": len(kept_indices),
        }

        if i < verify_terms:
            reduced_qc, qubit_map = build_reduced_circuit(tape, kept_indices, lightcone_qubits)
            reduced_ops = tuple((gate, [qubit_map[q] for q in qubit_list]) for gate, qubit_list in ops)
            backend = tc.backend
            full_value = float(np.asarray(backend.numpy(backend.real(qc.expectation(*ops)))))
            reduced_value = float(np.asarray(backend.numpy(backend.real(reduced_qc.expectation(*reduced_ops)))))
            record["verified"] = bool(np.allclose(full_value, reduced_value, atol=1e-10))
            record["full_value"] = full_value
            record["reduced_value"] = reduced_value

        results.append(record)

    return results


def plot_term_lightcone(ansatz, params, term_index: int, save_path: str = None):
    """
    Draw the backward-lightcone diagram (see `lightcone_plotting.plot_lightcone`)
    for a single Hamiltonian term, computed via this module's conservative
    tracer. Unlike the Qiskit version, no gate-matching is needed here -
    `kept_indices` already tells us directly, by position, which of `tape`'s
    entries survive.
    """
    with trace_circuit() as get_tape:
        qc = ansatz._circuit(params)
    tape = get_tape(qc)

    terms = ansatz._hamiltonian_terms()
    labels = ansatz.term_labels() if hasattr(ansatz, "term_labels") else [(f"term_{i}", "unknown") for i in range(len(terms))]
    ops, _coeff = terms[term_index]
    label, _family = labels[term_index]

    target_qubits = _term_qubits(ops)
    _lightcone_qubits, kept_indices = backward_lightcone(tape, target_qubits)
    kept_set = set(kept_indices)

    gate_qubits = [entry["qubits"] for entry in tape]
    kept_mask = [i in kept_set for i in range(len(tape))]

    return plot_lightcone(
        n_qubits=qc.nqubits,
        gate_qubits=gate_qubits,
        kept_mask=kept_mask,
        target_qubits=target_qubits,
        title=f"{label} — custom conservative tracer ({len(kept_indices)}/{len(tape)} gates kept)",
        save_path=save_path,
    )


def print_lightcone_summary(results: List[Dict[str, Any]]) -> None:
    """Print a per-family summary table from `report_lightcones_for_ansatz`'s output."""
    families: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        families.setdefault(r["family"], []).append(r)

    n_qubits_total = results[0]["n_qubits_total"] if results else 0
    n_gates_total = results[0]["n_gates_total"] if results else 0

    print(f"{'family':<12} {'n_terms':>8} {'avg lightcone qubits':>22} {'avg gates kept':>16}")
    for family, records in families.items():
        avg_qubits = sum(r["n_qubits_lightcone"] for r in records) / len(records)
        avg_gates = sum(r["n_gates_kept"] for r in records) / len(records)
        print(f"{family:<12} {len(records):>8} {avg_qubits:>18.1f}/{n_qubits_total:<3} {avg_gates:>12.1f}/{n_gates_total}")

    verified = [r for r in results if "verified" in r]
    if verified:
        n_ok = sum(r["verified"] for r in verified)
        print(f"\ncorrectness check: {n_ok}/{len(verified)} spot-checked terms had reduced-circuit "
              f"expectation values matching the full circuit to 1e-10")
        for r in verified:
            status = "OK" if r["verified"] else "MISMATCH"
            print(f"  [{status}] {r['label']}: full={r['full_value']:.10f} reduced={r['reduced_value']:.10f}")
