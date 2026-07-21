# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Compare backward lightcones (via Qiskit's LightCone pass) across different
`reset_layers` configurations for the reset-capable toric-code ansatz:
resets on layer 0 only, layer 1 only, both layers, or no resets at all.

Note the total qubit/gate counts differ between configurations too - more
active reset layers means more probabilistic-reset ancillas
(nancillas = 2 * nresets_per_layer * len(reset_layers)), so a bigger full
circuit to begin with. The per-config totals are printed alongside the
lightcone averages so that's visible rather than hidden.

Purely read-only with respect to training - does not touch the optimizer.
"""

from jax import numpy as jnp

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.lightcone_analysis_qiskit import report_lightcones_for_ansatz

Lx = 2
Ly = 2
nlayers = 2
h = 0

RESET_LAYER_CONFIGS = [[], [0], [1], [0, 1]]

if __name__ == "__main__":
    for reset_layers in RESET_LAYER_CONFIGS:
        ansatz = ToricCodeAnsatz(
            Lx=Lx,
            Ly=Ly,
            nlayers=nlayers,
            h=h,
            trials=1,
            maxiter=1,
            use_reset_capable_ansatz=True,
            reset_layers=reset_layers,
            sparse=False,
        )

        params = jnp.array(ansatz.initparams[0])
        results = report_lightcones_for_ansatz(ansatz, params)

        families = {}
        for r in results:
            families.setdefault(r["family"], []).append(r)

        n_qubits_total = results[0]["n_qubits_total"] if results else 0
        n_gates_total = results[0]["n_gates_total"] if results else 0

        print(f"=== reset_layers={reset_layers!r}  (n_qubits_total={n_qubits_total}, n_gates_total={n_gates_total}) ===")
        print(f"{'family':<12} {'n_terms':>8} {'avg lightcone qubits':>22} {'avg gates kept':>16}")
        for family, records in families.items():
            avg_qubits = sum(rr["n_qubits_lightcone"] for rr in records) / len(records)
            avg_gates = sum(rr["n_gates_kept"] for rr in records) / len(records)
            print(f"{family:<12} {len(records):>8} {avg_qubits:>18.1f}/{n_qubits_total:<3} {avg_gates:>12.1f}/{n_gates_total}")
        print()
