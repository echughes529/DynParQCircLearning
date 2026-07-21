# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Diagnostic: how much does each Hamiltonian term's backward lightcone shrink
the circuit, for the reset-capable toric-code ansatz?

Purely read-only with respect to training - does not touch the optimizer.
"""

from jax import numpy as jnp

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.lightcone_analysis import print_lightcone_summary, report_lightcones_for_ansatz

Lx = 2
Ly = 2
nlayers = 2
h = 0.1

if __name__ == "__main__":
    ansatz = ToricCodeAnsatz(
        Lx=Lx,
        Ly=Ly,
        nlayers=nlayers,
        h=h,
        trials=1,
        maxiter=1,
        use_reset_capable_ansatz=True,
        reset_layers=[0, 1],
        sparse=False,
    )

    params = jnp.array(ansatz.initparams[0])
    results = report_lightcones_for_ansatz(ansatz, params, verify_terms=5)
    print_lightcone_summary(results)
