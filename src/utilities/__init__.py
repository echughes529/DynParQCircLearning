from __future__ import annotations
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector, partial_trace, entropy
import numpy as np
import jax.numpy as jnp
import jax.experimental.sparse as jss

I = Pauli('I')
Z = Pauli('Z')
X = Pauli('X')
Y = Pauli('Y')

def sparse_to_pauli_string(indices, qubits, pauli='X'):
    string = ''
    for i in range(qubits):
        if i in indices:
            string = pauli + string # Needed because of little endian convention in Qiskit
        else:
            string = 'I' + string
    return string

def array_to_tc_structure(arrays,nqubits,pauli='X'):
    structure = []
    for ar in arrays:
        str = np.zeros(nqubits)
        for idx in ar:
            if pauli == 'X':
                str[idx] = 1
            if pauli == 'Y':
                str[idx] = 2
            if pauli == 'Z':
                str[idx] = 3
        structure.append(str)
        
    return list(structure)

def scipy_to_bcoo(mat) -> jss.BCOO:
    """Convert a CSR/COO matrix to JAX BCOO (CPU‑only, works with JIT)."""
    # COO format (row, col, data)
    coo = mat.tocoo()
    data = jnp.asarray(coo.data)
    indices = jnp.vstack([jnp.asarray(coo.row), jnp.asarray(coo.col)]).T
    shape = coo.shape
    return jss.BCOO((data, indices), shape=shape)

__all__ = [
    "Pauli",
    "SparsePauliOp",
    "Statevector",
    "partial_trace",
    "entropy",
    "sparse_to_pauli_string",
    "array_to_tc_structure",
    "scipy_to_bcoo"
]