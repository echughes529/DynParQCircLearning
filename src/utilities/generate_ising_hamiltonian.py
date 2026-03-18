# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transverse field Ising Hamiltonian Generation."""

import numpy as np
import tensorcircuit as tc
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
# dtype = np.complex128
from typing import List, Tuple, Literal
import bisect
from dataclasses import dataclass
from src.utilities import *


xx = tc.gates._xx_matrix  # xx gate matrix to be utilized

@dataclass
class OneDimTFIM():
    num_qubits : int = 2
    n_ancillas :int = 0
    J : jnp.float_ = 1.0 # coupling term X_iX_j
    h : jnp.float_ = 0.5 # coupling term Z_i

    def name(self):
        return "1dtfim"

    def tfi_hamiltonian_dense(self, 
                        n_ancillas: int = 0,
                        extend: bool = False,
                        dense: bool = True,
                        individual_paulis: bool = False):
        """

        :param n: Number of qubits.
        :param n_ancillas: Number of ancillary qubits used to purify the reset operation.
        :param extend: Whether to extend or not the Hamiltonian so that it is supported on system + ancilla qubits
        :param dense: Boolean value that determines whether the Hamiltonian is returned in dense or sparse format
        :param individual_paulis: Boolean value that determines whether the individual Paulis making up the Hamiltonian are
                                returned alongside the complete Hamiltonian
        :return: Transverse field Ising Hamiltonian on n qubits
        """
        terms = []
        weights = []

        terms = self.tfi_hamiltonian_terms("tc_sparse")
        if dense:
            hamiltonian_constructor = tc.quantum.PauliStringSum2Dense
        else:
            hamiltonian_constructor = tc.quantum.PauliStringSum2COO_numpy

        hamiltonian = hamiltonian_constructor(*terms)
        hamiltonian = scipy_to_bcoo(hamiltonian)
        return hamiltonian

    def tfi_hamiltonian_terms(self, backend="tc"):
        """
        H = -h * sum_i Z_i  - J * sum_<i,j> X_i X_{j} for i, j neighbours when they're both system qubits.
        """
        sys_phys, _ = system_and_ancilla_physical_indices(self.num_qubits, self.n_ancillas)
        if backend == "tc":
            terms: list[tuple[tuple[tuple, list[int]], float]] = []
            for p in sys_phys:
                ops = ((tc.gates.z(), [p]),)
                terms.append((ops, -self.h))

            for k in range(len(sys_phys) - 1):
                i = sys_phys[k]
                j = sys_phys[k + 1]
                ops = ((tc.gates.x(), [i]), (tc.gates.x(), [j]))
                terms.append((ops, -self.J))

        # elif backend=="qiskit":
        #     terms: list[tuple[str, list[int], float]] = []
        #     for phys_idx in sys_phys:
        #         terms.append(("Z", [phys_idx], -self.h))
        #     for k in range(len(sys_phys) - 1):
        #         i_phys = sys_phys[k]
        #         j_phys = sys_phys[k + 1]
        #         terms.append(("XX", [i_phys, j_phys], -self.J))

        # elif backend=="qiskit_sparse":
        #     pauli_labels: list[str] = []
        #     coeffs:      list[complex] = []         
        #     for phys_idx in sys_phys:
        #         label = ["I"] * (self.num_qubits + self.n_ancillas)
        #         label[phys_idx] = "Z"
        #         pauli_labels.append("".join(label))
        #         coeffs.append(-self.h)                   
        #     for k in range(len(sys_phys) - 1):
        #         i_phys = sys_phys[k]
        #         j_phys = sys_phys[k + 1]
        #         label = ["I"] * (self.num_qubits + self.n_ancillas)
        #         label[i_phys] = "X"
        #         label[j_phys] = "X"
        #         pauli_labels.append("".join(label))
        #         coeffs.append(-self.J)
            
        #     terms = (pauli_labels, coeffs)
        
        elif backend=="tc_sparse":
            ls = []
            weights = []

            z_indices = [[idx] for idx in sys_phys]
            z_rows = array_to_tc_structure(z_indices, self.num_qubits+self.n_ancillas, pauli="Z")
            z_weights = np.full(len(z_rows), -self.h)

            xx_indices = [[sys_phys[k], sys_phys[k + 1]] for k in range(len(sys_phys) - 1)]
            xx_rows = array_to_tc_structure(xx_indices, self.num_qubits+self.n_ancillas, pauli="X")
            xx_weights = np.full(len(xx_rows), -self.J)
            ls = np.vstack(z_rows + xx_rows)
            weights = np.concatenate([z_weights, xx_weights])
            terms = (ls, weights)
        return terms

    def all_claws_measurements(self):
        sys_phys, anc_phys = system_and_ancilla_physical_indices(self.num_qubits,self.n_ancillas)
        claws: List[Tuple[int, int]] = []
        for k in range(len(sys_phys) - 1):
            i = sys_phys[k]
            j = sys_phys[k + 1]
            claws.append((i, j))
        
        if not anc_phys:
            return claws

        for a in anc_phys:
            pos = bisect.bisect_left(sys_phys, a)
            if pos > 0:
                claws.append((sys_phys[pos - 1], a))

        return claws
    
    def ancillas(self):
        return system_and_ancilla_physical_indices(self.num_qubits,self.n_ancillas)[1]

def system_and_ancilla_physical_indices(nq: int, n_ancillas: int) -> Tuple[List[int], List[int]]:
    """
    Returns (system_phys, ancilla_phys): the physical site indices (0..N-1) for the
    system and ancilla rails.
    interleave: [q0, a0, q1, a1, q2, a2, ...] until ancillas run out, then just system
    """
    if n_ancillas is None or n_ancillas <= 0:
        return list(range(nq)), []

    # interleave
    order_roles: List[str] = []
    sys_phys: List[int] = []
    anc_phys: List[int] = []

    a = 0
    for q in range(nq):
        sys_phys.append(len(order_roles))
        order_roles.append("sys")
        # place an ancilla after roughly every other system site, until we run out
        if a < n_ancillas and (q % 2 == 0 or nq <= 2 * n_ancillas):
            anc_phys.append(len(order_roles))
            order_roles.append("anc")
            a += 1

    # any leftover ancillas go at the end (rare unless n_ancillas >> nq/2)
    while a < n_ancillas:
        anc_phys.append(len(order_roles))
        order_roles.append("anc")
        a += 1

    return sys_phys, anc_phys

def tfi_hamiltonian(n: int,
                    n_resets: int = 0,
                    extend: bool = False,
                    dense: bool = True,
                    individual_paulis: bool = False):
    """

    :param n: Number of qubits.
    :param n_resets: Number of ancillary qubits used to purify the reset operation.
    :param extend: Whether to extend the Hamiltonian to be supported on the virtual ancillas 
    :param dense: Boolean value that determines whether the Hamiltonian is returned in dense or sparse format
    :param individual_paulis: Boolean value that determines whether the individual Paulis making up the Hamiltonian are
                              returned alongside the complete Hamiltonian
    :return: Transverse field Ising Hamiltonian on n qubits
    """
    h = []
    w = []

    ### Z
    for i in range(n):
        h.append([])
        w.append(-0.5)  # weight
        for j in range(n):
            if j == i:
                h[i].append(3)
            else:
                h[i].append(0)
    ### XX
    for i in range(n):
        h.append([])
        w.append(-1.0)  # weight
        for j in range(n):
            if j == (i + 1) % n or j == i:
                h[i + n].append(1)
            else:
                h[i + n].append(0)

    ### I^n
    if dense:
        f = tc.quantum.PauliStringSum2Dense
    else:
        f = tc.quantum.PauliStringSum2COO_numpy
    if extend:
        for element in h:
            for k in range(n_resets):
                element.append(0)
        hamiltonian = f(h,w)
    else:
        hamiltonian = f(h, w)

    if individual_paulis:
        z_array = []
        z_pauli = jnp.asarray([[1, 0], [0, -1]])
        paulis = []
        for i in range(n):
            z_array.append([i])
            paulis.append(z_pauli)
        x_array = []
        x_pauli = jnp.asarray([[0, 1], [1,0]])
        xx_pauli = jnp.asarray(np.kron(x_pauli, x_pauli))

        for i in range(n-1):
            x_array.append([i, i+1])
            paulis.append(xx_pauli)
        pauli_indices = z_array+x_array


        return hamiltonian, pauli_indices, paulis
    else:
        return hamiltonian
