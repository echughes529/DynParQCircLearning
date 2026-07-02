# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Concrete ansatz classes inheriting from VariationalAnsatz."""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import jax
from jax import numpy as jnp
import tensorcircuit as tc
import tensorcircuit.quantum as qu

from src.find_gs import VariationalAnsatz, HamiltonianTerm
from src.utilities.generate_toric_code_hamiltonian import ToricCode
from src.utilities.generate_ising_hamiltonian import OneDimTFIM
from src.utilities.generate_ansatz import (
    construct_smallangle_init_toriccodelattice,
    construct_dyn_circuit_toriccodelattice_prob_resets,
    construct_unitary_circuit_toriccodelattice,
    construct_dyn_circuit_toriccodelattice,
    get_nresets_per_layer_toriccode,
    construct_unitary_circuit_brickwork,
    construct_dyn_circuit_brickwork
)


@dataclass
class ToricCodeAnsatz(VariationalAnsatz):
    """
    Toric code ansatz with various circuit construction options.
    """
    Lx: int = 2
    Ly: int = 2
    nlayers: int = 2
    howoften_toreset: int = 1
    h: float = 0.0
    
    # Ansatz selection
    use_prob_resets: bool = True
    which_qubits_for_prob_reset: Optional[list] = None
    prob_reset_direction: int = 1 # vertical
    reset_layers: Optional[list] = None

    use_small_angle_initialization: bool = False
    range_initial_parameters: float = 0

    unitary: bool = False

    def __post_init__(self):
        self.lattice = ToricCode(self.Lx, self.Ly)
        self.nplaquettes = (self.Lx - 1) * (self.Ly - 1)
        
        if self.use_small_angle_initialization:
            self.nparams = 3 * self.lattice.num_qubits * (self.nlayers + 1)
            self.nancillas = 0
        elif self.use_prob_resets:
            if self.which_qubits_for_prob_reset is None:
                self.nresets_per_layer = get_nresets_per_layer_toriccode(
                    self.Lx, self.Ly, reset_direction=self.prob_reset_direction
                )
            else:
                self.nresets_per_layer = len(self.which_qubits_for_prob_reset)

            if self.reset_layers is None:
                self.active_reset_layers = list(range(self.nlayers))
            else:
                self.active_reset_layers = sorted(set(int(layer) for layer in self.reset_layers))
                invalid_layers = [layer for layer in self.active_reset_layers if layer < 0 or layer >= self.nlayers]
                if invalid_layers:
                    raise ValueError(
                        f"reset_layers contains invalid layer indices: {invalid_layers}. "
                        f"Valid range is 0 to {self.nlayers - 1}."
                    )

            self.total_resets = self.nresets_per_layer * len(self.active_reset_layers)
            self.nancillas = 2 * self.total_resets
            self.nparams = (self.nplaquettes * 3 * 9 * self.nlayers + 
                          self.total_resets + 3 * self.lattice.num_qubits)
        elif self.unitary:
            self.nparams = self.nplaquettes * 4 * 9 * self.nlayers + 3 * self.lattice.num_qubits
            self.nancillas = 0
        else:
            self.nmeasurements = self.nplaquettes * (self.nlayers // self.howoften_toreset)
            self.nancillas = self.nplaquettes + self.nmeasurements
            self.nparams = self.nplaquettes * 4 * 9 * self.nlayers + 3 * self.lattice.num_qubits
        
        print(self.__dict__)
        super().__post_init__()

    @property
    def reset_param_slice(self):
        """
        Slice of the flat parameter vector holding the probabilistic-reset
        theta parameters, or None if this ansatz has no reset parameters.

        Layout (only valid when use_prob_resets is True):
            [0, n_two_qubit)                        -> Cartan-block params
            [n_two_qubit, n_two_qubit+total_resets)  -> reset-theta params
            [n_two_qubit+total_resets, nparams)      -> final single-qubit params
        """
        if not getattr(self, "use_prob_resets", False):
            return None
        total_resets = getattr(self, "total_resets", 0)
        if not total_resets:
            return None
        n_two_qubit = self.nplaquettes * 3 * 9 * self.nlayers
        return slice(n_two_qubit, n_two_qubit + total_resets)

    def __hash__(self):
        reset_layers_key = None if self.reset_layers is None else tuple(self.active_reset_layers)
        return hash((self.Lx, self.Ly, self.nlayers, self.howoften_toreset, self.h, 
                    self.trials, self.maxiter, self.howoften_tosave, self.learning_rate,
                    self.sparse, self.use_prob_resets, self.prob_reset_direction,
                    reset_layers_key))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_full_hamiltonian(self):
        """Build the full sparse Hamiltonian for toric code."""
        strings, weights = self.lattice.hamiltonian_tc(1 - self.h, self.nancillas)
        perturbed_strings, perturbed_weights = self.lattice.hamiltonian_tc_perturbation(
            self.h, self.nancillas
        )
        strings.extend(perturbed_strings)
        weights = np.concatenate((weights, perturbed_weights))
        return qu.PauliStringSum2COO(strings, weights)
        
    def _initialise_parameters(self):
        """Initialize parameters with optional small angle range."""
        randint = np.random.randint(1e5)
        key = jax.random.PRNGKey(randint)

        if self.use_small_angle_initialization:
            return jax.random.uniform(
                key,
                shape=[self.trials, self.nparams],
                minval=0,
                maxval=self.range_initial_parameters
            )

        # First initialise all parameters in [0, pi]
        key_all, key_reset = jax.random.split(key)
        params = jax.random.uniform(
            key_all,
            shape=[self.trials, self.nparams],
            minval=0.0,
            maxval=jnp.pi
        )

        # If using probabilistic resets, overwrite the reset parameters
        # with very small values in [0, 1e-5]---- changed to regular initialisation
        if self.use_prob_resets:
            n_reset = self.total_resets
            reset_vals = jax.random.uniform(
                key_reset,
                shape=[self.trials, n_reset],
                minval=0.0,
                maxval=jnp.pi # changed from small
            )

            # reset parameters sit between the two-qubit block and the final
            # single-qubit block
            n_two_qubit = self.nplaquettes * 3 * 9 * self.nlayers
            reset_start = n_two_qubit
            reset_end = reset_start + n_reset

            params = params.at[:, reset_start:reset_end].set(reset_vals)

        return params

    def _circuit(self, params, *args, seed=None):
        """Construct the circuit based on ansatz type."""
        if self.use_small_angle_initialization:
            return construct_smallangle_init_toriccodelattice(
                params, self.Lx, self.Ly, self.nlayers
            )
        elif self.use_prob_resets:
            return construct_dyn_circuit_toriccodelattice_prob_resets(
                params, self.Lx, self.Ly, self.nlayers,
                self.which_qubits_for_prob_reset,
                reset_direction=self.prob_reset_direction,
                reset_layers=self.active_reset_layers,
            )
        elif self.unitary:
            return construct_unitary_circuit_toriccodelattice(
                params, self.Lx, self.Ly, self.nlayers
            )
        else:
            return construct_dyn_circuit_toriccodelattice(
                params, self.Lx, self.Ly, self.nlayers, self.howoften_toreset
            )

    def _hamiltonian_terms(self) -> List[HamiltonianTerm]:
        """Return Hamiltonian terms for toric code."""
        terms: List[HamiltonianTerm] = []
        t = self.lattice
        stars = t.all_stars()
        plqs = t.all_plaquettes()
        
        for s in stars:
            ops = tuple((tc.gates.z(), [op]) for op in s)
            terms.append((ops, -(1.0 - self.h)))
        
        for p in plqs:
            ops = tuple((tc.gates.x(), [op]) for op in p)
            terms.append((ops, -(1.0 - self.h)))
        
        for i in range(t.num_qubits):
            ops = ((tc.gates.z(), [i]),)
            terms.append((ops, -self.h))
        
        return terms

@dataclass
class OneDBrickwork(VariationalAnsatz):
    """
    A class that manages all relevant settings, including optimization options.
    """
    num_qubits: int = 2
    nlayers: int = 2
    howoften_toreset: int = 1
    h : float = 0.0
    J : float = 1.0
    unitary : bool = False
    n_ancillas = None

    def __post_init__(self):
        if self.n_ancillas is None and not(self.unitary):
            self.n_ancillas = self.num_qubits-1
        if self.unitary:
            self.n_ancillas = 0
        self.lattice = OneDimTFIM(self.num_qubits,self.n_ancillas,self.J,self.h)
        if not self.unitary:
            self.nparams = 2*self.n_ancillas * 9*self.nlayers + 3*self.num_qubits # The 2 comes from the number of claws in each "plaquette" or per ancilla
        else:
            self.nparams = (self.num_qubits - 1) * 9*self.nlayers + 3*self.num_qubits
        self.nmeasurements = self.n_ancillas * (self.nlayers//self.howoften_toreset)
        # if not self.unitary:
        #     self.nancillas = self.nplaquettes + self.nplaquettes * (self.nlayers//self.howoften_toreset)
        # else:
        #     self.nancillas = self.nmeasurements
        # if self.trajectories:
        #     self.seeds = K.implicit_randu([10,self.nmeasurements])
        print(self.__dict__)
        super().__post_init__()


    def __hash__(self):
        return hash((self.num_qubits, self.nlayers, self.howoften_toreset, self.h, self.trials, self.maxiter, self.howoften_tosave, self.learning_rate, self.sparse))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_full_hamiltonian(self):
        self.fullham = self.lattice.tfi_hamiltonian_dense(0,False,False)
        return self.fullham

    def _circuit(self, params,seeds=None):
        sc = self._build_scaffold()
        if self.unitary:
            qc = construct_unitary_circuit_brickwork(params,sc)
        # elif seeds is None:
        #     qc = construct_dyn_circuit_brickwork_seeded(params,self.Lx,self.Ly,self.nlayers,self.howoften_toreset)
        else:
            qc = construct_dyn_circuit_brickwork(params,sc)
        
        return qc

    def _hamiltonian_terms(self,*args,**kwargs):
        return self.lattice.tfi_hamiltonian_terms(*args,**kwargs)

    def _build_scaffold(self):
        nq = int(self.num_qubits)
        n_ancillas = int(getattr(self, "n_ancillas", 0))
        nlayers = int(getattr(self, "nlayers", nq))
        nparams = int(getattr(self, "nparams", 1))
        howoften = int(getattr(self, "howoften_toreset", 3))
        backend = getattr(self,"backend", "tc")

        if getattr(self, "unitary", False):
            lattice = OneDimTFIM(nq)
            claws = lattice.all_claws_measurements()
            ancillas = []
        else:
            lattice = OneDimTFIM(nq,n_ancillas)
            claws = lattice.all_claws_measurements()
            ancillas = lattice.ancillas()
            # number of measurement outcomes you expect per sample:
            nmeas = n_ancillas * (nlayers // howoften) if (howoften > 0 and n_ancillas > 0) else 0

        return {
            "nq": nq,
            "n_ancillas": n_ancillas,
            "nlayers": nlayers,
            "nparams": nparams,
            "howoften": howoften,
            "claws": claws,
            "ancillas": ancillas,
            "backend": backend
        }

# Made with Bob
