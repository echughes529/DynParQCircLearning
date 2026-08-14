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
    construct_dyn_circuit_brickwork,
    make_split_conf,
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
    use_prob_resets_ansatz: bool = True
    which_qubits_for_prob_reset: Optional[list] = None
    prob_reset_direction: int = 1 # vertical
    reset_layers: Optional[list] = None

    use_small_angle_initialization: bool = False
    range_initial_parameters: float = 0

    unitary: bool = False

    bond_dim: Optional[int] = None
    use_optimal_ordering: bool = True

    def __post_init__(self):
        self.split_conf = make_split_conf(self.bond_dim)

        self.lattice = ToricCode(self.Lx, self.Ly)
        self.nplaquettes = (self.Lx - 1) * (self.Ly - 1)

        if self.use_small_angle_initialization:
            self.nparams = 3 * self.lattice.num_qubits * (self.nlayers + 1)
            self.nancillas = 0
        elif self.use_prob_resets_ansatz:
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

            self._build_interleaved_ordering()
        elif self.unitary:
            self.nparams = self.nplaquettes * 4 * 9 * self.nlayers + 3 * self.lattice.num_qubits
            self.nancillas = 0
        else:
            self.nmeasurements = self.nplaquettes * (self.nlayers // self.howoften_toreset)
            self.nancillas = self.nplaquettes + self.nmeasurements
            self.nparams = self.nplaquettes * 4 * 9 * self.nlayers + 3 * self.lattice.num_qubits
        
        print(self.__dict__)
        super().__post_init__()

    def _build_interleaved_ordering(self):
        """
        Build an MPS qubit ordering.

        When use_optimal_ordering is True (default): random search over system
        qubit permutations to minimise the maximum two-qubit gate distance
        across all Cartan block (claw) gates and CCX reset gates, inserting
        each reset's ancilla pair immediately after its system qubit.

        When False: system qubits are kept in natural order [0, 1, ..., nq-1]
        and no search is run; all ancilla pairs are appended after the system
        qubits, at the end of the chain.
        """
        import random as _rng

        nq = self.lattice.num_qubits
        toriccode = self.lattice

        if self.which_qubits_for_prob_reset is None:
            reset_qubits = [toriccode.qubit_index(x, y, self.prob_reset_direction)
                            for x in range(self.Lx - 1) for y in range(self.Ly - 1)]
            reset_qubits = [q for q in reset_qubits if q is not None]
        else:
            reset_qubits = list(self.which_qubits_for_prob_reset)

        reset_set = set(reset_qubits)
        n_reset_layers = len(self.active_reset_layers)

        claw_edges = toriccode.all_claws()

        def _max_claw_dist(sys_order):
            s2m = {}
            pos = 0
            for sq in sys_order:
                s2m[sq] = pos
                pos += 1
                if sq in reset_set:
                    pos += 2 * n_reset_layers
            return max(abs(s2m[a] - s2m[b]) for a, b in claw_edges)

        natural_order = list(range(nq))
        if self.use_optimal_ordering:
            _rng.seed(42)
            best_order = list(natural_order)
            best_dist = _max_claw_dist(best_order)
            n_trials = min(2_000_000, max(500_000, nq * 100_000))
            for _ in range(n_trials):
                order = list(range(nq))
                _rng.shuffle(order)
                d = _max_claw_dist(order)
                if d < best_dist:
                    best_dist = d
                    best_order = order[:]
        else:
            best_order = natural_order
            n_trials = 0

        # Assign each reset gate a stable ancilla_idx in circuit-application
        # order, independent of where qubits end up placed in the chain.
        ancilla_idx_of = {}   # (layer, sq) -> ancilla_idx
        reset_qubit_of = []   # ancilla_idx -> sq
        ancilla_idx = 0
        for _layer in self.active_reset_layers:
            for sq in reset_qubits:
                ancilla_idx_of[(_layer, sq)] = ancilla_idx
                reset_qubit_of.append(sq)
                ancilla_idx += 1
        n_ancilla_pairs = ancilla_idx

        sys_to_mps = {}
        ancilla_mps_positions = []
        mps_reset_info = [None] * n_ancilla_pairs

        if self.use_optimal_ordering:
            # Inline: each reset qubit's ancilla pair(s) sit immediately after it.
            pending_by_sq = {}
            for (_layer, sq), a_idx in ancilla_idx_of.items():
                pending_by_sq.setdefault(sq, []).append(a_idx)

            pos = 0
            for sq in best_order:
                sys_to_mps[sq] = pos
                pos += 1
                for a_idx in pending_by_sq.get(sq, []):
                    prob_mps, purif_mps = pos, pos + 1
                    ancilla_mps_positions.extend([prob_mps, purif_mps])
                    mps_reset_info[a_idx] = (sys_to_mps[sq], prob_mps, purif_mps)
                    pos += 2
        else:
            # Natural order: all system qubits first, ancillas appended at the end.
            for pos, sq in enumerate(best_order):
                sys_to_mps[sq] = pos
            pos = nq
            for a_idx in range(n_ancilla_pairs):
                sq = reset_qubit_of[a_idx]
                prob_mps, purif_mps = pos, pos + 1
                ancilla_mps_positions.extend([prob_mps, purif_mps])
                mps_reset_info[a_idx] = (sys_to_mps[sq], prob_mps, purif_mps)
                pos += 2

        self.sys_to_mps = sys_to_mps
        self.ancilla_mps_positions = ancilla_mps_positions
        self.n_mps_qubits = pos
        self.sys_mps_positions = [sys_to_mps[i] for i in range(nq)]

        # Pre-remap claws
        claws = toriccode.all_claws()
        nplaq = (self.Lx - 1) * (self.Ly - 1)
        claws = [claws[i::3][j] for i in range(3) for j in range(nplaq)]
        self.mps_claws = [(sys_to_mps[a], sys_to_mps[b]) for a, b in claws]

        # Build reset triples in circuit-application order (layer by layer),
        # looking each one up by its stable ancilla_idx.
        self.mps_reset_qubits = [
            mps_reset_info[ancilla_idx_of[(_layer, sq)]]
            for _layer in self.active_reset_layers
            for sq in reset_qubits
        ]

        max_claw = max(abs(sys_to_mps[a] - sys_to_mps[b]) for a, b in claw_edges)
        max_ccx = max((abs(s - p) for s, p, _ in mps_reset_info), default=0)
        if self.use_optimal_ordering:
            ordering_desc = f"optimised ordering ({n_trials} random trials)"
        else:
            ordering_desc = "natural ordering (no search, ancillas appended at end)"
        print(f"MPS {ordering_desc}: "
              f"{self.n_mps_qubits} qubits ({nq} system + {self.nancillas} ancilla)")
        print(f"  System qubit order: {best_order}")
        print(f"  Max claw distance: {max_claw}, max CCX distance: {max_ccx}")
        print(f"  sys_to_mps: {self.sys_to_mps}")

    def __hash__(self):
        reset_layers_key = None if self.reset_layers is None else tuple(self.active_reset_layers)
        return hash((self.Lx, self.Ly, self.nlayers, self.howoften_toreset, self.h,
                    self.trials, self.maxiter, self.howoften_tosave, self.learning_rate,
                    self.sparse, self.use_prob_resets_ansatz, self.prob_reset_direction,
                    reset_layers_key, self.use_mps, self.normalize_state, self.bond_dim))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_full_hamiltonian(self):
        """Build the full sparse Hamiltonian for toric code.
        Not compatible with interleaved MPS ordering (use_prob_resets_ansatz + sparse).
        """
        if self.use_prob_resets_ansatz:
            raise NotImplementedError(
                "Sparse Hamiltonian is not supported with interleaved MPS ordering "
                "(use_prob_resets_ansatz=True). Use use_mps=True with term-by-term expectation."
            )
        strings, weights = self.lattice.hamiltonian_tc(1 - self.h, self.nancillas)
        perturbed_strings, perturbed_weights = self.lattice.hamiltonian_tc_perturbation(
            self.h, self.nancillas
        )
        strings.extend(perturbed_strings)
        weights = np.concatenate((weights, perturbed_weights))
        return qu.PauliStringSum2COO(strings, weights)
        
    @property
    def reset_param_slice(self):
        """
        Slice of the flat parameter vector holding the probabilistic-reset
        theta parameters, or None if this ansatz has no reset parameters.

        Layout (only valid when use_prob_resets_ansatz is True):
            [0, n_two_qubit)                        -> Cartan-block params
            [n_two_qubit, n_two_qubit+total_resets)  -> reset-theta params
            [n_two_qubit+total_resets, nparams)      -> final single-qubit params
        """
        if not getattr(self, "use_prob_resets_ansatz", False):
            return None
        total_resets = getattr(self, "total_resets", 0)
        if not total_resets:
            return None
        n_two_qubit = self.nplaquettes * 3 * 9 * self.nlayers
        return slice(n_two_qubit, n_two_qubit + total_resets)

    def _initialise_parameters(self):
        """Initialize parameters with optional small angle range."""
        if self.seed is None:
            self.seed = int(np.random.randint(1e5))
        print(f"[{type(self).__name__}] parameter init seed: {self.seed} "
              f"(pass seed={self.seed} to reproduce this run)")
        key = jax.random.PRNGKey(self.seed)

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
        # with values in [0, pi/2]
        if self.use_prob_resets_ansatz and self.reset_param_slice is not None:
            n_reset = self.total_resets
            reset_vals = jax.random.uniform(
                key_reset,
                shape=[self.trials, n_reset],
                minval= jnp.pi/2.0 - 0.2,
                maxval= jnp.pi/2.0- 0.1
            )

            params = params.at[:, self.reset_param_slice].set(reset_vals)

        return params

    def _circuit(self, params, *args, seed=None):
        """Construct the circuit based on ansatz type."""
        if self.use_small_angle_initialization:
            return construct_smallangle_init_toriccodelattice(
                params, self.Lx, self.Ly, self.nlayers, split_conf=self.split_conf,
                normalize_state=self.normalize_state,
            )
        elif self.use_prob_resets_ansatz:
            return construct_dyn_circuit_toriccodelattice_prob_resets(
                params,
                nlayers=self.nlayers,
                nparams=self.nparams,
                n_mps_qubits=self.n_mps_qubits,
                mps_claws=self.mps_claws,
                mps_reset_qubits=self.mps_reset_qubits,
                active_reset_layers=self.active_reset_layers,
                sys_mps_positions=self.sys_mps_positions,
                split_conf=self.split_conf,
                normalize_state=self.normalize_state,
            )
        elif self.unitary:
            return construct_unitary_circuit_toriccodelattice(
                params, self.Lx, self.Ly, self.nlayers, split_conf=self.split_conf,
                normalize_state=self.normalize_state,
            )
        else:
            return construct_dyn_circuit_toriccodelattice(
                params, self.Lx, self.Ly, self.nlayers, self.howoften_toreset,
                split_conf=self.split_conf, normalize_state=self.normalize_state,
            )

    def _remap_qubit(self, logical_idx):
        """Map a logical system qubit index to its MPS chain position."""
        if self.use_prob_resets_ansatz:
            return self.sys_to_mps[logical_idx]
        return logical_idx

    def _hamiltonian_terms(self) -> List[HamiltonianTerm]:
        """Return Hamiltonian terms for toric code, using MPS positions."""
        terms: List[HamiltonianTerm] = []
        t = self.lattice
        stars = t.all_stars()
        plqs = t.all_plaquettes()

        for s in stars:
            ops = tuple((tc.gates.z(), [self._remap_qubit(op)]) for op in s)
            terms.append((ops, -(1.0 - self.h)))

        for p in plqs:
            ops = tuple((tc.gates.x(), [self._remap_qubit(op)]) for op in p)
            terms.append((ops, -(1.0 - self.h)))

        for i in range(t.num_qubits):
            ops = ((tc.gates.z(), [self._remap_qubit(i)]),)
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
    bond_dim: Optional[int] = None

    def __post_init__(self):
        self.split_conf = make_split_conf(self.bond_dim)

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
        return hash((self.num_qubits, self.nlayers, self.howoften_toreset, self.h,
                     self.trials, self.maxiter, self.howoften_tosave,
                     self.learning_rate, self.sparse, self.use_mps,
                     self.normalize_state, self.bond_dim))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_full_hamiltonian(self):
        self.fullham = self.lattice.tfi_hamiltonian_dense(0,False,False)
        return self.fullham

    def _circuit(self, params,seeds=None):
        sc = self._build_scaffold()
        if self.unitary:
            qc = construct_unitary_circuit_brickwork(
                params, sc, split_conf=self.split_conf,
                normalize_state=self.normalize_state,
            )
        # elif seeds is None:
        #     qc = construct_dyn_circuit_brickwork_seeded(params,self.Lx,self.Ly,self.nlayers,self.howoften_toreset)
        else:
            qc = construct_dyn_circuit_brickwork(
                params, sc, split_conf=self.split_conf,
                normalize_state=self.normalize_state,
            )
        
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
