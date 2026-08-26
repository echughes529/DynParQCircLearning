# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Find ground state for a given Hamiltonian using the dynamic parameterized quantum circuit ansatz."""
import abc
import os
import resource
import sys
import time
from functools import partial

from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, Tuple, List

import tensorcircuit as tc
import tensorcircuit.quantum as qu
import tensorcircuit.noisemodel as noisemodel
from tensorcircuit.templates.measurements import operator_expectation, sparse_expectation
from jax import config
from tqdm import tqdm
config.update("jax_enable_x64", True)
from jax import numpy as jnp
import cotengra as ctg
optr = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    parallel=False,
    minimize="combo",
    # max_time=120,
    max_repeats=200,
    progbar=True,
    directory=True,
)
import warnings
warnings.filterwarnings("ignore", message=".*The inputs or output of this tree are not ordered.*")

def opt_reconf(inputs, output, size, **kws):
    tree = optr.search(inputs, output, size)
    tree_r = tree.subtree_reconfigure_forest(
        progbar=True, num_trees=4, num_restarts=20, subtree_weight_what=("size",),
        parallel=False
    )
    return tree_r.get_path()
K = tc.set_backend("jax")
tc.set_contractor("custom", optimizer=optr, preprocessing=True)

import optax

from src.utilities.generate_ansatz import *
from src.utilities.generate_ansatz import _normalize_mps_if_requested
from src.utilities.result_saver import ResultSaver

def _device_memory_bytes() -> Tuple[float, float]:
    """(current, peak) accelerator bytes in use, or (nan, nan) off-accelerator.

    Read from the XLA allocator rather than from nvidia-smi. The two do not
    measure the same thing: nvidia-smi reports what the process has *reserved*
    from the driver, which under the default XLA_PYTHON_CLIENT_PREALLOCATE=true
    is a flat 75% of the card no matter how much is actually live. These figures
    are the allocator's own accounting, so they stay honest either way, and they
    are per-process rather than per-node -- `free -m` in the job script cannot
    tell two arms sharing a node apart.

    `peak_bytes_in_use` is a high-water mark: it only ever rises, so the last
    sample of a run is the peak of the whole run.
    """
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return float("nan"), float("nan")
    if not stats:                       # CPU backend returns None
        return float("nan"), float("nan")
    return (float(stats.get("bytes_in_use", float("nan"))),
            float(stats.get("peak_bytes_in_use", float("nan"))))


def _host_peak_rss_bytes() -> float:
    """Peak resident set size of this process. ru_maxrss is KiB on Linux."""
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024.0


def _nanmax_or_nan(values) -> float:
    """np.nanmax without the all-NaN warning, which is the CPU-backend case."""
    values = np.asarray(values, dtype=float)
    return float(np.nanmax(values)) if np.any(np.isfinite(values)) else float("nan")


def _make_jit_helpers(ansatz: Any) -> Tuple[Callable, Callable]:
    """
    Create JIT helpers for a class implementing `energy_from_params`:

    * costs_vmapped  - vectorised (batched) energy evaluation over params
    * cost_vvag      - vectorised value & gradient for parameter batches
    
    This function captures necessary data from ansatz without passing the entire
    object to JIT, preventing recompilation on every call.
    """
    energy_fn = ansatz.energy_from_params
    perform_noisy_simulations = getattr(ansatz, "perform_noisy_simulations", False)

    if getattr(ansatz, "use_trajectory_resets", False):
        # (params, status, baseline). All three MUST be passed positionally:
        # K.vmap builds its in_axes from the positional arguments only, so a
        # keyword argument would silently not be vectorised over trials.
        traj_fn = ansatz.dice_energy_from_params
        costs_vmapped = K.jit(K.vmap(traj_fn, vectorized_argnums=[0, 1, 2]))
        cost_vvag = K.jit(K.vmap(K.value_and_grad(traj_fn, argnums=0),
                                 vectorized_argnums=[0, 1, 2]))
        return costs_vmapped, cost_vvag

    if perform_noisy_simulations:
        # For noisy simulations, we need to handle seeds
        def energy_with_seed(params, seed):
            return energy_fn(params, seed=seed)
        
        costs_vmapped = K.jit(K.vmap(energy_with_seed, vectorized_argnums=[0, 1]))
        cost_vvag = K.jit(K.vmap(K.value_and_grad(energy_with_seed, argnums=0),
                                 vectorized_argnums=[0, 1]))
    else:
        # For noiseless simulations
        costs_vmapped = K.jit(K.vmap(energy_fn, vectorized_argnums=0))
        cost_vvag = K.jit(K.vmap(K.value_and_grad(energy_fn, argnums=0),
                                 vectorized_argnums=0))
    
    return costs_vmapped, cost_vvag

HamiltonianTerm = Tuple[Tuple[Any, List[int]], float]

@dataclass(eq=False)
class VariationalAnsatz(abc.ABC):
    """
    Abstract base for all variational ansätze in this repo.

    Concrete subclasses need to implement abstract members:

    1. ``_circuit(self, params, *args)`` - builds the parameterised circuit.
    2. ``_hamiltonian_terms(self)`` - returns list of Hamiltonian terms.
    3. ``get_full_hamiltonian(self)`` - returns the sparse Hamiltonian operator.

    Everything else (optimisation loop, logging, initialisation, etc.) is
    provided here and works for *any* ansatz that respects those contracts.
    """

    # ---- generic optimisation hyper‑parameters (common to all ansätze) ----
    trials: int = 10                # number of independent random starts
    maxiter: int = 2001             # optimisation steps
    howoften_tosave: int = 10       # checkpoint frequency
    learning_rate: float = 1e-2     # Adam learning rate
    sparse: bool = True             # work with the sparse Hamiltonian representation
    use_mps: bool = True         # use MPSCircuit with term-by-term expectation values
    normalize_state: bool = True   # normalize after each MPS layer and before its energy
    seed: Optional[int] = None      # RNG seed for parameter init; None draws and logs a fresh one

    # Noise simulation parameters
    perform_noisy_simulations: bool = False
    noise_rate: float = 1e-2
    number_of_shots: int = 1000

    # ---- fields that are filled automatically in __post_init__ ----
    nparams: int = field(init=False)          # derived from concrete class
    initparams: Any = field(init=False)       # shape (trials, nparams)
    fullham: Any = field(init=False, default=None)
    _costs_vmapped: Callable = field(init=False, repr=False)
    _cost_vvag: Callable = field(init=False, repr=False)

    @abc.abstractmethod
    def _circuit(self, params, *args, seed=None):
        """Construct a circuit with the provided params.

        ``seed`` is passed by keyword from energy_from_params, so every
        implementation must accept it under that name.
        """
        ...

    @abc.abstractmethod
    def _hamiltonian_terms(self) -> List[HamiltonianTerm]:
        """Return list of Hamiltonian terms as (ops, coefficient) tuples."""
        ...

    @abc.abstractmethod
    def get_full_hamiltonian(self):
        """Build and return the full sparse Hamiltonian."""
        ...

    def __post_init__(self):
        """Initialize parameters and JIT helpers after dataclass initialization."""
        if self.sparse and not self.use_mps:
            print("Building full Hamiltonian")
            sys.stdout.flush()
            self.fullham = self.get_full_hamiltonian()
            print("Done")
        
        self.initparams = self._initialise_parameters()
        self._costs_vmapped, self._cost_vvag = _make_jit_helpers(self)

    @property
    def reset_mode(self) -> str:
        """Which of the three reset implementations this ansatz uses.

        One canonical label -- "pur", "traj" or "enum" -- so that checkpoints,
        HDF5 records and the log-scraping in collect_runs all agree. Previously
        each consumer re-derived it from the two booleans, and any consumer that
        only knew about `use_trajectory_resets` silently labelled enumerated
        runs as purified.
        """
        if getattr(self, "use_enumerated_resets", False):
            return "enum"
        if getattr(self, "use_trajectory_resets", False):
            return "traj"
        return "pur"

    def _initialise_parameters(self):
        """Initialize random parameters for all trials."""
        if self.seed is None:
            self.seed = int(np.random.randint(1e5))
        print(f"[{type(self).__name__}] parameter init seed: {self.seed} "
              f"(pass seed={self.seed} to reproduce this run)")
        key = jax.random.PRNGKey(self.seed)
        return jax.random.uniform(key, shape=[self.trials, self.nparams],
                                 minval=0, maxval=jnp.pi)

    def _energy_of_circuit(self, qc) -> Any:
        """Term-by-term Hamiltonian expectation on an already-built circuit."""
        energy = 0.0
        for ops, coeff in self._hamiltonian_terms():
            energy += coeff * qc.expectation(*ops)
        return K.real(energy)

    def dice_energy_from_params(self, params, status, baseline) -> Any:
        """Unbiased stochastic energy estimator for trajectory resets.

        Sampling one branch of the reset channel makes the energy an unbiased
        estimate of the exact channel energy, but plain backprop through a
        sampled branch gives *exactly zero* gradient with respect to every reset
        theta: theta only sets the branch probabilities, and the sampled
        branch's normalised state does not contain theta at all. Autodiff
        differentiates the branch that actually ran, finds no theta on that path,
        and returns 0. The upstream (Cartan) gradients are biased for the same
        reason -- they also move the measurement probabilities.

        The missing piece is the score-function identity

            d/dp E[X]  =  E[X * d/dp log P]

        which we obtain without hand-deriving anything by multiplying the energy
        by a "magic box" (the DiCE construction):

            box = exp(log_w - stop_gradient(log_w))

        stop_gradient passes its argument through unchanged but reports
        derivative zero, so ``log_w - stop_gradient(log_w)`` is bitwise 0.0 and
        box is exactly 1.0 -- the reported energy is untouched -- while
        d(box) = d(log_w). Autodiff then yields

            grad = grad(E_traj) + (E_traj - baseline) * grad(log_w)

        i.e. the ordinary pathwise term plus the score term. Because log_w is a
        sum over resets and d log(P1*P2) = d log P1 + d log P2, one box around
        the total covers any number of sequential resets.

        ``baseline`` is a control variate: E[grad log P] = 0, so subtracting a
        trajectory-independent constant leaves the expectation unchanged while
        shrinking the variance. It must be supplied as an argument -- reading it
        off ``self`` inside this traced function would bake the first step's
        value into the compiled program as a constant.
        """
        def one_trajectory(st):
            qc, log_w = self._circuit_traj(params, st)
            _normalize_mps_if_requested(qc, self.normalize_state)
            return self._energy_of_circuit(qc), log_w

        if self.n_trajectories == 1:
            energy, log_w = one_trajectory(status[0])
            box = K.exp(log_w - K.stop_gradient(log_w))
            return box * (energy - baseline) + baseline

        energies, log_ws = jax.vmap(one_trajectory)(status)
        # Leave-one-out baseline: each sample is corrected by the mean of the
        # others, so the baseline stays independent of the sample it corrects
        # and the estimator remains unbiased.
        total = K.stop_gradient(K.sum(energies))
        loo = (total - K.stop_gradient(energies)) / (self.n_trajectories - 1)
        box = K.exp(log_ws - K.stop_gradient(log_ws))
        return K.mean(box * (energies - loo) + loo)

    def enumerated_energy_from_params(self, params) -> Any:
        """Exact, deterministic reset-channel energy by enumerating every branch.

        The reset channel has three Kraus operators per event (see
        generate_ansatz._branch_reset_matrix), so R reset events give 3^R branch
        sequences. Evaluating all of them and recombining reproduces the purified
        (ancilla) energy exactly, since that energy is
        ``Tr[H * sum_j P_j |psi_j><psi_j|]`` and trace is linear -- the ancillas
        in the purified circuit are a which-path record whose only effect is to
        stop the branches interfering, which enumeration reproduces by keeping
        them as separate states.

        The trick that makes this cheap to write is to never normalise. For the
        unnormalised branch state ``|phi_j> = K_{j_R} U_R ... K_{j_1} U_1 |0..0>``,

            P_j        = <phi_j|phi_j>      the probability IS the surviving norm
            P_j * E_j  = <phi_j|H|phi_j>    so the weight cancels the normalisation

        and therefore

            E = sum_j <phi_j|H|phi_j> / sum_j <phi_j|phi_j>

        with no square roots, no Born-probability computation and no division by
        branch weights. The denominator is exactly 1 in exact arithmetic (the
        channel is trace preserving); dividing by it anyway is precisely the
        right correction for MPS truncation drift, which shrinks the norm.
        Zero-probability branches contribute 0 to both sums, so the 0/0 that
        _sampled_reset has to guard with q_safe never arises here.

        Unlike the trajectory estimator this needs no DiCE box and no baseline:
        theta appears inside |phi_j> as the literal cos(theta/2) / sin(theta/2)
        prefactor on each Kraus operator, so ordinary reverse-mode autodiff
        differentiates it. The score term and the pathwise term are the same
        derivative here.
        """
        def one_branch(branch):
            qc = self._circuit_branch(params, branch)
            # MPSCircuit.expectation defaults to normalize=False, so
            # _energy_of_circuit already returns the unnormalised <phi|H|phi>.
            # <phi|phi> read straight off the orthogonality centre. NOT
            # qc.get_norm()**2: get_norm computes sqrt(<phi|phi>), and squaring
            # it back recovers the value but NOT a finite gradient, because
            # d(sqrt(x))/dx = 1/(2 sqrt(x)) is infinite at x = 0 and autodiff
            # differentiates the sqrt node before the square undoes it. Branch
            # probabilities that are exactly zero are routine here -- measuring
            # 1 on a qubit already in |0>, say -- so that path NaNs the whole
            # trial. Summing |T|^2 directly is the same number with a gradient
            # that stays finite (and correctly zero) on a dead branch.
            centre = qc._mps.tensors[qc._mps.center_position]
            den = K.real(K.sum(centre * K.conj(centre)))
            return self._energy_of_circuit(qc), den

        branches = self._reset_branches
        chunk = getattr(self, "branch_chunk_size", None)
        if chunk is None:
            nums, dens = jax.vmap(one_branch)(branches)
        else:
            # Evaluate `chunk` branches at a time under lax.map (a sequential
            # scan) instead of one wide vmap. Measured NOT to help: lax.map under
            # reverse-mode autodiff lowers to a scan that still stores every
            # iteration's residuals, so peak tape is unchanged (33.8 vs 34.5 GB at
            # 3x3/bd64/trials=10) while the serialisation costs 40% more time.
            # Kept for the forward-only case and for experimentation; lower
            # `trials` if you need the memory back.
            nchunks, rem = divmod(len(branches), chunk)
            if rem:
                raise ValueError(
                    f"branch_chunk_size={chunk} does not divide the branch count "
                    f"{len(branches)} (3^{branches.shape[1]}). Pick a divisor."
                )
            chunked = branches.reshape(nchunks, chunk, -1)
            nums, dens = jax.lax.map(jax.vmap(one_branch), chunked)
        return K.sum(nums) / K.sum(dens)

    def energy_from_params(self, params, seed=None) -> Any:
        """Compute energy for given parameters."""
        if getattr(self, "use_enumerated_resets", False):
            # Deterministic exact-channel path: builds and recombines all 3^R
            # Kraus branches itself, so it never goes through _circuit.
            return self.enumerated_energy_from_params(params)

        qc = self._circuit(params, seed=seed)

        if self.normalize_state and not self.use_mps:
            raise ValueError("normalize_state=True is only supported with use_mps=True")
        # Layer builders already normalize at each boundary. Keep this
        # final normalization as a cheap safeguard immediately before the
        # Hamiltonian expectation values.
        _normalize_mps_if_requested(qc, self.normalize_state)


        if self.use_mps:
            return self._energy_of_circuit(qc)
        elif self.sparse:
            if self.perform_noisy_simulations:
                warnings.warn(
                    "Noisy simulations are not supported with sparse=True. "
                    "Set sparse=False to enable noisy simulations.",
                    UserWarning
                )
            return K.real(sparse_expectation(qc, self.fullham))
        else:
            # Non-sparse: compute expectation values term by term
            terms = self._hamiltonian_terms()
            energy = 0.0
            
            if self.perform_noisy_simulations:
                # Setup noise configuration
                noise_conf = noisemodel.NoiseConf()
                noise_conf.add_noise("depolarizing", [self.noise_rate*0.1],
                                    ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"])
                noise_conf.add_noise("depolarizing", [self.noise_rate],
                                    ["cnot", "cz", "swap", "iswap", "rxx", "ryy", "rzz"])
                
                nmc = self.number_of_shots
                if seed is not None:
                    K.set_random_state(seed)
                
                for ops, coeff in terms:
                    status = K.random_uniform(shape=[nmc], boundaries=(0.0, 1.0))
                    exp_val = tc.noisemodel.expectation_noisfy(
                        qc, *ops, noise_conf=noise_conf, nmc=nmc, status=status
                    )
                    energy += coeff * exp_val
            else:
                return self._energy_of_circuit(qc)

            return K.real(energy)

    def purity_from_params(self, params):
        """
        Calculate purity.
        Override this method in subclasses for system-specific purity calculations.
        """
        if getattr(self, "use_trajectory_resets", False):
            raise NotImplementedError(
                "purity_from_params is meaningless with use_trajectory_resets=True: "
                "each trajectory is a pure state, so the ancilla-cut purity would "
                "always read 1.0. The mixed state only exists as the ensemble over "
                "trajectories; estimating Tr[rho^2] would need |<phi_1|phi_2>|^2 "
                "averaged over independently sampled trajectory pairs."
            )
        if getattr(self, "use_enumerated_resets", False):
            raise NotImplementedError(
                "purity_from_params is not implemented for use_enumerated_resets=True: "
                "there is no single state to take an ancilla cut of, since the mixed "
                "state is carried as the ensemble {|phi_j>} over the 3^R Kraus "
                "branches. It is computable -- rho = sum_j |phi_j><phi_j| with the "
                "phi_j unnormalised, so Tr[rho^2] = sum_ij |<phi_i|phi_j>|^2 -- but "
                "that needs 3^(2R) pairwise MPS overlaps (6561 at the 3x3 production "
                "config) and is not wired up."
            )
        # This is a generic implementation that can be overridden
        if not hasattr(self, 'lattice'):
            raise NotImplementedError(
                "purity_from_params requires a 'lattice' attribute. "
                "Override this method in your subclass."
            )
        
        qc = self._circuit(params)

        if self.use_mps:
            s = qc.wavefunction()
        else:
            s = qc.state()

        if hasattr(self, 'ancilla_mps_positions') and self.ancilla_mps_positions:
            cut = self.ancilla_mps_positions
        else:
            n = self.lattice.num_qubits
            if qc._nqubits - n > n:
                cut = list(range(n))
            else:
                cut = list(range(n, qc._nqubits))

        rho = qu.reduced_density_matrix(s, cut=list(cut))
        return K.exp(-qu.renyi_entropy(rho, 2))

    def optimize(self, save_results: bool = False, track_purity: bool = False, track_params: bool = False, track_grads: bool = False, track_bond_dim: bool = False,
                 track_singular_values_per_step: bool = False, singular_value_trial_idx: int = 0):
        """
        Run optimization to find ground state.

        Parameters:
        -----------
        save_results : bool
            If True, automatically save results after optimization completes
        track_purity : bool
            If True, track purity during optimization
        track_bond_dim : bool
            If True, track the maximum bond dimension for each trial during optimization.

        Returns:
        --------
        tuple : (final_energies, final_parameters, all_energies, all_purities)
        """
        params = jnp.array(self.initparams)
        nsnapshots = 1 + (self.maxiter - 1) // self.howoften_tosave
        self.allpurities = np.zeros((self.trials, nsnapshots))
        self.allenergies = np.zeros((self.trials, nsnapshots))
        self.allparams = np.zeros((self.trials, nsnapshots, self.nparams))
        self.allgrads = np.zeros((self.trials, nsnapshots, self.nparams))
        self.all_bond_dims = np.zeros((self.trials, nsnapshots))
        self.singular_values_per_step = [None] * nsnapshots if track_singular_values_per_step else None
        self.reset_thetas_per_step = [None] * nsnapshots if track_singular_values_per_step else None

        if getattr(self, "use_prob_resets_ansatz", False):
            self.all_prob_reset_theta_means = np.full((self.trials, nsnapshots), np.nan) # trials x nsnapshots array initialised with null vector values
        else:
            self.all_prob_reset_theta_means = None

        counter = 0
        optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = optimizer.init(params)

        # Wall-clock instrumentation. step_times[0] is dominated by JIT
        # compilation; every later entry is a warm step. JAX dispatches
        # asynchronously, so each step is blocked on below -- without that these
        # numbers would measure queueing, not computation.
        self.step_times = np.full(self.maxiter, np.nan)
        self.device_str = str(jax.devices())
        print(f"[{type(self).__name__}] jax devices: {self.device_str}")

        # Memory instrumentation, sampled at snapshot cadence. Three arrays
        # because they answer different questions: `live` is the steady-state
        # working set, `peak` is the high-water mark including compilation and
        # backward-pass transients, and `rss` catches host-side growth that the
        # device allocator cannot see.
        self.gpu_live_bytes = np.full(nsnapshots, np.nan)
        self.gpu_peak_bytes = np.full(nsnapshots, np.nan)
        self.host_rss_bytes = np.full(nsnapshots, np.nan)
        _live0, _peak0 = _device_memory_bytes()
        if np.isnan(_peak0):
            print(f"[{type(self).__name__}] device memory_stats unavailable; "
                  f"GPU peak columns will be NaN (host RSS still recorded).")
        else:
            print(f"[{type(self).__name__}] device memory before step 0: "
                  f"live {_live0 / 2**30:.2f} GiB, peak {_peak0 / 2**30:.2f} GiB "
                  f"(XLA_PYTHON_CLIENT_PREALLOCATE="
                  f"{os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'unset')})")

        # Initialize random seeds for noisy simulations
        if self.perform_noisy_simulations:
            base_seed = 42
            all_seeds = jnp.arange(self.maxiter * self.trials).reshape(self.maxiter, self.trials) + base_seed

        use_traj = getattr(self, "use_trajectory_resets", False)
        n_nonfinite_steps = 0
        if use_traj:
            traj_seed = self.traj_seed if self.traj_seed is not None else self.seed
            print(f"[{type(self).__name__}] trajectory seed: {traj_seed} "
                  f"({self.n_trajectories} trajector"
                  f"{'y' if self.n_trajectories == 1 else 'ies'} per trial per step)")
            traj_key = jax.random.PRNGKey(traj_seed)
            status_shape = (self.trials, self.n_trajectories, self.total_resets)
            # Prime the baseline with one forward pass. Starting it at zero would
            # scale the score term by the full energy (~-13 at 3x3) for the first
            # tens of steps, exactly while Adam is calibrating its moments.
            traj_key, subkey = jax.random.split(traj_key)
            baseline = self._costs_vmapped(
                params, jax.random.uniform(subkey, status_shape), jnp.zeros(self.trials)
            )

        with tqdm(range(self.maxiter), miniters=self.howoften_tosave, mininterval=1) as pbar:
            for i in pbar:
                step_start = time.perf_counter()
                if use_traj:
                    traj_key, subkey = jax.random.split(traj_key)
                    status = jax.random.uniform(subkey, status_shape)
                    # Positional: K.vmap vectorises positional arguments only.
                    value, gradient = self._cost_vvag(params, status, baseline)
                elif self.perform_noisy_simulations:
                    seeds = all_seeds[i]
                    value, gradient = self._cost_vvag(params, seeds)
                else:
                    value, gradient = self._cost_vvag(params)

                # A single non-finite gradient permanently poisons that trial's
                # Adam moment buffers, so feed Adam zeros for affected trials
                # and freeze their parameters for this step instead.
                finite_trials = jnp.isfinite(gradient).all(axis=-1) & jnp.isfinite(value)
                safe_gradient = jnp.where(finite_trials[:, None], gradient, 0.0)
                updates, opt_state = optimizer.update(safe_gradient, opt_state)
                updates = jnp.where(finite_trials[:, None], updates, 0.0)
                params = optax.apply_updates(params, updates)

                # Force the asynchronously dispatched work to actually finish
                # before the clock is read, otherwise every step but the last
                # would appear nearly free.
                params = jax.block_until_ready(params)
                self.step_times[i] = time.perf_counter() - step_start

                # Counted for every arm, not just the trajectory one: NaN
                # robustness is a property of the MPS numerics (the forward SVD
                # kernel), not of the reset estimator, so all three arms are
                # exposed to it and all three need the number. Kept live rather
                # than summarised after the loop so a checkpoint from a job that
                # dies at walltime still carries it.
                if not bool(jnp.all(finite_trials)):
                    n_nonfinite_steps += 1
                self.n_nonfinite_steps = n_nonfinite_steps

                if use_traj:
                    # Frozen trials keep their old baseline so a single bad step
                    # cannot poison the control variate.
                    baseline = jnp.where(
                        finite_trials,
                        self.baseline_beta * baseline + (1.0 - self.baseline_beta) * value,
                        baseline,
                    )

                if i % self.howoften_tosave == 0:
                    self.allenergies[:, counter] = value

                    # Sampled before the tracking blocks below, which build extra
                    # un-jitted circuits and would otherwise be charged to the
                    # training step's memory rather than to the diagnostics.
                    #
                    # Sampling first only protects `live`, though. `peak` is a
                    # monotone high-water mark, so once track_bond_dim's extra
                    # circuit builds have raised it, every later reading carries
                    # them. So the two columns mean different things on a run
                    # with tracking on:
                    #   gpu_live_bytes  steady-state working set of the training
                    #                   step alone -- clean, comparable per arm
                    #   gpu_peak_bytes  high-water including the diagnostics
                    # The uncontaminated peak comes from reset_impl_bench, which
                    # runs each arm in its own process and does no tracking.
                    self.gpu_live_bytes[counter], self.gpu_peak_bytes[counter] = _device_memory_bytes()
                    self.host_rss_bytes[counter] = _host_peak_rss_bytes()

                    if track_purity:
                        self.allpurities[:, counter] = purity_vec(self, params)
                    
                    if track_params:
                        self.allparams[:, counter, :] = params
                        
                    if track_grads:
                        self.allgrads[:, counter, :] = gradient
                        grad_norm = np.linalg.norm(np.asarray(gradient), axis=-1)
                        print(f"step {i}: energy: {np.asarray(value)} | gradient norm: {grad_norm}")
                        
                    if track_bond_dim and self.use_mps:
                        for trial in range(self.trials):
                            qc = self._circuit(params[trial])
                            bd = np.asarray(qc.get_bond_dimensions())
                            self.all_bond_dims[trial, counter] = np.max(bd)

                    if track_singular_values_per_step:
                        if not self.use_mps:
                            if counter == 0:
                                print("track_singular_values_per_step requested but use_mps=False; skipping (MPS-only diagnostic).")
                        else:
                            qc = self._circuit(params[singular_value_trial_idx])
                            spectra = get_singular_values_per_cut(qc)
                            self.singular_values_per_step[counter] = spectra

                            spectra_str = " | ".join(
                                f"cut {cut_idx}: {np.array2string(np.asarray(sv), precision=4)}"
                                for cut_idx, sv in enumerate(spectra)
                            )
                            print(f"step {i} (trial {singular_value_trial_idx}): singular values -- {spectra_str}")

                            reset_indices = getattr(self, "reset_param_indices", None)
                            if reset_indices is not None:
                                reset_thetas = np.asarray(params[singular_value_trial_idx])[reset_indices]
                                self.reset_thetas_per_step[counter] = reset_thetas
                                print(f"step {i} (trial {singular_value_trial_idx}): reset thetas = {reset_thetas}")
                            elif counter == 0:
                                print(f"step {i}: no reset_param_indices on this ansatz ({type(self).__name__}); skipping reset-theta print.")

                    counter += 1
                    self._write_checkpoint(i, counter, params)
                    n_bad = int(self.trials - jnp.sum(finite_trials))
                    postfix = f"Current value: {str(jnp.nanmin(jnp.where(finite_trials, value, jnp.nan)))}"
                    if n_bad:
                        postfix += f" ({n_bad} non-finite trial{'s' if n_bad > 1 else ''} frozen)"
                    pbar.set_postfix_str(postfix)

        self.n_nonfinite_steps = n_nonfinite_steps
        if n_nonfinite_steps:
            frac = n_nonfinite_steps / self.maxiter
            msg = (f"{n_nonfinite_steps}/{self.maxiter} steps ({frac:.1%}) had at least "
                   f"one non-finite trial, which optimize() silently freezes.")
            if frac > 0.02:
                warnings.warn(
                    msg + " Above a couple of percent this is a numerical fault rather"
                          " than sampling noise -- check the reset sites and the"
                          " SVD/QR configuration.",
                    RuntimeWarning,
                )
            print(f"[{type(self).__name__}] {msg}")

        # Final memory reading. peak_bytes_in_use is a high-water mark, so this
        # last sample is the peak of the entire run -- including anything the
        # snapshot cadence stepped over.
        final_live, final_peak = _device_memory_bytes()
        final_rss = _host_peak_rss_bytes()
        self.peak_gpu_bytes = _nanmax_or_nan(np.append(self.gpu_peak_bytes, final_peak))
        self.peak_host_rss_bytes = _nanmax_or_nan(np.append(self.host_rss_bytes, final_rss))
        print(f"[{type(self).__name__}] peak memory: "
              f"GPU {self.peak_gpu_bytes / 2**30:.2f} GiB "
              f"(live at exit {final_live / 2**30:.2f} GiB), "
              f"host RSS {self.peak_host_rss_bytes / 2**30:.2f} GiB")

        # Optionally save results
        if save_results:
            self.save_results(value, params, self.allenergies, self.allpurities, self.allparams, self.allgrads, self.all_bond_dims)

        return value, params, self.allenergies, self.allpurities, self.allparams, self.allgrads, self.all_bond_dims, self.singular_values_per_step, self.reset_thetas_per_step

    def _write_checkpoint(self, step, counter, params):
        """Dump partial results at snapshot cadence.

        save_results() only runs after the whole optimisation loop, so a job
        killed at walltime saves nothing at all -- which has already cost several
        multi-hour runs. This writes the same arrays incrementally. The write is
        to a temporary file followed by os.replace, which is atomic on POSIX, so
        a kill mid-write leaves the previous checkpoint intact rather than a
        half-written file.

        Set DPQC_CHECKPOINT to choose the path; it otherwise lands next to the
        run's plots (DPQC_OUTDIR) or in tmp/.
        """
        path = os.environ.get("DPQC_CHECKPOINT")
        if path is None:
            outdir = os.environ.get("DPQC_OUTDIR", "tmp")
            path = os.path.join(outdir, "checkpoint.npz")
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp_path = path + ".partial"
            # Pass an open file object, not a name: given a name, savez appends
            # ".npz" to it, and os.replace would then look for a file that isn't
            # there.
            with open(tmp_path, "wb") as fh:
                np.savez_compressed(
                    fh,
                    step=step,
                    nsnapshots_filled=counter,
                    params=np.asarray(params),
                    all_energies=self.allenergies,
                    all_params=self.allparams,
                    all_bond_dims=self.all_bond_dims,
                    step_times=self.step_times,
                    # Cost and memory accounting.
                    gpu_live_bytes=getattr(self, "gpu_live_bytes", np.array([])),
                    gpu_peak_bytes=getattr(self, "gpu_peak_bytes", np.array([])),
                    host_rss_bytes=getattr(self, "host_rss_bytes", np.array([])),
                    # Run identity, so a checkpoint is self-describing. Without
                    # these the only way to tell which arm and which chain length
                    # produced a checkpoint is to regex the job's stdout, and
                    # stdout is not guaranteed to survive alongside it.
                    reset_mode=self.reset_mode,
                    device=str(getattr(self, "device_str", "unknown")),
                    n_nonfinite_steps=int(getattr(self, "n_nonfinite_steps", 0)),
                    n_mps_qubits=int(getattr(self, "n_mps_qubits", -1)),
                    total_resets=int(getattr(self, "total_resets", -1)),
                    nancillas=int(getattr(self, "nancillas", -1)),
                    bond_dim=int(self.bond_dim) if getattr(self, "bond_dim", None) else -1,
                    trials=int(self.trials),
                    seed=int(self.seed) if getattr(self, "seed", None) is not None else -1,
                    n_trajectories=int(getattr(self, "n_trajectories", 1)),
                    **(getattr(self, "mpo_track", {}) or {}),
                )
            os.replace(tmp_path, path)
        except Exception as exc:  # never let checkpointing kill a run
            if counter == 1:
                print(f"[{type(self).__name__}] checkpoint write failed ({exc}); continuing without checkpoints.")

    def save_results(self, final_energies, final_parameters, all_energies, all_purities, all_params, all_grads, all_bond_dims,
                     save_individual: bool = True):
        """
        Save optimization results to HDF5 files.
        
        This method saves:
        1. A record in the master file: results/<lattice_name>.h5
        2. An individual timestamped file in: tmp/<lattice_name>_TIMESTAMP.h5
        
        Parameters:
        -----------
        final_energies : array
            Final energy values from optimization
        final_parameters : array
            Final parameter values from optimization
        all_energies : array
            Energy trajectory during optimization
        all_purities : array
            Purity trajectory during optimization
        save_individual : bool
            If True, also save individual timestamped file in tmp/
            
        Returns:
        --------
        tuple : (master_file_path, individual_file_path or None)
        """
        
        
        # Initialize result saver
        saver = ResultSaver(self.lattice.name())
        
        # Prepare results dictionary
        results = {
            'final_energies': np.array(final_energies),
            'final_parameters': np.array(final_parameters),
            'all_energies': np.array(all_energies),
            'all_purities': np.array(all_purities),
            'all_params' : np.array(all_params),
            'all_grads' : np.array(all_grads),
            'all_bond_dims' : np.array(all_bond_dims),
            'min_energy': float(np.min(final_energies)),
            'mean_energy': float(np.mean(final_energies)),
            'std_energy': float(np.std(final_energies)),
            # Cost accounting. step_times[0] carries JIT compilation, so the
            # warm cost is the median over the rest.
            'step_times': np.asarray(getattr(self, 'step_times', [])),
            'compile_seconds': float(getattr(self, 'step_times', [np.nan])[0]),
            'warm_step_seconds': float(np.nanmedian(getattr(self, 'step_times', [np.nan])[1:])
                                       if len(getattr(self, 'step_times', [])) > 1 else np.nan),
            'total_optimize_seconds': float(np.nansum(getattr(self, 'step_times', []))),
            'ordering_search_seconds': float(getattr(self, 'ordering_search_seconds', np.nan)),
            'device': str(getattr(self, 'device_str', 'unknown')),
            'n_nonfinite_steps': int(getattr(self, 'n_nonfinite_steps', 0)),
            # Memory accounting, from the XLA allocator rather than nvidia-smi
            # so it is per-process and preallocation-proof. See
            # _device_memory_bytes for why that distinction matters.
            'gpu_live_bytes': np.asarray(getattr(self, 'gpu_live_bytes', [])),
            'gpu_peak_bytes': np.asarray(getattr(self, 'gpu_peak_bytes', [])),
            'host_rss_bytes': np.asarray(getattr(self, 'host_rss_bytes', [])),
            'peak_gpu_bytes': float(getattr(self, 'peak_gpu_bytes', np.nan)),
            'peak_host_rss_bytes': float(getattr(self, 'peak_host_rss_bytes', np.nan)),
            # Run identity. asdict() in the result saver only captures declared
            # dataclass fields, so everything computed in __post_init__ would
            # otherwise be absent and the record could not say what chain length
            # produced it.
            'reset_mode': self.reset_mode,
            'n_mps_qubits': int(getattr(self, 'n_mps_qubits', -1)),
            'total_resets': int(getattr(self, 'total_resets', -1)),
            'nancillas': int(getattr(self, 'nancillas', -1)),
        }

        # Save using the result saver
        master_file, individual_file = saver.save_run(
            settings=self,
            results=results,
            save_individual=save_individual
        )
        
        print(f"Results saved to master file: {master_file}")
        if individual_file:
            print(f"Individual run saved to: {individual_file}")
        
        return master_file, individual_file

def get_initial_costs(self, params=None):
    """Get initial cost values for all trials."""
    if params is None:
        params = self.initparams
    return self._costs_vmapped(params)

def get_initial_costs_and_gradients(self, params=None):
    """Get initial costs and gradients for all trials."""
    if params is None:
        params = self.initparams
    return self._cost_vvag(params)

# JIT-compiled purity functions
def _purity_wrapper(ansatz, params):
    """Wrapper for JIT compilation of purity calculation."""
    return ansatz.purity_from_params(params)

purity = K.jit(_purity_wrapper, static_argnums=[0])
purity_vec = K.jit(K.vmap(_purity_wrapper, vectorized_argnums=[1]), static_argnums=[0])

# ToricCodeAnsatz has been moved to src/utilities/ansatz_classes.py
# Import it from there if needed
