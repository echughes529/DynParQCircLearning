# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Find ground state for a given Hamiltonian using the dynamic parameterized quantum circuit ansatz."""
import abc
import sys
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

    def energy_from_params(self, params, seed=None) -> Any:
        """Compute energy for given parameters."""
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

                if use_traj:
                    # Frozen trials keep their old baseline so a single bad step
                    # cannot poison the control variate.
                    baseline = jnp.where(
                        finite_trials,
                        self.baseline_beta * baseline + (1.0 - self.baseline_beta) * value,
                        baseline,
                    )
                    if not bool(jnp.all(finite_trials)):
                        n_nonfinite_steps += 1

                if i % self.howoften_tosave == 0:
                    self.allenergies[:, counter] = value

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
                    n_bad = int(self.trials - jnp.sum(finite_trials))
                    postfix = f"Current value: {str(jnp.nanmin(jnp.where(finite_trials, value, jnp.nan)))}"
                    if n_bad:
                        postfix += f" ({n_bad} non-finite trial{'s' if n_bad > 1 else ''} frozen)"
                    pbar.set_postfix_str(postfix)

        if use_traj:
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

        # Optionally save results
        if save_results:
            self.save_results(value, params, self.allenergies, self.allpurities, self.allparams, self.allgrads, self.all_bond_dims)

        return value, params, self.allenergies, self.allpurities, self.allparams, self.allgrads, self.all_bond_dims, self.singular_values_per_step, self.reset_thetas_per_step

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
            'std_energy': float(np.std(final_energies))
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
