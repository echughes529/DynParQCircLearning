"""
Run parameters for src/examples/plot_training_curve_tc.py.

WHY THIS FILE EXISTS
--------------------
`sbatch` does not snapshot your Python source. It queues a job that will later run
`python -m src.examples.plot_training_curve_tc`, and Python opens the .py files *when the
job starts*, not when you submit. So editing parameters between submissions is unsafe:
on 2026-08-18 five jobs submitted 06:00-06:05 all sat PENDING and started together at
06:53:26, so all five read the file's final state (Lx=4, Ly=4) instead of the five
different states that had been saved.

Every parameter below can therefore be overridden by an environment variable named
DPQC_<UPPERCASE NAME>. Slurm captures --export values at *submit* time, so a queued job
carries its own parameters no matter what this file says later. Use ./submit.sh, which
wires that up for you.

STILL NOT COVERED
-----------------
Only the parameters in this file travel with the job. Edits to other source files --
notably src/utilities/ansatz_classes.py, generate_ansatz.py, and the rest of src/ -- are
still read from disk at job start. If you change those while jobs are queued, the queued
jobs will pick up the new code. Let queued jobs drain before editing them, or snapshot the
tree at submit time (rsync into a per-run dir and run from there).

USAGE
-----
    ./submit.sh                                 # job name: 3x3_bd64_r1
    ./submit.sh LX=4 LY=4                       # job name: 4x4_bd64_r1
    ./submit.sh LX=4 LY=4 BOND_DIM=32           # job name: 4x4_bd32_r1
    ./submit.sh LX=4 LY=4 TAG=traj_pls_wrk      # job name: 4x4_bd64_r1_traj_pls_wrk
    ./submit.sh NAME=whatever_i_like            # job name: whatever_i_like

The default job name is {Lx}x{Ly}_bd{bond_dim}_r{reset_layers}. TAG= appends your own
descriptive text after that; NAME= replaces the whole name. reset_layers is sanitised for
the name: [1] -> r1, [0,1] -> r0-1, None -> rall.

Sweeping lattice sizes, five jobs from a file you never touch:

    for cfg in "2 2" "3 2" "3 3" "4 3" "4 4"; do
      set -- $cfg
      ./submit.sh LX=$1 LY=$2 TAG=traj_pls_wrk
    done

Sweeping bond dimension:

    for bd in 16 32 64; do ./submit.sh LX=3 LY=3 BOND_DIM=$bd TAG=bdsweep; done

Any knob below works the same way, e.g. MAXITER=2000, TRIALS=20, RESET_LAYERS="[0,1]",
TRACK_GRADS=true, H_LIST="[0,0.5]". Anything you do not pass keeps its default below, so
a bare `python -m src.examples.plot_training_curve_tc` behaves exactly as it always has.
"""

import ast
import os


def _env(name, default):
    """Return DPQC_<NAME> from the environment, coerced to the type of `default`."""
    raw = os.environ.get("DPQC_" + name.upper())
    if raw is None:
        return default
    if isinstance(default, bool):
        # This must come before the int check: in Python bool is a subclass of int, so
        # isinstance(True, int) is True and a bool would otherwise be sent to int(),
        # making DPQC_UNITARY=false raise instead of parsing.
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, (list, type(None))):
        # ast.literal_eval parses a Python literal out of a string safely -- it accepts
        # only lists/numbers/strings/None, never executable code the way eval() would.
        # Lets DPQC_RESET_LAYERS="[0,1]" or "None" arrive as real Python objects.
        return ast.literal_eval(raw)
    return raw


# ---------------------------------------------------------------------------------------
# Global simulation parameters
# Each is overridable as DPQC_<UPPERCASE NAME>; see the module docstring above.
# ---------------------------------------------------------------------------------------
Lx = _env("Lx", 3)
Ly = _env("Ly", 3)
h_list = _env("h_list", [0])

nlayers = _env("nlayers", 2)
howoften_tosave = _env("howoften_tosave", 10)
trials = _env("trials", 10)
maxiter = _env("maxiter", 1500)
howoften_toreset = _env("howoften_toreset", 7)
unitary = _env("unitary", True)
sparse = _env("sparse", False)
perform_noisy_simulations = _env("perform_noisy_simulations", False)
noise_rate = _env("noise_rate", 5e-2)
number_of_shots = _env("number_of_shots", 500)
use_prob_resets_ansatz = _env("use_prob_resets_ansatz", True)
use_mps = _env("use_mps", True)
bond_dim = _env("bond_dim", 64)
use_optimal_ordering = _env("use_optimal_ordering", True)

# Reset implementation. True samples ONE branch of the reset channel per step
# (ancilla-free, chain stays at nq sites, stochastic but unbiased energy and
# gradient); False keeps the channel coherent with two ancillas per reset event
# (chain is nq + 2*total_resets sites, deterministic energy). The two share a
# parameter layout, so the same seed gives both arms identical initial
# parameters -- which is what makes an A/B comparison meaningful.
use_trajectory_resets = _env("use_trajectory_resets", True)
n_trajectories = _env("n_trajectories", 1)
# Third option, mutually exclusive with use_trajectory_resets: evaluate ALL
# 3^total_resets Kraus branches of the reset channel and recombine them. Same
# ancilla-free nq-site chain as the trajectory path, but the energy and gradient
# are exact and deterministic -- it reproduces the purified energy to machine
# precision, with no DiCE score term and hence no gradient variance. Costs 3^R
# circuit evaluations per energy, so it is a small-R tool: 81 branches at 3x3
# with reset_layers=[1], but 6561 with two reset layers (see max_reset_branches).
use_enumerated_resets = _env("use_enumerated_resets", False)
# Must divide 3^R. MEASURED not to help: at 3x3/bd64/trials=10 on an A40 it saved
# nothing (33.8 vs 34.5 GB) and cost 40% more time, because reverse-mode autodiff
# turns lax.map into a scan that still stores every iteration's residuals. Lower
# `trials` instead -- memory is near-linear in it (5.6 GB at trials=2) and this
# estimator is deterministic, so it needs far fewer restarts than a noisy one.
branch_chunk_size = _env("branch_chunk_size", None)
# Refuse-to-run guard on 3^R. The default of 512 admits 3x3 with reset_layers=[1]
# (R=4, 81 branches) but not 4x3 (R=6, 729), which is deliberate -- the cost is
# exponential and should not be paid by accident. Raise it explicitly when the
# cost is the thing being measured.
max_reset_branches = _env("max_reset_branches", 512)
# Parameter-init seed. None draws a fresh one and prints it. Pin it to compare
# two arms from bit-identical initial parameters.
seed = _env("seed", None)
# Trajectory randomness. None reuses `seed`, so a run is reproducible from the
# single seed printed at startup; set it explicitly to hold the parameter init
# fixed while varying only the sampling.
traj_seed = _env("traj_seed", None)

# Choose which ansatz layers get probabilistic resets.
# Use None to apply resets on every layer, preserving the old behaviour.
# Layer indexing is zero-based, so [0] means only the first layer.
reset_layers = _env("reset_layers", [1])

track_grads = _env("track_grads", False)
track_params = _env("track_params", False)
track_bond_dim = _env("track_bond_dim", False)
track_singular_values = _env("track_singular_values", False)
# Preferred: average singular values over all trials converged within 0.1% of
# the reference energy density. This is only the fallback trial count, used
# when no reference is defined for (Lx, Ly) or zero trials converge.
singular_value_ntrials = _env("singular_value_ntrials", 5)

# Track the per-cut singular-value spectrum and print reset-theta values for a
# single trial at every save step during training. Only correct for reset_layers
# equal to the last layer (see ToricCodeAnsatz.reset_param_slice); set trials=1
# when using this for a clean, cheap diagnostic run.
track_singular_values_per_step = _env("track_singular_values_per_step", False)
singular_value_trial_idx = _env("singular_value_trial_idx", 0)
# used by plot_singular_values_per_step to count "active" singular values per cut
singular_value_threshold = _env("singular_value_threshold", 1e-4)

plot_final_energies = _env("plot_final_energies", True)
save_final_energies = _env("save_final_energies", False)
save_training_history = _env("save_training_history", False)
save_singular_values_per_step = _env("save_singular_values_per_step", False)
