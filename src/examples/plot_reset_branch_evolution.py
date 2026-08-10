# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Track reset-branch probability and (truncated) bond dimension throughout
training, at the same cadence as the rest of this codebase's per-step
tracking (howoften_tosave).

Builds on src/diagnostics/reset_branches.py's single-snapshot branch
enumeration (used elsewhere for the final trained circuit only) by
re-running it at every saved checkpoint for one tracked trial
(branch_trial_idx), piggybacking entirely on ToricCodeAnsatz.optimize's
existing track_params=True history -- no changes to the training loop
itself, this is a purely post-hoc analysis.

Produces:
  1. The regular training curve (energy density) for the tracked trial.
  2. Max bond dimension per reset branch, over training.
  3. Probability per reset branch, over training.
  4. Per-branch, per-cut bond dimension breakdown (one plot per branch).
  5. A single figure of subplots (one per branch) showing that branch's raw
     singular values (all cuts overlaid) over training.

Run with: python -m src.examples.plot_reset_branch_evolution
"""

import itertools
import os
import warnings

import numpy as np

# use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.reset_branches import enumerate_branches


def track_branch_evolution(ansatz, all_param, branch_trial_idx, steps,
                            max_ancillas, prob_floor, truncation_error):
    """
    For the single tracked trial, rebuild the circuit from its saved
    parameters at every checkpoint and enumerate its reset-ancilla branches.

    Returns (all_branches, p_x_history, bond_dims_history, singular_values_history):
      all_branches: list of all 2**nancillas possible ancilla outcomes, in a
        fixed order used to index the arrays/lists below.
      p_x_history: array (n_branches, n_snapshots). 0.0 for any branch/step
        where that branch fell below prob_floor -- a real, meaningful value
        (this branch had ~0 probability there), not a placeholder.
      bond_dims_history: array (n_branches, n_snapshots, n_cuts), NaN for any
        branch/step below prob_floor -- bond dimension is *undefined*, not
        zero, when a branch isn't meaningfully present. matplotlib skips NaN
        when drawing lines, so this shows up as a genuine gap rather than a
        misleading dip to 0.
      singular_values_history: nested list [branch_idx][step_idx] -> the raw
        per-cut spectrum (list of arrays) from enumerate_branches, or None if
        that branch was below prob_floor at that step. Kept as a plain nested
        list rather than a fixed array since truncation makes each cut's
        array length vary over time.
    """
    ancilla_indices = ansatz.ancilla_mps_positions
    n_system_qubits = ansatz.lattice.num_qubits
    nancillas = len(ancilla_indices)
    n_cuts = n_system_qubits - 1
    n_snapshots = len(steps)

    all_branches = list(itertools.product((0, 1), repeat=nancillas))
    n_branches = len(all_branches)
    branch_index = {x: i for i, x in enumerate(all_branches)}

    p_x_history = np.zeros((n_branches, n_snapshots))
    bond_dims_history = np.full((n_branches, n_snapshots, n_cuts), np.nan)
    singular_values_history = [[None] * n_snapshots for _ in range(n_branches)]

    for step_idx in range(n_snapshots):
        params_at_step = all_param[branch_trial_idx, step_idx]
        qc = ansatz._circuit(params_at_step)
        branch_results, total_p = enumerate_branches(
            qc, ancilla_indices, n_system_qubits,
            max_ancillas=max_ancillas, prob_floor=prob_floor,
            truncation_error=truncation_error,
        )

        for x, info in branch_results.items():
            i = branch_index[x]
            p_x_history[i, step_idx] = info["p_x"]
            bond_dims_history[i, step_idx, :] = info["bond_dims"]
            singular_values_history[i][step_idx] = info["singular_values"]

        print(f"step {steps[step_idx]}: {len(branch_results)}/{n_branches} branches "
              f"above prob_floor (sum p_x = {total_p:.6f})")

    return all_branches, p_x_history, bond_dims_history, singular_values_history


def get_reference_energy_density(Lx, Ly):
    """Hardcoded reference ground-state energy density for supported lattice
    sizes, else None (same convention as plot_bond_dim_by_reset_config.py)."""
    if Lx == 2 and Ly == 2:
        return -5 / 4
    if Lx == 3:
        if Ly == 2:
            return -8 / 7
        if Ly == 3:
            return -13 / 12
    return None


def plot_training_curve(steps, energy_trial, n_qubits, Lx, Ly, outdir, title_suffix):
    """The regular energy-density training curve for the single tracked
    trial (branch_trial_idx) -- for context alongside the branch plots,
    since they're analyzing that exact same trial/run."""
    energy_density = np.asarray(energy_trial, dtype=float) / n_qubits

    plt.figure(figsize=(6, 4))
    plt.plot(steps, energy_density, marker="o", markersize=2)

    ref = get_reference_energy_density(Lx, Ly)
    if ref is not None:
        plt.axhline(y=ref, color="tab:orange", linestyle="--", label=f"reference ({ref:.4f})")
        plt.legend()

    plt.xlabel("Training steps")
    plt.ylabel("E / n")
    plt.title(f"Training curve\n{title_suffix}")
    plt.tight_layout()
    plt.grid(visible=True, which="both", linestyle="--")

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "training_curve.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved training curve plot to: {fname}")


def plot_max_bond_dim_per_branch(steps, all_branches, bond_dims_history, outdir, title_suffix):
    """Plot 1: each branch's max-over-cuts bond dimension, over training."""
    n_branches = len(all_branches)

    with warnings.catch_warnings():
        # np.nanmax on an all-NaN slice (a branch absent at that step) is
        # expected and produces the NaN we want -- suppress the routine
        # RuntimeWarning that comes with it.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_bond_history = np.nanmax(bond_dims_history, axis=2)  # (n_branches, n_snapshots)

    plt.figure(figsize=(6, 4))
    for i, x in enumerate(all_branches):
        label = "".join(map(str, x)) if n_branches <= 12 else None
        plt.plot(steps, max_bond_history[i], marker="o", markersize=2, label=label)

    plt.xlabel("Training steps")
    plt.ylabel("Max bond dimension (truncated)")
    plt.title(f"Reset-branch max bond dimension over training\n{title_suffix}")
    if n_branches <= 12:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.grid(visible=True, which="both", linestyle="--")

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "branch_max_bond_dim_over_training.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved max-bond-dim-per-branch plot to: {fname}")


def plot_probability_per_branch(steps, all_branches, p_x_history, outdir, title_suffix):
    """Plot 2: each branch's probability, over training."""
    n_branches = len(all_branches)

    plt.figure(figsize=(6, 4))
    for i, x in enumerate(all_branches):
        label = "".join(map(str, x)) if n_branches <= 12 else None
        plt.plot(steps, p_x_history[i], marker="o", markersize=2, label=label)

    plt.xlabel("Training steps")
    plt.ylabel("Branch probability $p_x$")
    plt.title(f"Reset-branch probability over training\n{title_suffix}")
    if n_branches <= 12:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.grid(visible=True, which="both", linestyle="--")

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "branch_probability_over_training.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved probability-per-branch plot to: {fname}")


def plot_per_branch_bond_dims(steps, all_branches, bond_dims_history, outdir, title_suffix):
    """Plot 3: one plot per branch, showing all of that branch's per-cut
    bond dims over training, legend = cut index. Skips branches that never
    rose above prob_floor at any tracked step (all-NaN throughout)."""
    n_cuts = bond_dims_history.shape[2]
    os.makedirs(outdir, exist_ok=True)
    n_saved = 0

    for i, x in enumerate(all_branches):
        branch_bond_dims = bond_dims_history[i]  # (n_snapshots, n_cuts)
        if np.all(np.isnan(branch_bond_dims)):
            continue

        label = "".join(map(str, x))
        plt.figure(figsize=(6, 4))
        for cut_idx in range(n_cuts):
            plt.plot(steps, branch_bond_dims[:, cut_idx], marker="o", markersize=2,
                      label=f"cut {cut_idx}")

        plt.xlabel("Training steps")
        plt.ylabel("Bond dimension (truncated)")
        plt.title(f"Branch {label} bond dims over training\n{title_suffix}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.grid(visible=True, which="both", linestyle="--")

        fname = os.path.join(outdir, f"branch_{label}_bond_dims_over_training.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        n_saved += 1

    print(f"Saved {n_saved}/{len(all_branches)} per-branch bond-dim plots to: {outdir}")


def _extract_sv_line(branch_step_spectra, n_snapshots, cut_idx, sv_idx):
    """Build one (cut, singular-value-index) line across all snapshots for a
    single branch. NaN wherever that step is absent (below prob_floor) or
    truncation left fewer than sv_idx+1 singular values at that cut/step --
    both cases mean "this particular singular value doesn't exist here",
    which should be a gap, not a 0."""
    line = np.full(n_snapshots, np.nan)
    for step_idx in range(n_snapshots):
        spectra = branch_step_spectra[step_idx]
        if spectra is None:
            continue
        sv_array = np.asarray(spectra[cut_idx])
        if sv_idx < len(sv_array):
            line[step_idx] = sv_array[sv_idx]
    return line


def plot_branch_singular_values_subplots(steps, all_branches, singular_values_history, outdir, title_suffix):
    """Plot 5: one figure, one subplot per branch (skipping branches that
    never rose above prob_floor), each showing every singular value at every
    cut for that branch, overlaid, over training. Singular values are always
    sorted descending per cut/step (tensorcircuit's own SVD convention), so
    "sv index 0" consistently tracks the largest singular value at that cut,
    "sv index 1" the second largest, etc., even as truncation changes how
    many survive at a given step."""
    n_snapshots = len(steps)
    active = [(i, x) for i, x in enumerate(all_branches)
              if any(spectra is not None for spectra in singular_values_history[i])]

    if not active:
        print("No branches ever appeared above prob_floor; skipping singular-value subplot figure.")
        return

    n_active = len(active)
    ncols = min(4, n_active)
    nrows = int(np.ceil(n_active / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for plot_idx, (branch_idx, x) in enumerate(active):
        ax = axes[plot_idx // ncols][plot_idx % ncols]
        branch_step_spectra = singular_values_history[branch_idx]

        n_cuts = None
        max_len_per_cut = []
        for spectra in branch_step_spectra:
            if spectra is None:
                continue
            if n_cuts is None:
                n_cuts = len(spectra)
                max_len_per_cut = [0] * n_cuts
            for cut_idx, sv in enumerate(spectra):
                max_len_per_cut[cut_idx] = max(max_len_per_cut[cut_idx], len(np.asarray(sv)))

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for cut_idx in range(n_cuts):
            color = colors[cut_idx % len(colors)]
            for sv_idx in range(max_len_per_cut[cut_idx]):
                line = _extract_sv_line(branch_step_spectra, n_snapshots, cut_idx, sv_idx)
                # one legend entry per cut (color), not per singular value --
                # only label the first line drawn for each cut.
                label = f"cut {cut_idx}" if sv_idx == 0 else None
                ax.plot(steps, line, marker="o", markersize=1.5, linewidth=1,
                          color=color, alpha=0.5, label=label)

        ax.set_title(f"Branch {''.join(map(str, x))}", fontsize=9)
        ax.set_xlabel("Training steps", fontsize=8)
        ax.set_ylabel("Singular value", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(visible=True, which="both", linestyle="--", alpha=0.5)
        if n_cuts <= 12:
            ax.legend(fontsize=5)

    for hide_idx in range(n_active, nrows * ncols):
        axes[hide_idx // ncols][hide_idx % ncols].axis("off")

    fig.suptitle(f"Reset-branch singular values over training\n{title_suffix}", fontsize=11)
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, "branch_singular_values_subplots.png")
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"Saved per-branch singular-value subplots to: {fname}")


# ---------------------------------------------------------------------------------------------------------------------
# Global simulation parameters
# ---------------------------------------------------------------------------------------------------------------------
Lx = 3
Ly = 2
nlayers = 2
howoften_tosave = 20
trials = 1
maxiter = 600
learning_rate = 1e-2
h = 0.96

use_prob_resets_ansatz = True
reset_layers = [1]
use_optimal_ordering = False

# Explicit int -- keeps the known-good SVD split path for the live training
# circuit (see generate_ansatz.py's make_split_conf docstring for why
# bond_dim=None's QR path is avoided anywhere gradients could flow). This
# script's own post-hoc analysis calls (rebuilding qc from saved params,
# and the branch reconstruction inside enumerate_branches) never have
# gradients flowing through them, so bond_dim=None would actually be safe
# there too -- kept explicit here just for consistency with
# plot_bond_dim_by_reset_config.py's convention.
bond_dim = 32

# Which trial's per-step saved parameters to track (a single trial, matching
# the existing track_singular_values_per_step/singular_value_trial_idx
# convention used elsewhere for per-step diagnostics -- averaging over many
# trials here would multiply the cost of every checkpoint's branch
# enumeration by the number of trials).
branch_trial_idx = 0

# Reset-branch enumeration settings (see src/diagnostics/reset_branches.py).
max_ancillas = 6          # full enumeration is O(2**nancillas); raise at your own risk
prob_floor = 1e-9         # branches below this probability are excluded entirely
truncation_error = 1e-4   # real MPS truncation applied ONLY to each branch's post-hoc
                          # reconstruction (never to the live training circuit) -- strips
                          # numerically-negligible trailing singular values so bond-dim
                          # counts reflect real entanglement, not floating-point noise.

outdir = os.environ.get("DPQC_OUTDIR", "outputs")
# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    print("===== Python run parameters =====")
    print(f"Lx={Lx}, Ly={Ly}, nlayers={nlayers}")
    print(f"trials={trials}, maxiter={maxiter}, howoften_tosave={howoften_tosave}")
    print(f"reset_layers={reset_layers}, bond_dim={bond_dim}")
    print(f"branch_trial_idx={branch_trial_idx}, max_ancillas={max_ancillas}")
    print(f"prob_floor={prob_floor}, truncation_error={truncation_error}")
    print(f"outdir={outdir}")
    print("==================================")

    ansatz = ToricCodeAnsatz(
        Lx=Lx, Ly=Ly, nlayers=nlayers, h=h, trials=trials, maxiter=maxiter,
        howoften_tosave=howoften_tosave, learning_rate=learning_rate,
        use_prob_resets_ansatz=use_prob_resets_ansatz, reset_layers=reset_layers,
        bond_dim=bond_dim, use_optimal_ordering=use_optimal_ordering,
    )

    # Check this BEFORE running (possibly expensive) training, not after --
    # full branch enumeration is O(2**nancillas) per checkpoint.
    if ansatz.nancillas > max_ancillas:
        raise SystemExit(
            f"This ansatz has {ansatz.nancillas} reset ancillas, which exceeds "
            f"max_ancillas={max_ancillas}. Full branch enumeration is "
            f"O(2**{ansatz.nancillas}) per checkpoint -- reduce reset_layers / "
            f"lattice size, or raise max_ancillas at your own risk, before "
            f"running training. Stopping now rather than after training completes."
        )

    print(f"Training with {ansatz.nancillas} reset ancillas "
          f"({2 ** ansatz.nancillas} possible branches per checkpoint)...")

    (final_E, final_params, all_E, all_P, all_param, all_grads,
     all_bond_dims, sv_per_step, reset_thetas_per_step) = ansatz.optimize(track_params=True)

    n_snapshots = all_param.shape[1]
    steps = np.arange(n_snapshots) * howoften_tosave

    all_branches, p_x_history, bond_dims_history, singular_values_history = track_branch_evolution(
        ansatz, all_param, branch_trial_idx, steps,
        max_ancillas, prob_floor, truncation_error,
    )

    title_suffix = (f"{Lx}x{Ly}, nlayers={nlayers}, reset_layers={reset_layers}, "
                     f"trial={branch_trial_idx}")

    plot_training_curve(steps, all_E[branch_trial_idx], ansatz.lattice.num_qubits,
                         Lx, Ly, outdir, title_suffix)
    plot_max_bond_dim_per_branch(steps, all_branches, bond_dims_history, outdir, title_suffix)
    plot_probability_per_branch(steps, all_branches, p_x_history, outdir, title_suffix)
    plot_per_branch_bond_dims(steps, all_branches, bond_dims_history, outdir, title_suffix)
    plot_branch_singular_values_subplots(steps, all_branches, singular_values_history, outdir, title_suffix)

    print("Done.")
