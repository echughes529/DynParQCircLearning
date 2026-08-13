import csv
import numpy as np
import os

# use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utilities.generate_toric_code_hamiltonian import ToricCode
from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.generate_ansatz import get_singular_values_per_cut


# ------------------------------------------------------------------------------------------------------------
# Save raw per-trial training histories to CSV
# ------------------------------------------------------------------------------------------------------------
def save_training_history_csv(results, csv_path):
    """
    Save raw per-trial training histories to CSV so plots can be regenerated
    later without rerunning the optimisation.

    One row is written for each `(h, trial, training_step)` triple.
    """
    rows = []

    for h in h_list:
        result_for_h = results[h]

        all_E = np.asarray(result_for_h["all_E"], dtype=float)
        all_P = result_for_h.get("all_P")
        all_P = None if all_P is None else np.asarray(all_P, dtype=float)

        n_trials, n_available_snapshots = all_E.shape
        steps_for_h = steps[:n_available_snapshots]

        for trial_idx in range(n_trials):
            for snapshot_idx, step in enumerate(steps_for_h):
                row = {
                    "h": h,
                    "trial": int(trial_idx + 1),
                    "training_step": int(step),
                    "energy": float(all_E[trial_idx, snapshot_idx]),
                    "energy_density": float(all_E[trial_idx, snapshot_idx] / n_qubits),
                }

                if all_P is not None and all_P.ndim == 2:
                    row["purity"] = float(all_P[trial_idx, snapshot_idx])
                else:
                    row["purity"] = ""

                rows.append(row)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "h",
                "trial",
                "training_step",
                "energy",
                "energy_density",
                "purity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved training history CSV to: {csv_path}")
    return csv_path


# ------------------------------------------------------------------------------------------------------------
# Save final energies per trial to CSV
# ------------------------------------------------------------------------------------------------------------
def save_final_energies_csv(results, csv_path):
    """
    Save one row per trial containing the final energy and final energy density.
    """
    rows = []

    for h in h_list:
        result_for_h = results[h]

        final_E = np.asarray(result_for_h["final_E"], dtype=float)
        final_purity = result_for_h.get("final_purity")
        final_purity = None if final_purity is None else np.asarray(final_purity, dtype=float)

        n_trials = final_E.shape[0]

        for trial_idx in range(n_trials):
            row = {
                "h": h,
                "trial": int(trial_idx + 1),
                "final_energy": float(final_E[trial_idx]),
                "final_energy_density": float(final_E[trial_idx] / n_qubits),
            }

            if final_purity is not None and final_purity.ndim == 1:
                row["final_purity"] = float(final_purity[trial_idx])
            else:
                row["final_purity"] = ""

            rows.append(row)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "h",
                "trial",
                "final_energy",
                "final_energy_density",
                "final_purity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved final energies CSV to: {csv_path}")
    return csv_path


# ------------------------------------------------------------------------------------------------------------
# Save per-step singular values and reset-theta values for the tracked trial to CSV
# ------------------------------------------------------------------------------------------------------------
def save_singular_values_csv(results, csv_path):
    """
    Save per-step singular-value spectra and reset-theta values for the single
    tracked trial (see `singular_value_trial_idx`) to CSV.

    One row is written for each `(h, training_step, bond_cut)` triple. The
    reset thetas at that step (same for every cut, since they're a single
    snapshot of the trial's parameters) are repeated on every row for that step.
    """
    rows = []

    for h in h_list:
        result_for_h = results[h]
        sv_per_step = result_for_h.get("singular_values_per_step")
        reset_thetas_per_step = result_for_h.get("reset_thetas_per_step")
        trial_idx = result_for_h.get("singular_value_trial_idx_used")

        if sv_per_step is None:
            print(f"No per-step singular-value data found for h={h}; skipping CSV rows.")
            continue

        for snapshot_idx, spectra in enumerate(sv_per_step):
            if spectra is None:
                continue

            step = steps[snapshot_idx]

            reset_thetas = None
            if reset_thetas_per_step is not None:
                reset_thetas = reset_thetas_per_step[snapshot_idx]
            reset_thetas_str = (
                ";".join(f"{v:.6g}" for v in reset_thetas) if reset_thetas is not None else ""
            )

            for cut_idx, sv in enumerate(spectra):
                rows.append({
                    "h": h,
                    "trial": int(trial_idx) if trial_idx is not None else "",
                    "training_step": int(step),
                    "reset_thetas": reset_thetas_str,
                    "bond_cut": cut_idx,
                    "singular_values": ";".join(f"{v:.6g}" for v in sv),
                })

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["h", "trial", "training_step", "reset_thetas", "bond_cut", "singular_values"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-step singular-value CSV to: {csv_path}")
    return csv_path


def running_for_hs(Lx=2, Ly=2, nlayers_current=2, howoften_toreset=7, trials=10, maxiter=201, howoften_tosave=10,
                   unitary=True, sparse=True, perform_noisy_simulations=False, number_of_shots=1000, use_prob_resets_ansatz=False,
                   reset_layers=None, use_mps=True, bond_dim=None, use_optimal_ordering=True,
                   ):

    if nlayers_current is None:
        nlayers_current = nlayers

    # run all h for this parameter value
    results = {}
    for h in h_list:
        ansatz = ToricCodeAnsatz(
            Lx=Lx,
            Ly=Ly,
            nlayers=nlayers_current,
            howoften_toreset=howoften_toreset,
            h=h,
            trials=trials,
            maxiter=maxiter,
            howoften_tosave=howoften_tosave,
            unitary=unitary,
            sparse=sparse,
            perform_noisy_simulations=perform_noisy_simulations,
            noise_rate=noise_rate,
            number_of_shots=number_of_shots,
            use_prob_resets_ansatz=use_prob_resets_ansatz,
            reset_layers=reset_layers,
            use_mps=use_mps,
            bond_dim=bond_dim,
            use_optimal_ordering=use_optimal_ordering,
        )

        final_E, final_purity, all_E, all_P, all_param, all_grads, all_bond_dims, sv_per_step, reset_thetas_per_step = ansatz.optimize(
            track_params=track_params,
            track_grads=track_grads,
            track_bond_dim=track_bond_dim,
            track_singular_values_per_step=track_singular_values_per_step,
            singular_value_trial_idx=singular_value_trial_idx,
        )
        reset_layers_used = None
        if use_prob_resets_ansatz and hasattr(ansatz, "active_reset_layers"):
            reset_layers_used = list(ansatz.active_reset_layers)

        singular_values_mean_per_cut = None
        singular_values_std_per_cut = None
        n_sv_trials_used = None
        if track_singular_values and use_mps:
            final_E_density = np.asarray(final_E, dtype=float) / n_qubits
            reference_energy_density = get_reference_energy_density(Lx, Ly)

            converged_trial_indices = []
            if reference_energy_density is not None:
                relative_error = np.abs(
                    (final_E_density - reference_energy_density) / reference_energy_density
                )
                converged_trial_indices = np.where(relative_error <= 0.001)[0].tolist()

            if converged_trial_indices:
                selected_trial_indices = converged_trial_indices
            else:
                if reference_energy_density is None:
                    print(f"No reference energy density defined for Lx={Lx}, Ly={Ly}; "
                          f"falling back to first {singular_value_ntrials} trials for singular-value averaging.")
                else:
                    print(f"h={h}: 0/{trials} trials converged within 0.1% of reference energy density "
                          f"({reference_energy_density}); falling back to first {singular_value_ntrials} trials for singular-value averaging.")
                selected_trial_indices = list(range(min(singular_value_ntrials, trials)))

            n_sv_trials_used = len(selected_trial_indices)
            per_trial_spectra = [
                get_singular_values_per_cut(ansatz._circuit(final_purity[trial_idx]))
                for trial_idx in selected_trial_indices
            ]  # list[trial] of list[cut] of array(bond_dim_at_cut,)

            n_cuts = len(per_trial_spectra[0])
            singular_values_mean_per_cut = [
                np.mean([per_trial_spectra[t][cut] for t in range(n_sv_trials_used)], axis=0)
                for cut in range(n_cuts)
            ]
            singular_values_std_per_cut = [
                np.std([per_trial_spectra[t][cut] for t in range(n_sv_trials_used)], axis=0)
                for cut in range(n_cuts)
            ]

        results[h] = {
            "final_E": final_E,
            "final_purity": final_purity,
            "all_E": all_E,
            "all_P": all_P,
            "all_param": all_param,
            "all_grads": all_grads,
            "all_bond_dims": all_bond_dims,
            "reset_layers_used": reset_layers_used,
            "reset_layers_input": reset_layers,
            "use_prob_resets_ansatz": use_prob_resets_ansatz,
            "singular_values_mean_per_cut": singular_values_mean_per_cut,
            "singular_values_std_per_cut": singular_values_std_per_cut,
            "singular_value_ntrials_used": n_sv_trials_used,
            "singular_values_per_step": sv_per_step,
            "reset_thetas_per_step": reset_thetas_per_step,
            "singular_value_trial_idx_used": singular_value_trial_idx if track_singular_values_per_step else None,
        }

    return results

def plotting(results):
        # make plots for given h values
        plt.figure(figsize=(5, 4))
        for h, c in zip(h_list, colours):
            all_E = results[h]["all_E"]      # all_E shape: (trials, n_snapshots)

            mean_E = all_E.mean(axis=0)
            std_E  = all_E.std(axis=0)

            n_available_snapshots = all_E.shape[1]
            steps_for_h = steps[:n_available_snapshots]

            mean_E_dens = mean_E / n_qubits
            std_E_dens  = std_E / n_qubits

            label = rf"$h = {h}$"

            plt.plot(steps_for_h, mean_E_dens, color=c, label=label)
            plt.fill_between(
                steps_for_h,
                mean_E_dens - std_E_dens,
                mean_E_dens + std_E_dens,
                color=c,
                alpha=0.3,
            )
        
        plt.xlabel("Training steps")
        plt.ylabel("E/n")
        plt.title(f"{Lx}x{Ly}, nlayers:{nlayers}, trials: {trials}, resets: {use_prob_resets_ansatz}")
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')
        os.makedirs(outdir, exist_ok=True)
        if Lx==2 and Ly==2:
            plt.axhline(y=-5/4, color = "tab:orange", linestyle = '--') # for 2x2
        if Lx==3:
            if Ly==2:
                plt.axhline(y=-8/7, color = "tab:orange", linestyle = '--') # for 3x2
            if Ly == 3:
                plt.axhline(y=-13/12, color = "tab:orange", linestyle = '--') # for 3x3
        
        # ------------------------------------------------------------------------------------------------------------
        fname = os.path.join(outdir, f"{Lx}x{Ly}_nlayers_{nlayers}_resets_{use_prob_resets_ansatz}.png") 
        # ------------------------------------------------------------------------------------------------------------
        
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot to: {fname}")

def plotting_params(results, trial=0):
    """
    Only works for single value of h

    Args:
        results (_type_): _description_
    """
    # make plots for given h values
    plt.figure(figsize=(5, 4))
    h = h_list[0]
    all_param = np.asarray(results[h]["all_param"], dtype=float)
    print(f"all_param shape for h={h}: {all_param.shape}")
    print(f"all_param min/max for h={h}: {np.nanmin(all_param)}, {np.nanmax(all_param)}")

    if all_param.ndim != 3:
        raise ValueError(
            f"Expected all_param to have shape (trials, snapshots, nparams), "
            f"but got shape {all_param.shape}"
        )

    all_param_trial = all_param[trial]

    n_available_snapshots = all_param_trial.shape[0]
    steps_for_h = steps[:n_available_snapshots]

    for param_idx in range(all_param_trial.shape[1]):
        plt.plot(steps_for_h, all_param_trial[:, param_idx], label=f"param: {param_idx + 1}")

    
    plt.xlabel("Training steps")
    plt.ylabel("param")
    plt.title(f"params")
    # There can be many parameters, so the legend can make this plot unreadable.
    # plt.legend()
    plt.tight_layout()
    plt.grid(visible=True, which='both', linestyle='--')
    os.makedirs(outdir, exist_ok=True)
    
    # ------------------------------------------------------------------------------------------------------------
    fname = os.path.join(outdir, f"params.png") 
    # ------------------------------------------------------------------------------------------------------------
    
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved plot to: {fname}")



def plotting_thetas(results):
    """
    Plot the magnitude of the probabilistic-reset theta parameters for all trials.

    Only works for a single value of h, using h_list[0].
    Assumes the parameter vector is ordered as:
        [two-qubit gate params][probabilistic-reset theta params][final single-qubit params]

    This saves one plot per reset-theta parameter. Each plot contains all trials.
    
    Currently theta tracking is done by saving all params and then just extracting the theta params.
    This should be done more efficiently when scaling up (just saving theta, not all params).
    """
    h = h_list[0]
    all_param = np.asarray(results[h]["all_param"], dtype=float)

    if all_param.ndim != 3:
        raise ValueError(
            f"Expected all_param to have shape (trials, snapshots, nparams), "
            f"but got shape {all_param.shape}"
        )

    n_trials, n_available_snapshots, _ = all_param.shape
    steps_for_h = steps[:n_available_snapshots]

    n_resets_per_layer = (Lx - 1) * (Ly - 1) # number of plaquettes

    reset_layers_used = results[h].get("reset_layers_used")
    reset_layers_input = results[h].get("reset_layers_input")

    if not results[h].get("use_prob_resets_ansatz", False):
        active_reset_layers = []
    elif reset_layers_used is None:
        active_reset_layers = list(range(nlayers))
    else:
        active_reset_layers = list(reset_layers_used)

    n_reset_thetas = n_resets_per_layer * len(active_reset_layers)

    if n_reset_thetas == 0:
        print("No reset-theta parameters for this run (use_prob_resets_ansatz=False or "
              "reset_layers=[]); skipping theta plot.")
        return

    if reset_layers_input is None:
        reset_layers_label = f"all layers {active_reset_layers}"
    else:
        reset_layers_label = str(active_reset_layers)

    print(f"reset_layers input: {reset_layers_input}")
    print(f"reset_layers used: {active_reset_layers}")
    print(f"n_reset_thetas: {n_reset_thetas}")

    theta_start = n_two_q_params_ron
    theta_stop = theta_start + n_reset_thetas

    theta_history = np.abs(all_param[:, :, theta_start:theta_stop])
    # shape: (trials, snapshots, n_reset_thetas)

    print(f"theta parameter indices: {theta_start} to {theta_stop - 1}")
    print(f"abs(theta_history) shape: {theta_history.shape}")
    print(f"abs(theta_history) min/max: {np.nanmin(theta_history)}, {np.nanmax(theta_history)}")

    os.makedirs(outdir, exist_ok=True)

    for theta_idx in range(n_reset_thetas):
        plt.figure(figsize=(5, 4))

        for trial_idx in range(n_trials):
            plt.plot(
                steps_for_h,
                theta_history[trial_idx, :, theta_idx],
                alpha=0.7,
                linewidth=1,
            )

        mean_theta = theta_history[:, :, theta_idx].mean(axis=0)
        plt.plot(
            steps_for_h,
            mean_theta,
            color="black",
            linewidth=2,
            label="mean across trials",
        )

        plt.xlabel("Training steps")
        plt.ylabel(r"Reset $|\theta|$")
        plt.title(f"Reset |theta| {theta_idx + 1}, reset_layers={reset_layers_label}")
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')


        fname = os.path.join(outdir, f"reset_abs_theta_{theta_idx + 1}_all_trials.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot to: {fname}")


# ------------------------------------------------------------------------------------------------------------
# Reference ground-state energy density per lattice size
# ------------------------------------------------------------------------------------------------------------
def get_reference_energy_density(Lx, Ly):
    """Hardcoded reference ground-state energy density for supported lattice sizes, else None."""
    if Lx == 2 and Ly == 2:
        return -5/4  # for 2x2
    if Lx == 3:
        if Ly == 2:
            return -8/7  # for 3x2
        if Ly == 3:
            return -13/12  # for 3x3
    return None


# ------------------------------------------------------------------------------------------------------------
# Plotting final energies per trial
# ------------------------------------------------------------------------------------------------------------
def plotting_final_energies(results):
    """
    Plot the final energy density for each trial, with a reference ground-state
    energy density line for the current lattice size.
    """
    os.makedirs(outdir, exist_ok=True)

    for h in h_list:
        final_E = np.asarray(results[h]["final_E"], dtype=float)
        trial_numbers = np.arange(1, final_E.shape[0] + 1)
        final_E_density = final_E / n_qubits

        plt.figure(figsize=(5, 4))
        plt.plot(trial_numbers, final_E_density, marker="o", linestyle="None")

        reference_energy_density = get_reference_energy_density(Lx, Ly)

        if reference_energy_density is not None:
            plt.axhline(y=reference_energy_density, linestyle="--")

            relative_error = np.abs(
                (final_E_density - reference_energy_density)
                / reference_energy_density
            )
            n_within = int(np.sum(relative_error <= 0.001))  # 0.1%
            percentage_within = 100 * n_within / len(final_E_density)

            print(
                f"h={h}: {n_within}/{len(final_E_density)} trials "
                f"({percentage_within:.1f}%) within 0.1% of reference energy density "
                f"({reference_energy_density})"
            )
        else:
            print(
                f"No reference energy density defined for Lx={Lx}, Ly={Ly}; "
                "skipping 0.1% trial count."
            )

        plt.xlabel("Trial number")
        plt.ylabel("Final E/n")
        if reference_energy_density is not None:
            plt.title(
                f"Final energies, h={h}, {Lx}x{Ly}, nlayers:{nlayers}\n"
                f"{n_within}/{len(final_E_density)} within 0.1%"
            )
        else:
            plt.title(f"Final energies, h={h}, {Lx}x{Ly}, nlayers:{nlayers}")
        plt.tight_layout()
        plt.grid(visible=True, which="both", linestyle="--")

        fname = os.path.join(outdir, f"final_energies_h_{h}_by_trial.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved final energies plot to: {fname}")



def plotting_bond_dims(results):
    """
    Plot the max bond dimension over training.

    all_bond_dims has shape (trials, snapshots).
    Shows all trials plus the mean across trials.
    """
    os.makedirs(outdir, exist_ok=True)

    for h in h_list:
        all_bond_dims = results[h].get("all_bond_dims")

        if all_bond_dims is None:
            print(f"No bond-dimension history found for h={h}; skipping bond-dim plot.")
            continue

        all_bond_dims = np.asarray(all_bond_dims, dtype=float)

        if all_bond_dims.ndim != 2:
            raise ValueError(
                f"Expected all_bond_dims to have shape (trials, snapshots), "
                f"but got shape {all_bond_dims.shape} for h={h}"
            )

        n_trials, n_available_snapshots = all_bond_dims.shape
        steps_for_h = steps[:n_available_snapshots]

        print(f"all_bond_dims shape for h={h}: {all_bond_dims.shape}")
        print(f"all_bond_dims min/max for h={h}: {np.nanmin(all_bond_dims)}, {np.nanmax(all_bond_dims)}")

        plt.figure(figsize=(5, 4))

        for trial_idx in range(n_trials):
            plt.plot(
                steps_for_h,
                all_bond_dims[trial_idx],
                alpha=0.35,
                linewidth=1,
                label=f"trial {trial_idx + 1}" if n_trials <= 10 else None,
            )

        mean_bond_dim = np.nanmean(all_bond_dims, axis=0)
        plt.plot(
            steps_for_h,
            mean_bond_dim,
            color="black",
            linewidth=2,
            label="mean across trials",
        )

        plt.xlabel("Training steps")
        plt.ylabel("Max bond dimension")
        plt.title(f"Max bond dim, h={h}, {Lx}x{Ly}, nlayers:{nlayers}")
        if n_trials <= 10:
            plt.legend()
        else:
            plt.legend(["mean across trials"])
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')

        fname = os.path.join(outdir, f"bond_dim_max_h_{h}_all_trials.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved max bond-dimension plot to: {fname}")


def print_singular_value_table(results):
    """
    Print, per h, a table of the singular-value spectrum at every bond of the
    final trained circuit: cut index, bond dimension, and singular values as
    mean +/- std across the trials used.
    """
    for h in h_list:
        sv_mean = results[h].get("singular_values_mean_per_cut")
        sv_std = results[h].get("singular_values_std_per_cut")
        n_trials_used = results[h].get("singular_value_ntrials_used")
        if sv_mean is None:
            print(f"No singular-value data found for h={h}; skipping.")
            continue

        print(f"\nSingular value spectrum per bond (h={h}, mean +/- std over {n_trials_used} trials):")
        header = f"{'cut':>5} | {'bond dim':>8} | singular values (mean +/- std)"
        print(header)
        print("-" * len(header))
        for cut_idx, (means, stds) in enumerate(zip(sv_mean, sv_std)):
            sv_str = ", ".join(f"{m:.4g}+/-{s:.1g}" for m, s in zip(means, stds))
            print(f"{cut_idx:>5} | {len(means):>8} | {sv_str}")


def plot_singular_values_per_step(results, threshold=1e-5):
    """
    Plot, for each bond cut, the number of singular values above `threshold`
    for a single tracked trial as it evolves over training. This is a proxy
    for the "effective" bond dimension the circuit is actually using at each
    cut, as opposed to the truncation bond_dim ceiling.
    """
    os.makedirs(outdir, exist_ok=True)

    for h in h_list:
        sv_per_step = results[h].get("singular_values_per_step")
        trial_idx = results[h].get("singular_value_trial_idx_used")

        if sv_per_step is None:
            print(f"No per-step singular-value data found for h={h}; skipping.")
            continue

        populated = [(s, spectra) for s, spectra in enumerate(sv_per_step) if spectra is not None]
        if not populated:
            print(f"Per-step singular-value tracking was on for h={h}, but no snapshots were populated "
                  f"(likely use_mps=False); skipping.")
            continue

        n_cuts = len(populated[0][1])
        steps_used = [steps[s] for s, _ in populated]

        plt.figure(figsize=(6, 4))
        for cut_idx in range(n_cuts):
            n_above_per_step = [int(np.sum(spectra[cut_idx] > threshold)) for _, spectra in populated]
            plt.plot(
                steps_used,
                n_above_per_step,
                marker="o",
                markersize=2,
                label=f"cut {cut_idx}" if n_cuts <= 12 else None,
            )

        plt.xlabel("Training steps")
        plt.ylabel(f"# singular values > {threshold:g}")
        plt.title(f"Per-cut singular values above threshold vs. step, h={h}, trial {trial_idx}")
        if n_cuts <= 12:
            plt.legend(fontsize=7)
        plt.tight_layout()
        plt.grid(visible=True, which="both", linestyle="--")

        fname = os.path.join(outdir, f"singular_values_per_step_h_{h}_trial_{trial_idx}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved per-step singular-value plot to: {fname}")


# ------------------------------------------------------------------------------------------------------------
# Plot gradient norms over training
# ------------------------------------------------------------------------------------------------------------
def plotting_gradient_norms(results):
    """
    Plot the L2 norm of the gradient vector over training.

    This uses the stored full gradient history `all_grads`, which should have
    shape (trials, snapshots, nparams). One plot is saved per h value.
    """
    os.makedirs(outdir, exist_ok=True)

    for h in h_list:
        all_grads = results[h].get("all_grads")

        if all_grads is None:
            print(f"No gradient history found for h={h}; skipping gradient-norm plot.")
            continue

        all_grads = np.asarray(all_grads, dtype=float)

        if all_grads.ndim != 3:
            raise ValueError(
                f"Expected all_grads to have shape (trials, snapshots, nparams), "
                f"but got shape {all_grads.shape} for h={h}"
            )

        gradient_norms = np.linalg.norm(all_grads, axis=2)
        # shape: (trials, snapshots)

        n_trials, n_available_snapshots = gradient_norms.shape
        steps_for_h = steps[:n_available_snapshots]

        print(f"all_grads shape for h={h}: {all_grads.shape}")
        print(f"gradient_norms shape for h={h}: {gradient_norms.shape}")
        print(f"gradient_norms min/max for h={h}: {np.nanmin(gradient_norms)}, {np.nanmax(gradient_norms)}")

        plt.figure(figsize=(5, 4))

        for trial_idx in range(n_trials):
            plt.plot(
                steps_for_h,
                gradient_norms[trial_idx],
                alpha=0.35,
                linewidth=1,
                label=f"trial {trial_idx + 1}" if n_trials <= 10 else None,
            )

        mean_grad_norm = np.nanmean(gradient_norms, axis=0)
        plt.plot(
            steps_for_h,
            mean_grad_norm,
            color="black",
            linewidth=2,
            label="mean across trials",
        )

        plt.yscale("log")
        plt.xlabel("Training steps")
        plt.ylabel("Gradient norm")
        plt.title(f"Gradient norm, h={h}, {Lx}x{Ly}, nlayers:{nlayers}")
        if n_trials <= 10:
            plt.legend()
        else:
            plt.legend(["mean across trials"])
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')

        fname = os.path.join(outdir, f"gradient_norm_h_{h}_all_trials.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved gradient-norm plot to: {fname}")


# ---------------------------------------------------------------------------------------------------------------------
# Global simulation parameters 
# ---------------------------------------------------------------------------------------------------------------------
Lx = 3
Ly = 3
h_list = [0] 

nlayers = 2
howoften_tosave = 1
trials = 10
maxiter = 1500
howoften_toreset = 7
unitary = True
sparse = False
perform_noisy_simulations = False
noise_rate = 5e-2
number_of_shots = 500 
use_prob_resets_ansatz = True
use_mps = True
bond_dim = 64
use_optimal_ordering = False

# Choose which ansatz layers get probabilistic resets.
# Use None to apply resets on every layer, preserving the old behaviour.
# Layer indexing is zero-based, so [0] means only the first layer.
reset_layers = []

track_grads = True
track_params = False
track_bond_dim = False
track_singular_values = False
# Preferred: average singular values over all trials converged within 0.1% of
# the reference energy density. This is only the fallback trial count, used
# when no reference is defined for (Lx, Ly) or zero trials converge.
singular_value_ntrials = 5

# Track the per-cut singular-value spectrum and print reset-theta values for a
# single trial at every save step during training. Only correct for reset_layers
# equal to the last layer (see ToricCodeAnsatz.reset_param_slice); set trials=1
# when using this for a clean, cheap diagnostic run.
track_singular_values_per_step = False
singular_value_trial_idx = 0
singular_value_threshold = 1e-4  # used by plot_singular_values_per_step to count "active" singular values per cut

plot_final_energies = True
save_final_energies = False
save_training_history = False
save_singular_values_per_step = False
# ---------------------------------------------------------------------------------------------------------------------


tc_ = ToricCode(Lx, Ly)
n_qubits = tc_.num_qubits   
n_two_q_params_ron = 3 * (Lx-1) * (Ly-1) * 9 * nlayers

# Allow job scripts (e.g. Eddie/Grid Engine) to set a per-run output directory.
# Falls back to a local "outputs" folder when DPQC_OUTDIR is not set.
outdir = os.environ.get("DPQC_OUTDIR", "outputs")

colours   = ["tab:orange", "tab:blue", "turquoise"]
n_snapshots = 1 + maxiter // howoften_tosave
steps = np.arange(n_snapshots) * howoften_tosave

if __name__ == "__main__":
    # ------- Printing Parameters ---------
    print("===== Python run parameters =====")
    print(f"Lx={Lx}, Ly={Ly}")
    print(f"nlayers={nlayers}")
    print(f"howoften_toreset={howoften_toreset}")
    print(f"trials={trials}, maxiter={maxiter}")
    print(f"outdir={outdir}")
    print(f"use_mps={use_mps}, bond_dim={bond_dim}")
    print(f"reset_layers={reset_layers}")
    print("=================================")

    results = running_for_hs(Lx=Lx, 
                             Ly=Ly, 
                             nlayers_current=nlayers, 
                             howoften_toreset=howoften_toreset, 
                             trials=trials, 
                             maxiter=maxiter,
                             howoften_tosave=howoften_tosave, 
                             unitary=unitary, 
                             sparse=sparse, 
                             perform_noisy_simulations=perform_noisy_simulations,
                             number_of_shots=number_of_shots,
                             use_prob_resets_ansatz=use_prob_resets_ansatz,
                             reset_layers=reset_layers,
                             use_mps=use_mps,
                             bond_dim=bond_dim,
                             use_optimal_ordering=use_optimal_ordering,
                             )
    # ------------------------------------------------------------------------------------------
    # Choosing which results to obtain
    # ------------------------------------------------------------------------------------------
    plotting(results)

    if track_grads:
        plotting_gradient_norms(results)
    if track_params:
        plotting_thetas(results)

    if plot_final_energies:
        plotting_final_energies(results)

    if save_final_energies:
        final_energies_csv_path = os.path.join(outdir, "final_energies_by_trial.csv")
        save_final_energies_csv(results, final_energies_csv_path)

    if save_training_history:
        training_history_csv_path = os.path.join(outdir, "training_history.csv")
        save_training_history_csv(results, training_history_csv_path)

    if track_bond_dim:
        plotting_bond_dims(results)

    if track_singular_values:
        print_singular_value_table(results)

    if track_singular_values_per_step:
        plot_singular_values_per_step(results, threshold=singular_value_threshold)

    if save_singular_values_per_step:
        singular_values_csv_path = os.path.join(outdir, "singular_values_per_step.csv")
        save_singular_values_csv(results, singular_values_csv_path)
