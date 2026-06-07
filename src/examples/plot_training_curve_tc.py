import csv
import numpy as np
import os

# use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utilities.generate_toric_code_hamiltonian import ToricCode
from src.utilities.ansatz_classes import ToricCodeAnsatz


def _extract_prob_reset_theta_history_from_ansatz(ansatz):
    """
    Try to extract a per-trial, per-snapshot history of mean probabilistic-reset
    theta values from the ansatz object.

    Expected shape:
        (trials, n_snapshots)

    This helper is deliberately defensive because different local branches may
    store the history under different attribute names.
    """
    candidate_names = [
        "all_prob_reset_theta_means",
        "prob_reset_theta_means_all",
        "prob_reset_theta_history",
        "all_prob_reset_theta_history",
        "all_theta_means",
        "theta_means_history",
    ]

    for name in candidate_names:
        if hasattr(ansatz, name):
            value = getattr(ansatz, name)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 2:
                return arr

    return None


def save_prob_reset_theta_mean_csv(results, csv_path):
    """
    Save a CSV containing the per-trial probabilistic-reset theta value at each
    saved training step, together with the across-trial mean and standard
    deviation at that same step.

    One row is written for each `(h, trial, training_step)` triple.

    Notes
    -----
    This expects each `results[h]` entry to contain a key
    `"all_prob_reset_theta_means"` with shape `(trials, n_snapshots)`.
    If that data is not available from the ansatz branch currently in use,
    the function writes nothing and prints a warning.
    """
    rows = []

    for h in h_list:
        result_for_h = results[h]
        theta_history = result_for_h.get("all_prob_reset_theta_means")

        if theta_history is None:
            continue

        theta_history = np.asarray(theta_history, dtype=float)
        if theta_history.ndim != 2:
            continue

        mean_theta = theta_history.mean(axis=0)
        std_theta = theta_history.std(axis=0)

        n_trials, n_available_snapshots = theta_history.shape
        steps_for_h = steps[:n_available_snapshots]

        for trial_idx in range(n_trials):
            for snapshot_idx, step in enumerate(steps_for_h):
                rows.append(
                    {
                        "h": h,
                        "trial": int(trial_idx + 1),
                        "training_step": int(step),
                        "theta_value": float(theta_history[trial_idx, snapshot_idx]),
                        "mean_theta_across_trials": float(mean_theta[snapshot_idx]),
                        "std_theta_across_trials": float(std_theta[snapshot_idx]),
                    }
                )

    if not rows:
        print(
            "Warning: no probabilistic-reset theta history was found on the ansatz object, "
            "so no theta CSV was written. The plotting script is ready, but the optimisation "
            "code must expose a per-trial theta history first."
        )
        return None

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "h",
                "trial",
                "training_step",
                "theta_value",
                "mean_theta_across_trials",
                "std_theta_across_trials",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved probabilistic-reset theta CSV to: {csv_path}")
    return csv_path


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
                    "energy_density_minus_reference": float(
                        (all_E[trial_idx, snapshot_idx] / n_qubits) - E_dens_ref_dict[h]
                    ),
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
                "energy_density_minus_reference",
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
                "final_energy_density_minus_reference": float(
                    (final_E[trial_idx] / n_qubits) - E_dens_ref_dict[h]
                ),
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
                "final_energy_density_minus_reference",
                "final_purity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved final energies CSV to: {csv_path}")
    return csv_path


def running_for_hs(Lx=2, Ly=2, nlayers_current=2, howoften_toreset=7, trials=10, maxiter=201, howoften_tosave=10,
                   unitary=True, sparse=True, perform_noisy_simulations=False, number_of_shots=1000, use_prob_resets=False,
                   reset_layers=None,
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
            use_prob_resets=use_prob_resets,
            reset_layers=reset_layers,
        )

        final_E, final_purity, all_E, all_P, all_param, all_grads, all_bond_dims = ansatz.optimize(
            track_params=track_params,
            track_grads=track_grads,
            track_bond_dim=track_bond_dim,
        )
        reset_layers_used = None
        if use_prob_resets and hasattr(ansatz, "active_reset_layers"):
            reset_layers_used = list(ansatz.active_reset_layers)

        results[h] = {
            "final_E": final_E,
            "final_purity": final_purity,
            "all_E": all_E,
            "all_P": all_P,
            "all_param": all_param,
            "reset_layers_used": reset_layers_used,
            "reset_layers_input": reset_layers,
        }

    return results

def plotting(results):
        # make plots for given h values
        plt.figure(figsize=(5, 4))
        for h, c in zip(h_list, colours):
            all_E = results[h]["all_E"]      # all_E shape: (trials, n_snapshots)

            mean_E = all_E.mean(axis=0)
            std_E  = all_E.std(axis=0)

            E_dens_ref = E_dens_ref_dict[h]
            mean_E_dens_diff = (mean_E / n_qubits) - E_dens_ref
            std_E_dens_diff  = std_E / n_qubits

            label = rf"$h = {h}$"

            plt.plot(steps, mean_E_dens_diff, color=c, label=label)
            plt.fill_between(
                steps,
                mean_E_dens_diff - std_E_dens_diff,
                mean_E_dens_diff + std_E_dens_diff,
                color=c,
                alpha=0.3,
            )
        
        plt.xlabel("Training steps")
        plt.ylabel("E/n")
        plt.title(f"{Lx}x{Ly}, nlayers:{nlayers}, trials: {trials}, resets: {use_prob_resets}")
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
        fname = os.path.join(outdir, f"{Lx}x{Ly}_nlayers_{nlayers}_resets_{use_prob_resets}.png") 
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

    if reset_layers_used is None:
        active_reset_layers = list(range(nlayers))
    else:
        active_reset_layers = list(reset_layers_used)

    if reset_layers_input is None:
        reset_layers_label = f"all layers {active_reset_layers}"
    else:
        reset_layers_label = str(active_reset_layers)

    n_reset_thetas = n_resets_per_layer * len(active_reset_layers)

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
        plt.title(f"Reset |theta| {theta_idx + 1} across trials, reset_layers={reset_layers_label}")
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')


        fname = os.path.join(outdir, f"reset_abs_theta_{theta_idx + 1}_all_trials.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot to: {fname}")


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

        if Lx == 2 and Ly == 2:
            plt.axhline(y=-5/4, linestyle="--")  # for 2x2
        if Lx == 3:
            if Ly == 2:
                plt.axhline(y=-8/7, linestyle="--")  # for 3x2
            if Ly == 3:
                plt.axhline(y=-13/12, linestyle="--")  # for 3x3

        plt.xlabel("Trial number")
        plt.ylabel("Final E/n")
        plt.title(f"Final energies, h={h}, {Lx}x{Ly}, nlayers:{nlayers}")
        plt.tight_layout()
        plt.grid(visible=True, which="both", linestyle="--")

        fname = os.path.join(outdir, f"final_energies_h_{h}_by_trial.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved final energies plot to: {fname}")



def plotting_bond_dims(results):
    """
    Plot the tracked maximum bond dimension for all trials over training.

    This saves one plot per h value. Each plot contains all trials, plus the
    mean across trials.
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
        plt.title(f"Bond dimension, h={h}, {Lx}x{Ly}, nlayers:{nlayers}")
        if n_trials <= 10:
            plt.legend()
        else:
            plt.legend(["mean across trials"])
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')

        fname = os.path.join(outdir, f"bond_dim_h_{h}_all_trials.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved bond-dimension plot to: {fname}")




# ---------------------------------------------------------------------------------------------------------------------
# Global simulation parameters 
# ---------------------------------------------------------------------------------------------------------------------
Lx = 3
Ly = 3
nlayers = 2
howoften_tosave = 10
trials = 100
maxiter = 1501
howoften_toreset = 7
unitary = True
sparse = True
perform_noisy_simulations = False
noise_rate = 5e-2
number_of_shots = 500 
use_prob_resets = True

# Choose which ansatz layers get probabilistic resets.
# Use None to apply resets on every layer, preserving the old behaviour.
# Layer indexing is zero-based, so [0] means only the first layer.
reset_layers = None

track_grads = True
track_params = True
track_bond_dim = False
# ---------------------------------------------------------------------------------------------------------------------


tc_ = ToricCode(Lx, Ly)
n_qubits = tc_.num_qubits   
n_two_q_params_ron = 3 * (Lx-1) * (Ly-1) * 9 * nlayers

# comments are my very rough estimates from graph in paper
E_dens_ref_dict = {
    0.0: 0,         # -1.05
    0.12: 0,        # -1.00
    0.96: 0,        # -1.00
}

# Allow job scripts (e.g. Eddie/Grid Engine) to set a per-run output directory.
# Falls back to a local "outputs" folder when DPQC_OUTDIR is not set.
outdir = os.environ.get("DPQC_OUTDIR", "outputs")
h_list    = [0.0] 
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
                             use_prob_resets=use_prob_resets,
                             reset_layers=reset_layers,
                             )
    plotting(results)
    plotting_thetas(results)
    plotting_final_energies(results)

    final_energies_csv_path = os.path.join(outdir, "final_energies_by_trial.csv")
    save_final_energies_csv(results, final_energies_csv_path)
    
    training_history_csv_path = os.path.join(outdir, "training_history.csv")
    save_training_history_csv(results, training_history_csv_path)
    
