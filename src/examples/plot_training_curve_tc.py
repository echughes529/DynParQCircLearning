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


def running_for_hs(Lx=2, Ly=2, nlayers_current=2, howoften_toreset=7, trials=10, maxiter=201, howoften_tosave=10,
                   unitary=True, sparse=True, perform_noisy_simulations=False, number_of_shots=1000, use_prob_resets=False,
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
        )

        final_E, final_purity, all_E, all_P, all_gradient_norms = ansatz.optimize()
        all_prob_reset_theta_means = _extract_prob_reset_theta_history_from_ansatz(ansatz)

        results[h] = {
            "all_E": all_E,
            "all_P": all_P,
            "all_gradient_norms": all_gradient_norms,
            "all_prob_reset_theta_means": all_prob_reset_theta_means,
        }

    return results

def plotting(results):
        # make plots for given h values
        plt.figure(figsize=(5, 4))

        if plot_each_trial:
            for h, c in zip(h_list, colours):
                all_E = np.asarray(results[h]["all_E"], dtype=float)  # shape: (trials, n_snapshots)
                E_dens_ref = E_dens_ref_dict[h]
                n_trials, n_available_snapshots = all_E.shape
                steps_for_h = steps[:n_available_snapshots]

                for trial_idx in range(n_trials):
                    trial_E_dens_diff = (all_E[trial_idx, :] / n_qubits) - E_dens_ref

                    if len(h_list) == 1:
                        label = f"trial {trial_idx + 1}"
                    else:
                        label = rf"$h = {h}$, trial {trial_idx + 1}"

                    plt.plot(steps_for_h, trial_E_dens_diff, label=label)
        else:
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
        plt.title(f"{Lx}x{Ly} resets {use_prob_resets}, nlayers = {nlayers}, trials = {trials}")
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')
        os.makedirs(outdir, exist_ok=True)
        
        # plotting gridline of exact energy for h=0 for given system size
        y_values = {
            (2, 2): -5/4,
            (2, 3): -8/7,
            (3, 2): -8/7,
            (3, 3): -13/12,
        }
        y = y_values.get((Lx, Ly), 0)
        plt.axhline(y=y, color = "tab:orange", linestyle = '--') 
            
        # ------------------------------------------------------------------------------------------------------------
        if plot_each_trial:
            fname = os.path.join(outdir, f"{Lx}x{Ly}_resets_{use_prob_resets}_individual_trials.png")
        else:
            fname = os.path.join(outdir, f"{Lx}x{Ly}_resets_{use_prob_resets}.png")
        # ------------------------------------------------------------------------------------------------------------
        
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot to: {fname}")


# ------------------------------------------------------------------------------------------------------------
# Plot gradient norms function
# ------------------------------------------------------------------------------------------------------------
def plot_gradient_norms(results):
        plt.figure(figsize=(6, 4))

        if plot_each_trial:
            for h, c in zip(h_list, colours):
                all_gradient_norms = results[h].get("all_gradient_norms")

                if all_gradient_norms is None:
                    continue

                all_gradient_norms = np.asarray(all_gradient_norms, dtype=float)
                n_trials, n_available_snapshots = all_gradient_norms.shape
                steps_for_h = steps[:n_available_snapshots]

                for trial_idx in range(n_trials):
                    if len(h_list) == 1:
                        label = f"trial {trial_idx + 1}"
                    else:
                        label = rf"$h = {h}$, trial {trial_idx + 1}"

                    plt.plot(steps_for_h, all_gradient_norms[trial_idx, :], label=label)
        else:
            for h, c in zip(h_list, colours):
                all_gradient_norms = results[h].get("all_gradient_norms")

                if all_gradient_norms is None:
                    continue

                all_gradient_norms = np.asarray(all_gradient_norms, dtype=float)

                mean_grad = np.mean(all_gradient_norms, axis=0)
                std_grad = np.std(all_gradient_norms, axis=0)

                n_available_snapshots = mean_grad.shape[0]
                steps_for_h = steps[:n_available_snapshots]

                label = rf"$h = {h}$"

                plt.plot(steps_for_h, mean_grad, color=c, label=label)

                plt.fill_between(
                    steps_for_h,
                    mean_grad - std_grad,
                    mean_grad + std_grad,
                    color=c,
                    alpha=0.3,
                )

        plt.yscale("log")
        plt.xlabel("Training steps")
        plt.ylabel("Gradient norm")
        plt.title(f"{Lx} x {Ly} gradient norms, Prob Resets: {use_prob_resets}")
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')

        os.makedirs(outdir, exist_ok=True)

        if plot_each_trial:
            fname = os.path.join(outdir, f"{Lx}x{Ly}_gradient_norms_prob_resets_{use_prob_resets}_individual_trials.png")
        else:
            fname = os.path.join(outdir, f"{Lx}x{Ly}_gradient_norms_prob_resets_{use_prob_resets}.png")

        plt.savefig(fname, dpi=200)
        plt.close()

        print(f"Saved gradient norm plot to: {fname}")


# ------------------------------------------------------------------------------------------------------------
# Plot probabilistic-reset theta values function
# ------------------------------------------------------------------------------------------------------------
def plot_prob_reset_theta_values(results):
        plt.figure(figsize=(6, 4))

        plotted_anything = False

        if plot_each_trial:
            for h, c in zip(h_list, colours):
                theta_history = results[h].get("all_prob_reset_theta_means")

                if theta_history is None:
                    continue

                theta_history = np.asarray(theta_history, dtype=float)

                if theta_history.ndim != 2:
                    continue

                n_trials, n_available_snapshots = theta_history.shape
                steps_for_h = steps[:n_available_snapshots]

                for trial_idx in range(n_trials):
                    if len(h_list) == 1:
                        label = f"trial {trial_idx + 1}"
                    else:
                        label = rf"$h = {h}$, trial {trial_idx + 1}"

                    plt.plot(steps_for_h, theta_history[trial_idx, :], label=label)
                    plotted_anything = True
        else:
            for h, c in zip(h_list, colours):
                theta_history = results[h].get("all_prob_reset_theta_means")

                if theta_history is None:
                    continue

                theta_history = np.asarray(theta_history, dtype=float)

                if theta_history.ndim != 2:
                    continue

                mean_theta = np.mean(theta_history, axis=0)
                std_theta = np.std(theta_history, axis=0)

                n_available_snapshots = mean_theta.shape[0]
                steps_for_h = steps[:n_available_snapshots]

                label = rf"$h = {h}$"

                plt.plot(steps_for_h, mean_theta, color=c, label=label)
                plt.fill_between(
                    steps_for_h,
                    mean_theta - std_theta,
                    mean_theta + std_theta,
                    color=c,
                    alpha=0.3,
                )
                plotted_anything = True

        if not plotted_anything:
            plt.close()
            print(
                "Warning: no probabilistic-reset theta history was found on the ansatz object, "
                "so no theta plot was written."
            )
            return None

        plt.xlabel("Training steps")
        plt.ylabel("Mean probabilistic-reset theta")
        plt.title(f"{Lx} x {Ly} reset theta values, Prob Resets: {use_prob_resets}")
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')

        os.makedirs(outdir, exist_ok=True)

        if plot_each_trial:
            fname = os.path.join(outdir, f"{Lx}x{Ly}_prob_reset_theta_values_resets_{use_prob_resets}_individual_trials.png")
        else:
            fname = os.path.join(outdir, f"{Lx}x{Ly}_prob_reset_theta_values_resets_{use_prob_resets}.png")

        plt.savefig(fname, dpi=200)
        plt.close()

        print(f"Saved probabilistic-reset theta plot to: {fname}")
        return fname


# ---------------------------------------------------------------------------------------------------------------------
# Global simulation parameters 
# ---------------------------------------------------------------------------------------------------------------------
Lx = 3
Ly = 2
nlayers = 1
howoften_tosave = 10
trials = 50
maxiter = 101
howoften_toreset = 7
unitary = True
sparse = True
perform_noisy_simulations = False
noise_rate = 5e-2
number_of_shots = 500
use_prob_resets = True
save_training_history = True
save_theta_history = True
plot_each_trial = False


tc_ = ToricCode(Lx, Ly)
n_qubits = tc_.num_qubits   

# comments are my very rough estimates from graph in paper
E_dens_ref_dict = {
    0.0: 0,         # -1.05
    0.12: 0,        # -1.00
    0.96: 0,        # -1.00
}

# Allow job scripts (e.g. Eddie/Grid Engine) to set a per-run output directory.
# Falls back to a local "outputs" folder when DPQC_OUTDIR is not set.
outdir = os.environ.get("DPQC_OUTDIR", "outputs")
h_list    = [0.0] #[0.0, 0.12, 0.96]
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
    print(f"plot_each_trial={plot_each_trial}")
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
                             )
    plotting(results)
    plot_gradient_norms(results)
    plot_prob_reset_theta_values(results)

    
    if save_training_history:
        training_history_csv_path = os.path.join(outdir, "training_history_by_trial_and_step.csv")
        save_training_history_csv(results, training_history_csv_path)

    if save_theta_history:
        theta_csv_path = os.path.join(outdir, "prob_reset_theta_by_trial_and_step.csv")
        save_prob_reset_theta_mean_csv(results, theta_csv_path)
