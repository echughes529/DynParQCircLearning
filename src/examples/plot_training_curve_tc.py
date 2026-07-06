import csv
import numpy as np
import os

# use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.utilities.generate_toric_code_hamiltonian import ToricCode
from src.utilities.ansatz_classes import ToricCodeAnsatz


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


# ------------------------------------------------------------------------------------------------------------
# Save per-term Hamiltonian expectation values to CSV
# ------------------------------------------------------------------------------------------------------------
def save_term_expectations_csv(results, csv_path):
    """
    Save the unweighted expectation value <O_term> of every individual Hamiltonian
    term, at every recorded training-step snapshot, for every trial.

    One row is written for each `(h, trial, training_step, term)` quadruple.
    """
    rows = []

    for h in h_list:
        result_for_h = results[h]
        all_term_exp = result_for_h.get("all_term_expectations")
        term_labels = result_for_h.get("term_labels")
        term_coeffs = result_for_h.get("term_coeffs")

        if all_term_exp is None or term_labels is None:
            continue

        all_term_exp = np.asarray(all_term_exp, dtype=float)
        n_trials, n_available_snapshots, _n_terms = all_term_exp.shape
        steps_for_h = steps[:n_available_snapshots]

        for trial_idx in range(n_trials):
            for snapshot_idx, step in enumerate(steps_for_h):
                for term_idx, (label, family) in enumerate(term_labels):
                    expectation = float(all_term_exp[trial_idx, snapshot_idx, term_idx])
                    coeff = float(term_coeffs[term_idx]) if term_coeffs is not None else None

                    rows.append({
                        "h": h,
                        "trial": int(trial_idx + 1),
                        "training_step": int(step),
                        "term_label": label,
                        "term_family": family,
                        "expectation": expectation,
                        "weighted_expectation": (coeff * expectation) if coeff is not None else "",
                    })

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "h",
                "trial",
                "training_step",
                "term_label",
                "term_family",
                "expectation",
                "weighted_expectation",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved term expectations CSV to: {csv_path}")
    return csv_path


def running_for_hs(Lx=2, Ly=2, nlayers_current=2, howoften_toreset=7, trials=10, maxiter=201, howoften_tosave=10,
                   unitary=True, sparse=True, perform_noisy_simulations=False, number_of_shots=1000, use_reset_capable_ansatz=False,
                   reset_layers=None, trial_batch_size=None,
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
            use_reset_capable_ansatz=use_reset_capable_ansatz,
            reset_layers=reset_layers,
            trial_batch_size=trial_batch_size,
        )

        final_E, final_purity, all_E, all_P, all_param, all_grads, all_term_expectations = ansatz.optimize(
            track_params=track_params,
            track_grads=track_grads,
            track_term_expectations=track_term_expectations,
            term_expectations_batch_size=term_expectations_batch_size,
        )
        reset_layers_used = None
        if use_reset_capable_ansatz and hasattr(ansatz, "active_reset_layers"):
            reset_layers_used = list(ansatz.active_reset_layers)

        reset_param_slice = getattr(ansatz, "reset_param_slice", None)

        term_labels = ansatz.term_labels() if hasattr(ansatz, "term_labels") else None
        term_coeffs = [coeff for _, coeff in ansatz._hamiltonian_terms()] if track_term_expectations else None

        results[h] = {
            "final_E": final_E,
            "final_purity": final_purity,
            "all_E": all_E,
            "all_P": all_P,
            "all_param": all_param,
            "all_grads": all_grads,
            "all_term_expectations": all_term_expectations,
            "term_labels": term_labels,
            "term_coeffs": term_coeffs,
            "reset_layers_used": reset_layers_used,
            "reset_layers_input": reset_layers,
            "reset_param_slice": reset_param_slice,
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

            E_dens_ref = E_dens_ref_dict[h]
            mean_E_dens_diff = (mean_E / n_qubits) - E_dens_ref
            std_E_dens_diff  = std_E / n_qubits

            label = rf"$h = {h}$"

            plt.plot(steps_for_h, mean_E_dens_diff, color=c, label=label)
            plt.fill_between(
                steps_for_h,
                mean_E_dens_diff - std_E_dens_diff,
                mean_E_dens_diff + std_E_dens_diff,
                color=c,
                alpha=0.3,
            )
        
        plt.xlabel("Training steps")
        plt.ylabel("E/n")
        plt.title(f"{Lx}x{Ly}, nlayers:{nlayers}, trials: {trials}, reset_layers: {reset_layers}")
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
        fname = os.path.join(outdir, f"{Lx}x{Ly}_nlayers_{nlayers}_resets_{bool(reset_layers)}.png")
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

    # reset_layers_used is None or [] => no resets (same ansatz, zero reset params).
    active_reset_layers = list(reset_layers_used) if reset_layers_used else []
    reset_layers_label = str(active_reset_layers)

    n_reset_thetas = n_resets_per_layer * len(active_reset_layers)

    print(f"reset_layers input: {reset_layers_input}")
    print(f"reset_layers used: {active_reset_layers}")
    print(f"n_reset_thetas: {n_reset_thetas}")

    if n_reset_thetas == 0:
        print(f"No reset-theta parameters for h={h}; skipping theta plot.")
        return

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

        reference_energy_density = None

        if Lx == 2 and Ly == 2:
            reference_energy_density = -5/4  # for 2x2
        if Lx == 3:
            if Ly == 2:
                reference_energy_density = -8/7  # for 3x2
            if Ly == 3:
                reference_energy_density = -13/12  # for 3x3

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



# ------------------------------------------------------------------------------------------------------------
# Plot gradient norms over training
# ------------------------------------------------------------------------------------------------------------
def _plot_one_gradient_norm_series(gradient_norms, steps_for_h, title, fname):
    """Plot one gradient-norm series (all trials + mean), log-scale y, save to fname."""
    n_trials = gradient_norms.shape[0]
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
    mean_line, = plt.plot(steps_for_h, mean_grad_norm, color="black", linewidth=2, label="mean across trials")

    plt.yscale("log")
    plt.xlabel("Training steps")
    plt.ylabel("Gradient norm")
    plt.title(title)
    if n_trials <= 10:
        plt.legend()
    else:
        # Explicitly target the mean line's own handle, not plot order --
        # legend([label_string]) zips labels against ALL plotted artists in
        # the order they were added, so it would otherwise mislabel the
        # first per-trial line (blue) as "mean across trials" instead of
        # the actual black mean line.
        plt.legend(handles=[mean_line], labels=["mean across trials"])
    plt.tight_layout()
    plt.grid(visible=True, which='both', linestyle='--')

    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Saved gradient-norm plot to: {fname}")


def plotting_gradient_norms(results):
    """
    Plot L2 gradient norm over training, split per h into reset-param and
    non-reset-param plots (using reset_param_slice from running_for_hs).
    Falls back to one combined plot if no reset-param split is available.
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

        n_trials, n_available_snapshots, n_params = all_grads.shape
        steps_for_h = steps[:n_available_snapshots]

        print(f"all_grads shape for h={h}: {all_grads.shape}")

        reset_param_slice = results[h].get("reset_param_slice")

        if reset_param_slice is None:
            gradient_norms = np.linalg.norm(all_grads, axis=2)
            print(f"gradient_norms shape for h={h}: {gradient_norms.shape}")
            print(f"gradient_norms min/max for h={h}: {np.nanmin(gradient_norms)}, {np.nanmax(gradient_norms)}")

            fname = os.path.join(outdir, f"gradient_norm_h_{h}_all_trials.png")
            _plot_one_gradient_norm_series(
                gradient_norms,
                steps_for_h,
                title=f"Gradient norm, h={h}, {Lx}x{Ly}, nlayers:{nlayers}",
                fname=fname,
            )
            continue

        reset_idx = np.arange(n_params)[reset_param_slice]
        nonreset_mask = np.ones(n_params, dtype=bool)
        nonreset_mask[reset_idx] = False

        reset_norms = np.linalg.norm(all_grads[:, :, reset_idx], axis=2)
        nonreset_norms = np.linalg.norm(all_grads[:, :, nonreset_mask], axis=2)

        print(f"reset param slice for h={h}: {reset_param_slice} ({reset_idx.size} params)")
        print(f"reset_norms shape for h={h}: {reset_norms.shape}")
        print(f"reset_norms min/max for h={h}: {np.nanmin(reset_norms)}, {np.nanmax(reset_norms)}")
        print(f"nonreset_norms shape for h={h}: {nonreset_norms.shape}")
        print(f"nonreset_norms min/max for h={h}: {np.nanmin(nonreset_norms)}, {np.nanmax(nonreset_norms)}")

        reset_fname = os.path.join(outdir, f"gradient_norm_reset_h_{h}_all_trials.png")
        _plot_one_gradient_norm_series(
            reset_norms,
            steps_for_h,
            title=f"Reset-param gradient norm, h={h}, {Lx}x{Ly}, nlayers:{nlayers}",
            fname=reset_fname,
        )

        nonreset_fname = os.path.join(outdir, f"gradient_norm_nonreset_h_{h}_all_trials.png")
        _plot_one_gradient_norm_series(
            nonreset_norms,
            steps_for_h,
            title=f"Non-reset-param gradient norm, h={h}, {Lx}x{Ly}, nlayers:{nlayers}",
            fname=nonreset_fname,
        )


# ------------------------------------------------------------------------------------------------------------
# Plot per-Hamiltonian-term expectation values over training
# ------------------------------------------------------------------------------------------------------------
def plotting_term_expectations_by_family(results):
    """
    Plot the mean expectation value of each Hamiltonian-term family (star,
    plaquette, and field when its coefficient is nonzero) over training, one
    line per trial, colored by that trial's own final energy.

    Unlike averaging across trials, this keeps every trial visible so that
    trials landing on the wrong final energy can be visually compared
    against trials that converged correctly, to see which family of
    stabilizers failed to lock in.
    """
    os.makedirs(outdir, exist_ok=True)

    for h in h_list:
        all_term_exp = results[h].get("all_term_expectations")
        term_labels = results[h].get("term_labels")
        term_coeffs = results[h].get("term_coeffs")
        final_E = results[h].get("final_E")

        if all_term_exp is None or term_labels is None:
            print(f"No term-expectation history found for h={h}; skipping by-family term plot.")
            continue

        all_term_exp = np.asarray(all_term_exp, dtype=float)
        final_E = np.asarray(final_E, dtype=float)

        if all_term_exp.ndim != 3:
            raise ValueError(
                f"Expected all_term_expectations to have shape (trials, snapshots, nterms), "
                f"but got shape {all_term_exp.shape}"
            )

        n_trials, n_available_snapshots, n_terms = all_term_exp.shape
        steps_for_h = steps[:n_available_snapshots]

        print(f"all_term_expectations shape for h={h}: {all_term_exp.shape}")

        # Group term indices by family, dropping any term whose coefficient
        # is 0 for this h (e.g. the field terms when h=0 -- they have no
        # effect on the energy at that h, so they're not worth plotting).
        family_indices = {}
        for term_idx, (label, family) in enumerate(term_labels):
            coeff = term_coeffs[term_idx] if term_coeffs is not None else None
            if coeff is not None and coeff == 0:
                continue
            family_indices.setdefault(family, []).append(term_idx)

        norm = mcolors.Normalize(vmin=np.nanmin(final_E), vmax=np.nanmax(final_E))
        cmap = plt.cm.viridis

        for family, idxs in family_indices.items():
            family_mean = all_term_exp[:, :, idxs].mean(axis=2)  # (trials, snapshots)

            print(f"family={family} for h={h}: {len(idxs)} terms, "
                  f"mean min/max: {np.nanmin(family_mean)}, {np.nanmax(family_mean)}")

            fig, ax = plt.subplots(figsize=(5, 4))
            for trial_idx in range(n_trials):
                ax.plot(
                    steps_for_h,
                    family_mean[trial_idx],
                    color=cmap(norm(final_E[trial_idx])),
                    alpha=0.8,
                    linewidth=1,
                )

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="final energy")

            ax.set_xlabel("Training steps")
            ax.set_ylabel(rf"mean $\langle O_{{\mathrm{{{family}}}}} \rangle$")
            ax.set_title(f"{family.capitalize()} terms, h={h}, {Lx}x{Ly}, nlayers:{nlayers}")
            ax.grid(visible=True, which='both', linestyle='--')
            fig.tight_layout()

            fname = os.path.join(outdir, f"term_expectations_{family}_h_{h}.png")
            fig.savefig(fname, dpi=200)
            plt.close(fig)
            print(f"Saved by-family term-expectation plot to: {fname}")


def plotting_term_expectations_individual(results, trial_idx, h=None, label_suffix=""):
    """
    Plot every individual Hamiltonian term's expectation value over training,
    for a single trial, colour-coded by family (star / plaquette / field).

    Drill-down companion to plotting_term_expectations_by_family: use this to
    inspect one specific trial (e.g. a trial that landed on the wrong energy)
    in full detail.
    """
    os.makedirs(outdir, exist_ok=True)
    family_colours = {"star": "tab:blue", "plaquette": "tab:red", "field": "tab:green"}

    for h_val in ([h] if h is not None else h_list):
        all_term_exp = results[h_val].get("all_term_expectations")
        term_labels = results[h_val].get("term_labels")
        term_coeffs = results[h_val].get("term_coeffs")

        if all_term_exp is None or term_labels is None:
            print(f"No term-expectation history found for h={h_val}; skipping individual-term plot.")
            continue

        all_term_exp = np.asarray(all_term_exp, dtype=float)
        n_trials, n_available_snapshots, n_terms = all_term_exp.shape
        steps_for_h = steps[:n_available_snapshots]

        if trial_idx >= n_trials:
            print(f"trial_idx={trial_idx} out of range (only {n_trials} trials) for h={h_val}; skipping.")
            continue

        plt.figure(figsize=(5, 4))
        families_seen = []
        for term_idx, (label, family) in enumerate(term_labels):
            coeff = term_coeffs[term_idx] if term_coeffs is not None else None
            if coeff is not None and coeff == 0:
                continue
            colour = family_colours.get(family, "tab:gray")
            plt.plot(
                steps_for_h,
                all_term_exp[trial_idx, :, term_idx],
                color=colour,
                alpha=0.6,
                linewidth=1,
            )
            if family not in families_seen:
                families_seen.append(family)

        legend_handles = [
            plt.Line2D([0], [0], color=family_colours.get(fam, "tab:gray"), label=fam)
            for fam in families_seen
        ]
        plt.legend(handles=legend_handles)

        plt.xlabel("Training steps")
        plt.ylabel(r"$\langle O_{\mathrm{term}} \rangle$")
        plt.title(f"Per-term expectations, trial {trial_idx + 1}{label_suffix}, h={h_val}, {Lx}x{Ly}, nlayers:{nlayers}")
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')

        fname = os.path.join(outdir, f"term_expectations_individual_trial{trial_idx + 1}_h_{h_val}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved individual-trial term-expectation plot to: {fname}")


def plotting_term_expectations_best_worst(results):
    """
    Convenience wrapper: auto-generate the individual-term drill-down plot
    for the best (lowest) and worst (highest) final-energy trial at each h.
    """
    for h in h_list:
        final_E = results[h].get("final_E")
        if final_E is None:
            continue
        final_E = np.asarray(final_E, dtype=float)
        best_idx = int(np.nanargmin(final_E))
        worst_idx = int(np.nanargmax(final_E))
        plotting_term_expectations_individual(results, best_idx, h=h, label_suffix=" (best)")
        if worst_idx != best_idx:
            plotting_term_expectations_individual(results, worst_idx, h=h, label_suffix=" (worst)")


# ---------------------------------------------------------------------------------------------------------------------
# Global simulation parameters
# ---------------------------------------------------------------------------------------------------------------------
Lx = 3
Ly = 3
nlayers = 2
howoften_tosave = 10
trials = 100
maxiter = 2000
howoften_toreset = 7
unitary = True
sparse = True
perform_noisy_simulations = False
noise_rate = 5e-2
number_of_shots = 500
use_reset_capable_ansatz = True

# Bounds how many trials get vmapped together per training step (the main
# energy/gradient path, not term-expectation tracking below). Each trial
# needs a full dense state vector live at once, so vmapping all `trials`
# together needs trials * 2**nqubits amplitudes simultaneously - this is
# the main lever for fitting bigger lattices onto smaller-VRAM GPUs (e.g.
# A40 instead of H200). Lower this if you hit out-of-memory errors during
# training specifically (not Hamiltonian construction); set to None to vmap
# all trials at once (fastest, highest peak memory).
trial_batch_size = None

# Choose which ansatz layers get probabilistic resets. This is the sole
# on/off control: None (or []) means no resets at all, using the exact
# same ansatz structure as when resets are active.
# Layer indexing is zero-based, so [0] means only the first layer.
reset_layers = [1]

track_grads = True
track_params = True
track_term_expectations = False
# Per-term expectation tracking is far more expensive than the energy/gradient
# path (one full circuit contraction per Hamiltonian term, not shared like the
# combined sparse Hamiltonian), so trials are processed in batches to bound
# peak memory rather than vmapping over all trials at once. Lower this if you
# still hit out-of-memory errors; set to None to disable batching entirely.
term_expectations_batch_size = None

plot_final_energies = True
save_final_energies = False
save_training_history = False
save_term_expectations = False
plot_term_expectations = False
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
    print(f"trials={trials}, maxiter={maxiter}, trial_batch_size={trial_batch_size}")
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
                             use_reset_capable_ansatz=use_reset_capable_ansatz,
                             reset_layers=reset_layers,
                             trial_batch_size=trial_batch_size,
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

    if save_term_expectations:
        term_expectations_csv_path = os.path.join(outdir, "term_expectations.csv")
        save_term_expectations_csv(results, term_expectations_csv_path)

    if plot_term_expectations:
        plotting_term_expectations_by_family(results)
        plotting_term_expectations_best_worst(results)