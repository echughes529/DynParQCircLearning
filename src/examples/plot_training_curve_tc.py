import numpy as np
import os

# use a non-interactive backend for headless environments 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utilities.generate_toric_code_hamiltonian import ToricCode
from src.utilities.ansatz_classes import ToricCodeAnsatz


def run_for_h(h, howoften_toreset, learning_rate, nlayers_current=None,
              ):
    """
    Run the optimisation for a single value of h.

    Parameters
    ----------
    h : float
        Transverse field strength.
    howoften_toreset : int or None
        Reset frequency to use in this run.
    nlayers_current : int or None, optional
        Number of layers to use. If None, falls back to the global `nlayers`.
    """
    if nlayers_current is None:
        nlayers_current = nlayers

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
        number_of_shots=number_of_shots
    )

    final_E, final_purity, all_E, all_P = ansatz.optimize()
    return all_E, all_P

def plots_varying_given_parameter(variable_name: str, variable_values):
    """
    Vary one of:
        - howoften_toreset
        - learning_rate
        - nlayers

    For each value in `variable_values`, run all h values and make a plot.
    """

    if variable_name not in ("howoften_toreset", "learning_rate", "nlayers"):
        raise ValueError("variable_name must be 'howoften_toreset', 'learning_rate', or 'nlayers'")
    
    # choose which parameter we are varying
    for value in variable_values:
        if variable_name == "howoften_toreset":
            current_reset = value
            current_lr = learning_rate          # use the global default learning rate
            current_nlayers = nlayers           # use the global default number of layers
        elif variable_name == "nlayers":
            current_reset = howoften_toreset
            current_lr = learning_rate
            current_nlayers = value
        else:
            # This should be unreachable because of the earlier check,
            # but keep it for safety.
            raise ValueError(f"Unsupported variable_name: {variable_name}")

        # run all h for this parameter value
        results = {}
        for h in h_list:
            all_E, all_P = run_for_h(h, current_reset, current_lr, nlayers_current=current_nlayers)
            results[h] = (all_E, all_P)

        # make the plot for this parameter value
        plt.figure(figsize=(5, 4))
        for h, c in zip(h_list, colours):
            all_E, _ = results[h]      # all_E shape: (trials, n_snapshots)

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
        plt.title(f"Energy Density, {variable_name} = {value}")
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')

        os.makedirs(outdir, exist_ok=True)
        # Make filename depend on the parameter being varied so plots are not overwritten
        value_str = str(value).replace('.', 'p')
        fname = os.path.join(outdir, f"training_curve_{variable_name}_{value_str}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot to: {fname}")



# global simulation parameters 
Lx = 2
Ly = 2
nlayers = 2
howoften_tosave = 10
trials = 200       
maxiter = 401
learning_rate = 0.1
howoften_toreset = 7
unitary = True
sparse=True
perform_noisy_simulations=False
noise_rate=5e-2
number_of_shots=2000
use_prob_resets=True

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
h_list    = [0.0, 0.12, 0.96]
colours   = ["tab:orange", "tab:blue", "turquoise"]
n_snapshots = 1 + maxiter // howoften_tosave
steps = np.arange(n_snapshots) * howoften_tosave

if __name__ == "__main__":
    # ------- Printing Parameters ---------
    print("===== Python run parameters =====")
    print(f"Lx={Lx}, Ly={Ly}")
    print(f"nlayers={nlayers}")
    print(f"howoften_toreset={howoften_toreset}")
    print(f"learning_rate={learning_rate}")
    print(f"trials={trials}, maxiter={maxiter}")
    print(f"outdir={outdir}")
    print("=================================")

    reset_list = [7]      # howoften_toreset
    lr_list = [1e-3, 1e-2, 1e-1]          # learning_rate
    layers_list = [2]           # nlayers

    plots_varying_given_parameter("howoften_toreset", reset_list)
