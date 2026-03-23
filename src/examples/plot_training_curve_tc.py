import numpy as np
import os

# use a non-interactive backend for headless environments 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utilities.generate_toric_code_hamiltonian import ToricCode
from src.utilities.ansatz_classes import ToricCodeAnsatz


def running_for_hs(Lx=2, Ly=2, nlayers_current=2, howoften_toreset=7, trials=10, maxiter=201,howoften_tosave=10, 
                   unitary=True, sparse=True, perform_noisy_simulations=False,number_of_shots=1000,use_prob_resets=False,
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

        final_E, final_purity, all_E, all_P = ansatz.optimize()
        results[h] = (all_E, all_P)

    return results

def plotting(results):
        # make plots for given h values
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
        plt.title(f"Energy Density")
        plt.legend()
        plt.tight_layout()
        plt.grid(visible=True, which='both', linestyle='--')
        os.makedirs(outdir, exist_ok=True)
        
        # ------------------------------------------------------------------------------------------------------------
        fname = os.path.join(outdir, f"training_curve.png") 
        # ------------------------------------------------------------------------------------------------------------
        
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved plot to: {fname}")


# ----------------------------------------------------------------------------------------------
# Global simulation parameters 
# ----------------------------------------------------------------------------------------------
Lx = 2
Ly = 2
nlayers = 2
howoften_tosave = 10
trials = 20      
maxiter = 301
howoften_toreset = 7
unitary = True
sparse = True
perform_noisy_simulations = False
noise_rate = 5e-2
number_of_shots = 600
use_prob_resets = True

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
    print(f"trials={trials}, maxiter={maxiter}")
    print(f"outdir={outdir}")
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
