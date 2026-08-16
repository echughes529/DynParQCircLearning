"""Full production-scale 3x3 ground-state run under the QR SVD driver.

The real pipeline end to end: VariationalAnsatz.optimize() (guarded Adam,
tqdm, snapshots every howoften_tosave, HDF5 saving via save_results=True) at
the production configuration that previously died with NaNs (3x3, bond_dim=96,
20 trials, lr=0.01, maxiter=1300), with the SVD forward switched to the QR
driver by the DPQC_SVD_FWD_ALG=qr env hook set in the sbatch script. A fresh
seed (default 2026) is used deliberately: seed 72314 (the run that died at
step ~215) is already covered by the 500-step probe, so this run demonstrates
the fix is not seed-specific AND that the full loop converges and saves.

At h=0 every Hamiltonian term is a commuting stabilizer with coefficient -1
(4 plaquettes + 9 stars at 3x3; the 12 h-perturbation terms carry weight 0)
and the ground state satisfies all of them, so the exact ground energy is the
sum of the term coefficients: -13.0, confirmed by sparse diagonalization of
the assembled 12-qubit Hamiltonian (next eigenvalues -11, -11). It is printed
up front for comparison.

Env: DPQC_CAP_SEED (2026), DPQC_CAP_NITER (1300), DPQC_CAP_BOND_DIM (96)
     plus DPQC_SVD_FWD_ALG=qr from the sbatch wrapper.
"""
import contextlib
import io
import os
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from src.utilities.ansatz_classes import ToricCodeAnsatz

BOND_DIM = int(os.environ.get("DPQC_CAP_BOND_DIM", 96))
SEED = int(os.environ.get("DPQC_CAP_SEED", 2026))
NITER = int(os.environ.get("DPQC_CAP_NITER", 1300))


def main():
    if os.environ.get("DPQC_SVD_FWD_ALG", "").lower() != "qr":
        print("WARNING: DPQC_SVD_FWD_ALG=qr is not set; this run uses the stock kernel",
              flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ansatz = ToricCodeAnsatz(
            Lx=3, Ly=3, nlayers=2, howoften_toreset=7, h=0.0,
            use_prob_resets_ansatz=True, prob_reset_direction=1, reset_layers=[1],
            unitary=True, bond_dim=BOND_DIM, use_optimal_ordering=True,
            cartan_mode="fused", toffoli_mode="direct",
            trials=20, maxiter=NITER, learning_rate=0.01,
            sparse=False, use_mps=True, normalize_state=True, seed=SEED,
        )
    terms = ansatz._hamiltonian_terms()
    e0 = float(sum(coeff for _, coeff in terms))
    print(f"qr_full_run: 3x3 bd={BOND_DIM} seed={SEED} maxiter={NITER} trials=20\n"
          f"exact ground energy (h=0, sum of {len(terms)} term coefficients): {e0:+.1f}",
          flush=True)

    final_E, final_params, all_E, *_ = ansatz.optimize(save_results=True)

    fe = np.asarray(final_E)
    print(f"\nfinal per-trial energies:\n{np.array2string(fe, precision=8)}")
    print(f"min {fe.min():+.10f} | mean {fe.mean():+.10f} | "
          f"finite {int(np.isfinite(fe).sum())}/20 | "
          f"gap to exact: min {fe.min()-e0:.3e}, mean {fe.mean()-e0:.3e}", flush=True)


if __name__ == "__main__":
    main()
