"""Shared pieces for Experiment 1 (scaling with system size).

See experiments/experiment_1_scaling_system_size/PLAN.md for the design.
"""

import json
import os

import numpy as np

from src.utilities.generate_toric_code_hamiltonian import ToricCode, toric_ground_energy_h0

EXP_DIR = os.path.join("experiments", "experiment_1_scaling_system_size")

# Folder name -> (plot label, colour). Order is the plotting order.
METHODS = {
    "mc_n1": ("MC, n=1", "tab:orange"),
    "mc_n8": ("MC, n=8", "tab:red"),
    "purified": ("Purified", "tab:blue"),
    "enumerated": ("Enumerated", "tab:green"),
}

# Convergence: exact energy within this fraction of E0.
CONVERGENCE_REL_TOL = 1e-3


def lattice_name(Lx, Ly):
    return f"{Lx}x{Ly}"


def n_qubits(Lx, Ly):
    return ToricCode(Lx, Ly).num_qubits


def exact_ground_energy(Lx, Ly):
    """E0 at h=0; see generate_toric_code_hamiltonian.toric_ground_energy_h0."""
    return toric_ground_energy_h0(Lx, Ly)


def size_label(Lx, Ly):
    """Axis label for a lattice: '3×2 (7)' -- size, then qubit count."""
    return f"{Lx}×{Ly} ({n_qubits(Lx, Ly)})"


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def dump_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.ndarray):
        return [_json_default(v) if isinstance(v, (np.generic, np.ndarray)) else v
                for v in x.tolist()]
    raise TypeError(f"not JSON serialisable: {type(x)}")


def steps_to_converge(rel_err, steps, tol=CONVERGENCE_REL_TOL):
    """First step after which rel_err stays <= tol to the end, else None.

    rel_err and steps are 1-D and aligned. A trial that dips under the
    threshold and comes back out has not converged at that dip.
    """
    ok = np.asarray(rel_err) <= tol
    if not ok[-1]:
        return None
    # last index that is NOT ok; convergence starts just after it
    bad = np.flatnonzero(~ok)
    first = 0 if bad.size == 0 else bad[-1] + 1
    return int(steps[first])
