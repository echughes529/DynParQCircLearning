"""Post-training analysis of one Experiment 1 run folder.

Run:
    python -m src.experiments.exp1_analyse_run experiments/experiment_1_scaling_system_size/3x2/mc_n8
    ./submit_diag.sh NAME=exp1_an_3x2_mc_n8 TIME=02:00:00 -- \
        python -m src.experiments.exp1_analyse_run experiments/experiment_1_scaling_system_size/3x2/mc_n8

A run folder is one method on one lattice. It reads run.h5 and run_config.json
(written by plot_training_curve_tc when DPQC_RUN_DIR is set) and writes

    analysis.h5   MC methods only: exact energy and sigma per (trial, snapshot)
    metrics.csv   one row per (trial, logged step)
    summary.json  one record for the comparison script
    plots/        training curve, sigma and p_reset histories

EXACT ENERGY AND SIGMA
----------------------
Purified and enumerated log the exact reset-channel energy during training, so
their logged energy is used as is and sigma is 0.

The MC methods log a noisy n-sample estimate. Every saved parameter vector is
re-evaluated with an enumerated twin of the ansatz (same lattice, reset qubits,
bond dimension and parameter layout), whose trajectory_table returns, for all
3^R trajectories i, (p^(i) E^(i), p^(i)). From those

    E        = sum_i p^(i) E^(i)
    sigma^2  = (1/n) sum_i p^(i) (E^(i) - E)^2        n = this method's n

which is the exact variance of the method's own n-sample energy estimate.

ALIGNMENT
---------
optimize() logs the energy of the parameters *before* step i's update but saves
the parameters *after* it. So the MC exact energy at row "step i" belongs to the
parameters at the start of step i+1. With logging every 10 steps this one-step
offset is immaterial for convergence; metrics.csv records it as params_step.
"""

import argparse
import csv
import os
import re
import sys

import numpy as np

from src.experiments.exp1_common import (
    CONVERGENCE_REL_TOL, METHODS, dump_json, exact_ground_energy, load_json,
    n_qubits, size_label, steps_to_converge,
)


def _nanmean(x):
    """Mean ignoring NaN; NaN (not a warning) when nothing was measured."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# ----------------------------------------------------------------------------
# Run status from the job's stdout
# ----------------------------------------------------------------------------
def run_status(run_dir):
    """'ok', 'oom', 'timeout', 'failed' or 'missing'."""
    stdout = os.path.join(run_dir, "stdout.out")
    stderr = os.path.join(run_dir, "stderr.err")
    if not os.path.exists(stdout):
        return "missing"
    text = open(stdout, errors="replace").read()
    err = open(stderr, errors="replace").read() if os.path.exists(stderr) else ""
    m = re.search(r"Job finished with exit code: (\d+)", text)
    if m and m.group(1) == "0" and os.path.exists(os.path.join(run_dir, "run.h5")):
        return "ok"
    if re.search(r"RESOURCE_EXHAUSTED|Out of memory|out of memory", text + err):
        return "oom"
    # Slurm's cgroup OOM killer sends SIGKILL: exit 137 and a bare "Killed",
    # with no Python traceback. That is host RAM, not GPU memory.
    if (m and m.group(1) == "137") or re.search(r"\bKilled\s+python", err):
        return "oom_host"
    if m is None:
        # No closing line: the job was killed, normally at walltime.
        return "timeout"
    return "failed"


# ----------------------------------------------------------------------------
# Exact energy + sigma via an enumerated twin
# ----------------------------------------------------------------------------
def build_enumerated_twin(cfg):
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    return ToricCodeAnsatz(
        Lx=cfg["Lx"], Ly=cfg["Ly"], nlayers=cfg["nlayers"], h=cfg["h"],
        trials=1, maxiter=1, bond_dim=cfg["bond_dim"],
        use_optimal_ordering=cfg["use_optimal_ordering"],
        use_ladder_ordering=cfg.get("use_ladder_ordering", False),
        use_prob_resets_ansatz=True, unitary=True, sparse=False, use_mps=True,
        reset_layers=cfg["reset_layers"],
        which_qubits_for_prob_reset=cfg["reset_qubits"],
        shared_reset_theta=cfg["shared_reset_theta"],
        use_trajectory_resets=False, use_enumerated_resets=True,
        max_reset_branches=3 ** cfg["total_resets"],
        seed=cfg["seed"],
    )


def exact_energy_and_variance(twin, params, batch):
    """params (M, nparams) -> (E (M,), Var_i[E^(i)] (M,)).

    Var is sum_i p^(i) (E^(i) - E)^2, computed from the branch table with
    zero-probability branches dropped (their E^(i) is undefined).
    """
    import jax
    import jax.numpy as jnp

    table = jax.jit(jax.vmap(twin.trajectory_table))
    Es, Vs = [], []
    for start in range(0, len(params), batch):
        chunk = jnp.asarray(params[start:start + batch])
        nums, dens = table(chunk)                      # (b, 3^R) each
        nums, dens = np.asarray(nums), np.asarray(dens)
        D = dens.sum(axis=1)
        E = nums.sum(axis=1) / D
        live = dens > 1e-14 * D[:, None]
        Ei = np.where(live, nums / np.where(live, dens, 1.0), 0.0)
        p = dens / D[:, None]
        V = np.sum(np.where(live, p * (Ei - E[:, None]) ** 2, 0.0), axis=1)
        Es.append(E)
        Vs.append(V)
        print(f"  evaluated {min(start + batch, len(params))}/{len(params)} parameter vectors",
              flush=True)
    return np.concatenate(Es), np.concatenate(Vs)


# ----------------------------------------------------------------------------
# Plots (per run)
# ----------------------------------------------------------------------------
def plot_run(out_dir, steps, exact_E, sigma, p_reset, E0, cfg, method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    label, colour = METHODS.get(method, (method, "tab:gray"))
    tag = f"{size_label(cfg['Lx'], cfg['Ly'])}, {label}"

    def trials_and_mean(ax, y, ylabel):
        for t in range(y.shape[0]):
            ax.plot(steps, y[t], color=colour, alpha=0.35, lw=1)
        ax.plot(steps, y.mean(axis=0), color=colour, lw=2)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.5)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    trials_and_mean(ax, exact_E, "Energy")
    ax.axhline(E0, color="k", ls="--", lw=1)
    ax.set_title(tag)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=200)
    plt.close(fig)

    if sigma is not None and np.any(sigma > 0):
        fig, ax = plt.subplots(figsize=(5, 3.8))
        trials_and_mean(ax, sigma, r"$\sigma[\hat E_n]$")
        ax.set_yscale("log")
        ax.set_title(tag)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sigma.png"), dpi=200)
        plt.close(fig)

    if p_reset is not None:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        trials_and_mean(ax, p_reset, r"$p_{\mathrm{reset}}$")
        ax.set_title(tag)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "p_reset.png"), dpi=200)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def analyse(run_dir, batch=None):
    method = os.path.basename(os.path.normpath(run_dir))
    lattice = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    status = run_status(run_dir)
    print(f"[exp1] {lattice}/{method}: status {status}")

    if status != "ok":
        # Failure is a data point: the comparison plots draw it as a hollow
        # marker. Keep whatever config we have so the size is known.
        Lx, Ly = (int(v) for v in lattice.split("x"))
        summary = {"lattice": lattice, "method": method, "status": status,
                   "Lx": Lx, "Ly": Ly, "n_qubits": n_qubits(Lx, Ly)}
        cfg_path = os.path.join(run_dir, "run_config.json")
        if os.path.exists(cfg_path):
            summary["bond_dim"] = load_json(cfg_path).get("bond_dim")
        dump_json(summary, os.path.join(run_dir, "summary.json"))
        return summary

    from src.utilities.result_saver import ResultSaver

    cfg = load_json(os.path.join(run_dir, "run_config.json"))
    res = ResultSaver.load_run(os.path.join(run_dir, "run.h5"))["results"]

    logged_E = np.asarray(res["all_energies"], dtype=float)       # (T, S)
    all_params = np.asarray(res["all_params"], dtype=float)       # (T, S, P)
    all_grads = np.asarray(res.get("all_grads", np.zeros((0,))), dtype=float)
    step_times = np.asarray(res["step_times"], dtype=float)
    T, S = logged_E.shape
    steps = np.arange(S) * cfg["howoften_tosave"]
    if all_params.ndim != 3 or all_params.shape[1] != S:
        raise SystemExit(f"{run_dir}: all_params has shape {all_params.shape}; "
                         f"was the run made with track_params=True?")

    Lx, Ly = cfg["Lx"], cfg["Ly"]
    E0 = exact_ground_energy(Lx, Ly)
    mode = cfg["reset_mode"]

    if mode == "traj":
        n = cfg["n_trajectories"]
        twin = build_enumerated_twin(cfg)
        if twin.nparams != all_params.shape[2]:
            raise SystemExit(f"twin has {twin.nparams} params, run has {all_params.shape[2]}")
        flat = all_params.reshape(T * S, -1)
        E_flat, V_flat = exact_energy_and_variance(twin, flat, batch or T)
        exact_E = E_flat.reshape(T, S)
        sigma = np.sqrt(np.maximum(V_flat, 0.0) / n).reshape(T, S)
        params_step = steps + 1
        import h5py
        with h5py.File(os.path.join(run_dir, "analysis.h5"), "w") as f:
            f.create_dataset("steps", data=steps)
            f.create_dataset("params_step", data=params_step)
            f.create_dataset("exact_energy", data=exact_E)
            f.create_dataset("var_trajectory_energy", data=V_flat.reshape(T, S))
            f.create_dataset("sigma", data=sigma)
            f.attrs["n_trajectories"] = n
    else:
        exact_E = logged_E
        sigma = np.zeros_like(exact_E)
        params_step = steps

    # p_reset from the (shared) reset theta
    idx = cfg["reset_param_indices"]
    p_reset = np.sin(all_params[:, :, idx[0]] / 2.0) ** 2 if idx else None

    grad_norm = (np.linalg.norm(all_grads, axis=-1)
                 if all_grads.ndim == 3 and all_grads.shape[1] == S else None)

    # Wall-clock: cumulative seconds at the end of each logged step, compilation
    # (step 0) included -- it is real time spent before that point.
    cum_t = np.nancumsum(step_times)
    t_at = cum_t[np.minimum(steps, len(cum_t) - 1)]

    rel_err = np.abs(exact_E - E0) / abs(E0)
    conv_step = [steps_to_converge(rel_err[t], steps) for t in range(T)]
    conv_time = [None if s is None else float(t_at[list(steps).index(s)]) for s in conv_step]

    # metrics.csv
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial", "step", "params_step", "cumulative_seconds", "logged_energy",
                    "exact_energy", "sigma", "p_reset", "grad_norm"])
        for t in range(T):
            for k in range(S):
                w.writerow([t, int(steps[k]), int(params_step[k]), f"{t_at[k]:.6g}",
                            f"{logged_E[t, k]:.12g}", f"{exact_E[t, k]:.12g}",
                            f"{sigma[t, k]:.6g}",
                            "" if p_reset is None else f"{p_reset[t, k]:.6g}",
                            "" if grad_norm is None else f"{grad_norm[t, k]:.6g}"])

    def mean_or_none(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    warm = step_times[1:]
    summary = {
        "lattice": f"{Lx}x{Ly}", "method": method, "status": "ok",
        "Lx": Lx, "Ly": Ly, "n_qubits": n_qubits(Lx, Ly), "E0": E0,
        "reset_mode": mode, "n_trajectories": cfg["n_trajectories"],
        "reset_qubits": cfg["reset_qubits"], "bond_dim": cfg["bond_dim"],
        "n_mps_qubits": cfg["n_mps_qubits"], "trials": T, "maxiter": cfg["maxiter"],
        # cost
        "compile_seconds": float(step_times[0]),
        "mean_step_seconds": float(np.nanmean(warm)) if warm.size else None,
        "total_seconds": float(np.nansum(step_times)),
        "peak_gpu_bytes": float(res.get("peak_gpu_bytes", np.nan)),
        "mean_live_gpu_bytes": _nanmean(res.get("gpu_live_bytes", [])),
        "device": str(res.get("device", "")),
        "n_nonfinite_steps": int(res.get("n_nonfinite_steps", 0)),
        # accuracy (exact sigma of this method's own n-sample estimate)
        "final_sigma_per_trial": sigma[:, -1],
        "final_sigma_mean": float(sigma[:, -1].mean()),
        "training_sigma_mean": float(sigma.mean()),
        # energy and convergence
        "final_energy_per_trial": exact_E[:, -1],
        "final_energy_mean": float(exact_E[:, -1].mean()),
        "final_p_reset_mean": None if p_reset is None else float(p_reset[:, -1].mean()),
        "convergence_rel_tol": CONVERGENCE_REL_TOL,
        "n_converged": int(sum(s is not None for s in conv_step)),
        "steps_to_converge_per_trial": conv_step,
        "seconds_to_converge_per_trial": conv_time,
        "steps_to_converge_mean": mean_or_none(conv_step),
        "seconds_to_converge_mean": mean_or_none(conv_time),
    }
    dump_json(summary, os.path.join(run_dir, "summary.json"))

    plot_run(os.path.join(run_dir, "plots"), steps, exact_E,
             sigma if mode == "traj" else None, p_reset, E0, cfg, method)
    print(f"[exp1] wrote metrics.csv, summary.json, plots/ in {run_dir}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--batch", type=int, default=None,
                    help="parameter vectors per enumerated evaluation (default: trials)")
    args = ap.parse_args(argv)
    for d in args.run_dirs:
        analyse(d, batch=args.batch)


if __name__ == "__main__":
    sys.exit(main())
