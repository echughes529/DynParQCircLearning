"""Cross-lattice comparison plots for Experiment 1.

Run (after exp1_analyse_run has written summary.json in every run folder):
    python -m src.experiments.exp1_compare
    python -m src.experiments.exp1_compare --exp-dir path/to/experiment_folder
    # Experiment 1.1 with Experiment 1 overlaid (extra overlay_*.png plots):
    python -m src.experiments.exp1_compare \
        --exp-dir experiments/experiment_1.1_scaling_system_size_nx3/caps_64_96 \
        --reference-exp-dir experiments/experiment_1_scaling_system_size

Reads <exp>/<lattice>/<method>/summary.json and metrics.csv and writes
<exp>/comparisons/: summary.csv and the plots listed in PLAN.md. Pilot runs
(<lattice>/pilot/...) are ignored. Bond dimension "needed" comes from
<exp>/config.json if Phase 1 has filled it in (bond_dim_needed[lattice][method]),
otherwise that plot is skipped.
"""

import argparse
import csv
import glob
import os

import numpy as np

from src.experiments.exp1_common import (
    EXP_DIR, METHODS, exact_ground_energy, load_json, size_label,
)

COST_METRICS = [
    # key in row, y-axis label, file name
    ("bond_dim_needed", "Bond dimension needed", "cost_vs_size_bond_dim.png"),
    ("peak_gpu_gib", "Peak GPU memory (GiB)", "cost_vs_size_peak_gpu.png"),
    ("mean_step_seconds", "Time per step (s)", "cost_vs_size_step_time.png"),
    ("total_hours", "Total training time (h)", "cost_vs_size_total_time.png"),
]
MARKERS = {"mc_n1": "o", "mc_n8": "s", "purified": "^", "enumerated": "D"}


def collect(exp_dir):
    cfg_path = os.path.join(exp_dir, "config.json")
    bd_needed = load_json(cfg_path).get("bond_dim_needed", {}) if os.path.exists(cfg_path) else {}
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "*x*", "*", "summary.json"))):
        s = load_json(path)
        # The folder name is the method. A summary moved to an archive folder
        # (e.g. 80x2/mc_n1_oom64G) still says method=mc_n1 inside, so the
        # field alone would double-count it alongside the current run.
        if os.path.basename(os.path.dirname(path)) != s.get("method") or s.get("method") not in METHODS:
            continue
        s["run_dir"] = os.path.dirname(path)
        s["bond_dim_needed"] = bd_needed.get(s["lattice"], {}).get(s["method"])
        if s.get("peak_gpu_bytes") is not None:
            s["peak_gpu_gib"] = s["peak_gpu_bytes"] / 2**30
        if s.get("total_seconds") is not None:
            s["total_hours"] = s["total_seconds"] / 3600
        # NaN (e.g. GPU memory on a CPU run) means "not measured": treat it as
        # missing so it is skipped rather than breaking a log axis.
        for k, v in list(s.items()):
            if isinstance(v, float) and not np.isfinite(v):
                s[k] = None
        rows.append(s)
    return rows


def write_summary_csv(rows, path):
    cols = ["lattice", "n_qubits", "method", "status", "bond_dim", "bond_dim_needed",
            "n_mps_qubits", "trials", "maxiter", "compile_seconds", "mean_step_seconds",
            "total_seconds", "peak_gpu_gib", "n_nonfinite_steps", "final_sigma_mean",
            "training_sigma_mean", "final_energy_mean", "E0", "final_p_reset_mean",
            "n_converged", "steps_to_converge_mean", "seconds_to_converge_mean"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["n_qubits"], list(METHODS).index(r["method"]))):
            w.writerow(r)


def size_axis(ax, rows):
    sizes = sorted({(r["n_qubits"], r["Lx"], r["Ly"]) for r in rows})
    # Log axis: sizes run from 4 to 238 qubits, and on a linear axis the small
    # lattices pile up on top of each other.
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([s[0] for s in sizes])
    ax.set_xticklabels([size_label(s[1], s[2]) for s in sizes], rotation=30, ha="right")
    ax.set_xlabel("System size (qubits)")


def plot_cost_vs_size(rows, out_dir, plt):
    from matplotlib.colors import Normalize
    sig = [r["final_sigma_mean"] for r in rows if r.get("final_sigma_mean") is not None]
    norm = Normalize(vmin=0.0, vmax=max(sig) if sig and max(sig) > 0 else 1.0)
    cmap = plt.get_cmap("viridis")

    for key, ylabel, fname in COST_METRICS:
        have = [r for r in rows if r.get(key) is not None]
        if not have:
            print(f"[exp1] no data for {key}; skipping {fname}")
            continue
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for m, (label, colour) in METHODS.items():
            ok = sorted([r for r in have if r["method"] == m and r["status"] == "ok"],
                        key=lambda r: r["n_qubits"])
            if ok:
                ax.plot([r["n_qubits"] for r in ok], [r[key] for r in ok],
                        color=colour, lw=1.2, label=label, zorder=1)
                ax.scatter([r["n_qubits"] for r in ok], [r[key] for r in ok],
                           c=[r["final_sigma_mean"] for r in ok], cmap=cmap, norm=norm,
                           marker=MARKERS[m], s=55, edgecolors=colour, linewidths=1.2,
                           zorder=2)
            # Failed runs: hollow marker at the failing size, plotted at the
            # method's last successful value so the axis stays meaningful.
            failed = [r for r in rows if r["method"] == m and r["status"] != "ok"]
            for r in failed:
                prev = [o for o in ok if o["n_qubits"] < r["n_qubits"]]
                if prev:
                    ax.scatter([r["n_qubits"]], [prev[-1][key]], marker=MARKERS[m], s=55,
                               facecolors="none", edgecolors=colour, linewidths=1.2)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        size_axis(ax, rows)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label=r"Mean $\sigma[\hat E_n]$")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close(fig)


def plot_cost_vs_accuracy(rows, out_dir, plt):
    ok = [r for r in rows if r["status"] == "ok"]
    lattices = sorted({(r["n_qubits"], r["Lx"], r["Ly"]) for r in ok})
    cmap = plt.get_cmap("plasma", max(len(lattices), 2))
    for key, ylabel, fname in [("mean_step_seconds", "Time per step (s)", "cost_vs_accuracy.png"),
                               ("peak_gpu_gib", "Peak GPU memory (GiB)", "cost_vs_accuracy_gpu.png")]:
        if not any(r.get(key) is not None for r in ok):
            print(f"[exp1] no data for {key}; skipping {fname}")
            continue
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for li, (nq, Lx, Ly) in enumerate(lattices):
            pts = [r for r in ok if r["n_qubits"] == nq and r.get(key) is not None]
            pts.sort(key=lambda r: list(METHODS).index(r["method"]))
            if not pts:
                continue
            ax.plot([r["final_sigma_mean"] for r in pts], [r[key] for r in pts],
                    color=cmap(li), lw=1, label=size_label(Lx, Ly))
            for r in pts:
                ax.scatter(r["final_sigma_mean"], r[key], color=cmap(li),
                           marker=MARKERS[r["method"]], s=45, zorder=2)
        for m, (label, _) in METHODS.items():
            ax.scatter([], [], marker=MARKERS[m], color="gray", label=label)
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_yscale("log")
        ax.set_xlabel(r"Mean $\sigma[\hat E_n]$ (final parameters)")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close(fig)


def plot_convergence(rows, out_dir, plt):
    for key, ylabel, fname, scale in [
        ("steps_to_converge_per_trial", "Steps to converge", "convergence_steps_vs_size.png", 1.0),
        ("seconds_to_converge_per_trial", "Time to converge (h)", "convergence_time_vs_size.png", 1 / 3600),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        any_data = False
        for i, (m, (label, colour)) in enumerate(METHODS.items()):
            pts = []
            for r in sorted([r for r in rows if r["method"] == m and r["status"] == "ok"],
                            key=lambda r: r["n_qubits"]):
                vals = [v * scale for v in (r.get(key) or []) if v is not None]
                if vals:
                    pts.append((r["n_qubits"], np.mean(vals), min(vals), max(vals)))
            if not pts:
                continue
            any_data = True
            x = np.array([p[0] for p in pts]) + (i - 1.5) * 0.12   # small dodge
            y = np.array([p[1] for p in pts])
            err = np.array([[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]])
            ax.errorbar(x, y, yerr=err, color=colour, marker=MARKERS[m], capsize=3,
                        lw=1.2, label=label)
        if not any_data:
            plt.close(fig)
            print(f"[exp1] no converged trials; skipping {fname}")
            continue
        ax.set_ylabel(ylabel)
        size_axis(ax, rows)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close(fig)


def plot_training_curves(rows, out_dir, plt):
    by_lattice = {}
    for r in rows:
        if r["status"] == "ok":
            by_lattice.setdefault((r["n_qubits"], r["Lx"], r["Ly"]), []).append(r)
    for (nq, Lx, Ly), rs in sorted(by_lattice.items()):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for r in sorted(rs, key=lambda r: list(METHODS).index(r["method"])):
            label, colour = METHODS[r["method"]]
            data = np.genfromtxt(os.path.join(r["run_dir"], "metrics.csv"), delimiter=",",
                                 names=True)
            trials = np.unique(data["trial"]).astype(int)
            curves = []
            for t in trials:
                sel = data["trial"] == t
                ax.plot(data["step"][sel], data["exact_energy"][sel], color=colour,
                        alpha=0.25, lw=0.8)
                curves.append(data["exact_energy"][sel])
            steps = data["step"][data["trial"] == trials[0]]
            # Purified and enumerated are both exact and share initial
            # parameters, so their curves coincide; dash purified so both show.
            ax.plot(steps, np.mean(curves, axis=0), color=colour, lw=2, label=label,
                    ls="--" if r["method"] == "purified" else "-", zorder=3)
        ax.axhline(exact_ground_energy(Lx, Ly), color="k", ls="--", lw=1)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Energy")
        ax.set_title(size_label(Lx, Ly))
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"training_curves_{Lx}x{Ly}.png"), dpi=200)
        plt.close(fig)


# ----------------------------------------------------------------------------
# Overlays across experiments (e.g. Experiment 1.1 Lx x 3 against Experiment 1
# Lx x 2): colour = method, line style = experiment.
# ----------------------------------------------------------------------------
OVERLAY_STYLES = ["-", "--", ":", "-."]


def default_label(rows, exp_dir):
    """'L×3' when every run in the folder has the same Ly, else the folder name."""
    lys = {r["Ly"] for r in rows}
    return f"L×{lys.pop()}" if len(lys) == 1 else os.path.basename(os.path.normpath(exp_dir))


def parse_reference(spec):
    """'DIR' or 'DIR:LABEL' -> (DIR, LABEL or None).

    The text after the last ':' is a label only if what precedes it is an
    existing directory, so a path that itself contains ':' is kept whole.
    """
    if ":" in spec:
        d, label = spec.rsplit(":", 1)
        if os.path.isdir(d):
            return d, (label or None)
    return spec, None


def overlay_size_axis(ax):
    # Plain qubit counts: lattice labels from two experiments would collide
    # (5x3 and 8x2 are both 22 qubits).
    from matplotlib.ticker import LogLocator, NullLocator, ScalarFormatter
    ax.set_xscale("log")
    # 1-2-5 ticks over whatever range the data spans (e.g. 4 to 238 qubits)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("System size (qubits)")


def plot_overlay(groups, out_dir, plt):
    """groups: list of (label, rows), the first being the main experiment."""
    for key, ylabel, fname in COST_METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        drawn = False
        for gi, (glabel, rows) in enumerate(groups):
            ls = OVERLAY_STYLES[gi % len(OVERLAY_STYLES)]
            for m, (label, colour) in METHODS.items():
                ok = sorted([r for r in rows if r["method"] == m and r["status"] == "ok"
                             and r.get(key) is not None], key=lambda r: r["n_qubits"])
                if ok:
                    drawn = True
                    ax.plot([r["n_qubits"] for r in ok], [r[key] for r in ok], color=colour,
                            ls=ls, marker=MARKERS[m], ms=5, lw=1.2, label=f"{label} ({glabel})")
                # failed runs: hollow marker at the method's last successful value
                for r in [r for r in rows if r["method"] == m and r["status"] != "ok"]:
                    prev = [o for o in ok if o["n_qubits"] < r["n_qubits"]]
                    if prev:
                        ax.scatter([r["n_qubits"]], [prev[-1][key]], marker=MARKERS[m], s=40,
                                   facecolors="none", edgecolors=colour, linewidths=1.2)
        if not drawn:
            plt.close(fig)
            print(f"[exp1] no data for {key}; skipping overlay_{fname}")
            continue
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        overlay_size_axis(ax)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"overlay_{fname}"), dpi=200)
        plt.close(fig)

    for key, ylabel, fname, scale in [
        ("steps_to_converge_per_trial", "Steps to converge", "convergence_steps_vs_size.png", 1.0),
        ("seconds_to_converge_per_trial", "Time to converge (h)", "convergence_time_vs_size.png", 1 / 3600),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        drawn = False
        for gi, (glabel, rows) in enumerate(groups):
            ls = OVERLAY_STYLES[gi % len(OVERLAY_STYLES)]
            for m, (label, colour) in METHODS.items():
                pts = []
                for r in sorted([r for r in rows if r["method"] == m and r["status"] == "ok"],
                                key=lambda r: r["n_qubits"]):
                    vals = [v * scale for v in (r.get(key) or []) if v is not None]
                    if vals:
                        pts.append((r["n_qubits"], np.mean(vals)))
                if pts:
                    drawn = True
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, ls=ls,
                            marker=MARKERS[m], ms=5, lw=1.2, label=f"{label} ({glabel})")
        if not drawn:
            plt.close(fig)
            print(f"[exp1] no converged trials; skipping overlay_{fname}")
            continue
        ax.set_ylabel(ylabel)
        overlay_size_axis(ax)
        ax.grid(True, ls="--", alpha=0.5)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"overlay_{fname}"), dpi=200)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp-dir", default=EXP_DIR)
    ap.add_argument("--label", default=None,
                    help="legend label for --exp-dir in the overlay plots (default L×<Ly>)")
    ap.add_argument("--reference-exp-dir", action="append", default=[], metavar="DIR[:LABEL]",
                    help="another experiment folder to overlay on the cost-vs-size and "
                         "convergence plots (overlay_*.png); repeatable. Its runs are drawn "
                         "only, never written to summary.csv")
    ap.add_argument("--out-dir", default=None,
                    help="where to write the plots (default <exp-dir>/comparisons)")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = collect(args.exp_dir)
    if not rows:
        raise SystemExit(f"no summary.json files under {args.exp_dir}/<lattice>/<method>/")
    out_dir = args.out_dir or os.path.join(args.exp_dir, "comparisons")
    os.makedirs(out_dir, exist_ok=True)
    write_summary_csv(rows, os.path.join(out_dir, "summary.csv"))
    plot_cost_vs_size(rows, out_dir, plt)
    plot_cost_vs_accuracy(rows, out_dir, plt)
    plot_convergence(rows, out_dir, plt)
    plot_training_curves(rows, out_dir, plt)

    if args.reference_exp_dir:
        groups = [(args.label or default_label(rows, args.exp_dir), rows)]
        for spec in args.reference_exp_dir:
            ref_dir, label = parse_reference(spec)
            ref_rows = collect(ref_dir)
            if not ref_rows:
                raise SystemExit(f"no summary.json files under {ref_dir}/<lattice>/<method>/")
            groups.append((label or default_label(ref_rows, ref_dir), ref_rows))
        labels = [g[0] for g in groups]
        if len(set(labels)) != len(labels):
            raise SystemExit(f"overlay labels must differ, got {labels}; use DIR:LABEL / --label")
        plot_overlay(groups, out_dir, plt)
        print(f"[exp1] overlays: {', '.join(labels)}")
    print(f"[exp1] wrote {out_dir}/ from {len(rows)} runs")


if __name__ == "__main__":
    main()
