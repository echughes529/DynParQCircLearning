"""Figures for the trajectory-vs-purified investigation.

Run:
    python -m src.diagnostics.plot_traj_vs_purified --indir outputs/traj_vs_purified

Consumes the CSVs written by bond_dim_requirement.py, traj_vs_purified_bench.py,
traj_estimator_stats.py and score_saved_params.py, and writes PNGs to
docs/figures/. Every input is optional -- whichever CSVs are present produce
their figures and the rest are skipped, so this can be run while jobs are still
in flight.
"""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two series, one per reset implementation. Validated as a categorical pair
# (CVD deltaE 24.7 protan, normal-vision 33.6, both >= 3:1 on a light surface),
# so the two arms stay distinguishable in print and for colorblind readers.
TRAJ = "#2a78d6"
PUR = "#eb6834"
COLOR = {"traj": TRAJ, "pur": PUR}
LABEL = {"traj": "trajectory (ancilla-free)", "pur": "purified (2 ancillas/reset)"}

INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e3e3e0"


def style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def tidy(ax):
    """Recessive frame: the data should be the only prominent thing."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.7, linewidth=0.6)
    ax.set_axisbelow(True)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def read_many(indir, pattern):
    rows = []
    for name in sorted(os.listdir(indir)) if os.path.isdir(indir) else []:
        if pattern in name and name.endswith(".csv"):
            rows.extend(read_csv(os.path.join(indir, name)))
    return rows


def fig_bond_dim(rows, outdir):
    """Discarded weight against bond dimension: the scaling story."""
    if not rows:
        return
    # Only the optimised-ordering measurement here; the natural-ordering control
    # is its own figure, since mixing the two would hide which effect is which.
    rows = [r for r in rows if r.get("use_optimal_ordering", "True") == "True"]
    lattices = sorted({r["lattice"] for r in rows}, key=lambda s: int(s[0]) * int(s[-1]))
    fig, axes = plt.subplots(1, len(lattices), figsize=(3.3 * len(lattices), 3.4), squeeze=False)

    # Exactly-representable settings come back as ~1e-16 or a hair below zero
    # from round-off. They are pinned to this floor and drawn as a hollow marker
    # rather than being allowed to stretch the axis over ten empty decades.
    FLOOR = 1e-11
    handles = {}

    # Solid = the converged ground state, which is what a finished run has to
    # hold; dashed = a random initialisation, which is far more entangled and
    # sets an upper bound nobody actually has to meet.
    STYLE = {"converged": "-", "init": (0, (5, 3))}

    for ax, lat in zip(axes[0], lattices):
        chains = {}
        # Which parameter set carries the annotation for this lattice: the
        # converged state where it was measured, otherwise the initialisation,
        # so every panel names the bond dimension it turns on.
        has_converged = any(r["lattice"] == lat and r["params"] == "converged" for r in rows)
        annotate_set = "converged" if has_converged else "init"
        for mode in ("traj", "pur"):
            for pset in ("converged", "init"):
                sub = sorted((r for r in rows if r["lattice"] == lat and r["mode"] == mode
                              and r["params"] == pset),
                             key=lambda r: int(r["bond_dim"]))
                if not sub:
                    continue
                x = [int(r["bond_dim"]) for r in sub]
                raw = [float(r["discarded_weight"]) for r in sub]
                y = [max(v, FLOOR) for v in raw]
                chains[mode] = sub[0]["chain"]
                line, = ax.plot(x, y, linestyle=STYLE[pset], color=COLOR[mode],
                                linewidth=2.0 if pset == "converged" else 1.5,
                                alpha=1.0 if pset == "converged" else 0.55)
                if pset == "converged":
                    handles[mode] = (line, LABEL[mode])
                    # Solid marker = measured; hollow = pinned to the floor
                    # because truncation was numerically zero.
                    for xi, yi, ri in zip(x, y, raw):
                        ax.plot([xi], [yi], "o", color="white" if ri <= FLOOR else COLOR[mode],
                                markeredgecolor=COLOR[mode], markeredgewidth=1.6,
                                markersize=6, zorder=3)
                elif mode not in handles:
                    handles[mode] = (line, LABEL[mode])

                # The smallest bond dimension clearing the gate is the number the
                # investigation turns on, so it is labelled on the chart.
                if pset == annotate_set:
                    passed = [(xi, yi) for xi, yi, ri in zip(x, y, raw) if ri <= 1e-5]
                    if passed:
                        bx, byy = passed[0]
                        dx, ha = (-9, "right") if mode == "traj" else (9, "left")
                        ax.annotate(f"bd {bx}", (bx, byy), textcoords="offset points",
                                    xytext=(dx, 7), ha=ha, fontsize=8.5,
                                    color=COLOR[mode], fontweight="bold")

        note = "  ·  ".join(f"{m}: {chains[m]} sites" for m in ("traj", "pur") if m in chains)
        ax.text(0.5, 0.965, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color=INK_2)

        ax.axhline(1e-5, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 3, 3)
        # Room at both ends so the bd annotations are not clipped by the frame.
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo * 0.78, hi * 1.45)
        ax.set_title(lat, loc="left")
        ax.set_xlabel("bond dimension")
        tidy(ax)
        ax.text(0.98, 1e-5 * 1.6, "gate", ha="right", va="bottom", fontsize=7.5,
                color=INK_MUTED, transform=ax.get_yaxis_transform())

    axes[0][0].set_ylabel("discarded weight  $1-\\||\\psi\\||^2$")
    ordered = [handles[m] for m in ("traj", "pur") if m in handles]
    style_h = [plt.Line2D([], [], color=INK_MUTED, linestyle=STYLE["converged"], linewidth=2),
               plt.Line2D([], [], color=INK_MUTED, linestyle=STYLE["init"], linewidth=1.5)]
    fig.legend([h for h, _ in ordered] + style_h,
               [t for _, t in ordered] + ["converged ground state", "random initialisation"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Bond dimension each reset path needs to represent the state",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    path = os.path.join(outdir, "bond_dim_requirement.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_ordering(rows, outdir):
    """The control that says where the bond-dimension gap actually comes from.

    Optimised ordering vs natural ordering, at the converged ground state. If the
    two paths need the same bond dimension once they share a qubit ordering, the
    gap was never about ancilla entanglement.
    """
    rows = [r for r in rows if r.get("params") == "converged" and r["lattice"] == "3x3"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    FLOOR = 1e-11

    for ax, ordering, title in ((axes[0], "True", "Optimised ordering"),
                                (axes[1], "False", "Natural ordering (control)")):
        for mode in ("traj", "pur"):
            sub = sorted((r for r in rows if r["mode"] == mode
                          and r["use_optimal_ordering"] == ordering),
                         key=lambda r: int(r["bond_dim"]))
            if not sub:
                continue
            x = [int(r["bond_dim"]) for r in sub]
            raw = [float(r["discarded_weight"]) for r in sub]
            y = [max(v, FLOOR) for v in raw]
            ax.plot(x, y, "-o", color=COLOR[mode], label=LABEL[mode],
                    markeredgecolor="white", markeredgewidth=0.8)
            passed = [(xi, yi) for xi, yi, ri in zip(x, y, raw) if ri <= 1e-5]
            if passed:
                bx, byy = passed[0]
                dx, ha = (-9, "right") if mode == "traj" else (9, "left")
                ax.annotate(f"bd {bx}", (bx, byy), textcoords="offset points",
                            xytext=(dx, 7), ha=ha, fontsize=8.5,
                            color=COLOR[mode], fontweight="bold")
        ax.axhline(1e-5, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 3, 3)
        # Room at both ends so the bd annotations are not clipped by the frame.
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo * 0.72, hi * 1.25)
        ax.set_title(title, loc="left")
        ax.set_xlabel("bond dimension")
        tidy(ax)
    axes[0].set_ylabel("discarded weight  $1-\\||\\psi\\||^2$")
    # Placed high, where the curves have already fallen away, rather than at the
    # floor where every series overlaps.
    axes[0].text(0.5, 0.80, "max gate span:  traj 3  ·  pur 6", transform=axes[0].transAxes,
                 ha="center", fontsize=8.5, color=INK_2)
    axes[1].text(0.5, 0.80, "max gate span:  both 6", transform=axes[1].transAxes,
                 ha="center", fontsize=8.5, color=INK_2)
    axes[1].legend(loc="upper right")
    fig.suptitle("Where the bond-dimension gap comes from (3×3, converged state)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(outdir, "ordering_control.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_estimator(rows, outdir):
    """Gradient direction quality and how it scales with n_trajectories."""
    if not rows:
        return
    lattices = sorted({r["lattice"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    base = [r for r in rows if r["baseline"] == "exact_energy"]

    # Panel 1: direction quality along training, at n_traj = 1.
    ax = axes[0]
    for lat in lattices:
        sub = sorted((r for r in base if r["lattice"] == lat and int(r["n_trajectories"]) == 1),
                     key=lambda r: int(r["step"]))
        if not sub:
            continue
        x = [int(r["step"]) for r in sub]
        y = [float(r["cos_mean"]) for r in sub]
        lo = [float(r["cos_p10"]) for r in sub]
        ax.plot(x, y, "-o", color=TRAJ, markeredgecolor="white", markeredgewidth=0.8,
                label=f"{lat} mean")
        ax.fill_between(x, lo, y, color=TRAJ, alpha=0.12, linewidth=0)
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.set_title("Sampled gradient vs exact direction", loc="left")
    ax.set_xlabel("training step")
    ax.set_ylabel("cosine similarity")
    ax.legend(loc="lower left")
    tidy(ax)

    # Panel 2: variance reduction against the ideal 1/sqrt(n).
    ax = axes[1]
    for lat in lattices:
        steps = sorted({int(r["step"]) for r in base if r["lattice"] == lat})
        if not steps:
            continue
        last = max(steps)
        sub = sorted((r for r in base if r["lattice"] == lat and int(r["step"]) == last),
                     key=lambda r: int(r["n_trajectories"]))
        n = [int(r["n_trajectories"]) for r in sub]
        noise = np.array([float(r["grad_noise_norm"]) for r in sub])
        if len(noise) == 0:
            continue
        ax.plot(n, noise / noise[0], "-o", color=TRAJ, markeredgecolor="white",
                markeredgewidth=0.8, label=f"{lat} measured (step {last})")
    ideal_n = np.array([1, 2, 4, 8, 16])
    ax.plot(ideal_n, 1 / np.sqrt(ideal_n), "--", color=INK_MUTED, linewidth=1.4,
            label="ideal $1/\\sqrt{n}$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Noise reduction per trajectory sampled", loc="left")
    ax.set_xlabel("trajectories per step")
    ax.set_ylabel("gradient noise, relative to n=1")
    ax.legend(loc="lower left")
    tidy(ax)

    # Panel 3: the control variate is load-bearing or it is not.
    ax = axes[2]
    lat = lattices[0]
    width = 0.36
    steps = sorted({int(r["step"]) for r in rows if r["lattice"] == lat})
    for offset, (blabel, bkey, color) in enumerate(
            [("with baseline", "exact_energy", TRAJ), ("baseline = 0", "zero", PUR)]):
        vals = []
        for s in steps:
            m = [r for r in rows if r["lattice"] == lat and int(r["step"]) == s
                 and r["baseline"] == bkey and int(r["n_trajectories"]) == 1]
            vals.append(float(m[0]["snr_reset_median"]) if m else np.nan)
        ax.bar(np.arange(len(steps)) + (offset - 0.5) * width, vals, width * 0.92,
               color=color, label=blabel, edgecolor="white", linewidth=1.5)
    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_title(f"Reset-theta gradient SNR ({lat})", loc="left")
    ax.set_xlabel("training step")
    ax.set_ylabel("median SNR")
    ax.legend(loc="upper right")
    tidy(ax)

    fig.tight_layout()
    path = os.path.join(outdir, "estimator_quality.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_accuracy(rows, outdir, step_seconds):
    """Error against E0, on both a step axis and a wall-clock axis."""
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[(r["lattice"], r["label"])].append(r)

    lattices = sorted({k[0] for k in by})
    fig, axes = plt.subplots(2, len(lattices), figsize=(3.6 * len(lattices), 6), squeeze=False)

    for col, lat in enumerate(lattices):
        for (l, label), rs in sorted(by.items()):
            if l != lat:
                continue
            mode = "traj" if "traj" in label else "pur"
            steps = sorted({int(r["step"]) for r in rs})
            # optimize() freezes a trial whose gradient goes non-finite but still
            # records its NaN, so filter before taking the best-trial minimum --
            # a stray NaN would otherwise propagate through min().
            best, kept_steps = [], []
            for s in steps:
                vals = [float(r["rel_error"]) for r in rs if int(r["step"]) == s
                        and np.isfinite(float(r["rel_error"]))]
                if vals:
                    kept_steps.append(s)
                    best.append(max(min(vals), 1e-12))
            steps = kept_steps
            secs = step_seconds.get(label)
            for row, xs, xlabel in ((0, steps, "optimisation step"),
                                    (1, [s * secs for s in steps] if secs else None,
                                     "cumulative wall-clock (s)")):
                if xs is None:
                    continue
                ax = axes[row][col]
                ax.plot(xs, best, "-", color=COLOR[mode], label=LABEL[mode])
                ax.set_yscale("log")
                ax.set_xlabel(xlabel)
                ax.set_title(lat if row == 0 else "", loc="left")
                tidy(ax)
    for row in (0, 1):
        axes[row][0].set_ylabel("relative error to $E_0$ (best trial)")
    axes[0][-1].legend(loc="upper right")
    fig.suptitle("Accuracy against exact ground-state energy", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(outdir, "accuracy_per_walltime.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_cost(rows, outdir):
    """Per-step cost, compared only where a comparison is actually valid.

    Benchmark runs used different trial counts per lattice, and vmapped trials do
    not cost the same each, so raw seconds are not comparable across lattices.
    Everything here is per trial, and the paired bars are only drawn where both
    paths were measured at the SAME bond dimension in the same process on the
    same device.
    """
    if not rows:
        return
    by = {}
    for r in rows:
        by[(r["lattice"], r["mode"], int(r["bond_dim"]))] = r

    # Keep only bond dimensions where both arms were measured.
    pairs = []
    for (lat, mode, bd), r in by.items():
        if mode != "traj":
            continue
        other = by.get((lat, "pur", bd))
        if other is not None:
            pairs.append((lat, bd, r, other))
    if not pairs:
        return
    pairs.sort(key=lambda p: (int(p[0][0]) * int(p[0][-1]), p[1]))

    labels = [f"{lat}\nbd {bd}" for lat, bd, _, _ in pairs]
    x = np.arange(len(pairs))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    ax = axes[0]
    t_vals = [float(t["warm_vg_per_trial_seconds"]) for _, _, t, _ in pairs]
    p_vals = [float(p["warm_vg_per_trial_seconds"]) for _, _, _, p in pairs]
    ax.bar(x - width / 2, t_vals, width * 0.92, color=TRAJ, label=LABEL["traj"],
           edgecolor="white", linewidth=1.5)
    ax.bar(x + width / 2, p_vals, width * 0.92, color=PUR, label=LABEL["pur"],
           edgecolor="white", linewidth=1.5)
    for xi, tv, pv in zip(x, t_vals, p_vals):
        # The ratio is the number the reader wants; put it where the eye already is.
        ax.annotate(f"{pv / tv:.1f}×", (xi, max(tv, pv)), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(top=max(p_vals) * 3.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title("Cost of one optimisation step, per trial", loc="left")
    ax.set_ylabel("seconds (log scale)")
    ax.legend(loc="upper left")
    tidy(ax)

    ax = axes[1]
    t_svd = [int(t["svd_calls"]) for _, _, t, _ in pairs]
    p_svd = [int(p["svd_calls"]) for _, _, _, p in pairs]
    ax.bar(x - width / 2, t_svd, width * 0.92, color=TRAJ, edgecolor="white", linewidth=1.5)
    ax.bar(x + width / 2, p_svd, width * 0.92, color=PUR, edgecolor="white", linewidth=1.5)
    for xi, tv, pv in zip(x, t_svd, p_svd):
        ax.annotate(str(tv), (xi - width / 2, tv), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=INK_2)
        ax.annotate(str(pv), (xi + width / 2, pv), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(top=max(p_svd) * 1.18)
    ax.set_title("SVD calls per circuit build", loc="left")
    ax.set_ylabel("calls")
    tidy(ax)

    fig.suptitle("What the trajectory path saves, at matched bond dimension",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(outdir, "cost_anatomy.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="outputs/traj_vs_purified")
    parser.add_argument("--outdir", default="docs/figures")
    parser.add_argument("--step-seconds", default="",
                        help="comma-free spec label=seconds_per_step;label=seconds_per_step, "
                             "used to put the accuracy plot on a wall-clock axis")
    args = parser.parse_args()

    style()
    os.makedirs(args.outdir, exist_ok=True)

    step_seconds = {}
    for item in filter(None, args.step_seconds.split(";")):
        k, v = item.split("=")
        step_seconds[k] = float(v)

    print("figures:")
    bd_rows = read_many(args.indir, "bond_dim_requirement")
    fig_bond_dim(bd_rows, args.outdir)
    fig_ordering(bd_rows, args.outdir)
    fig_estimator(read_many(args.indir, "estimator_stats"), args.outdir)
    fig_cost(read_many(args.indir, "bench"), args.outdir)
    fig_accuracy(read_many(args.indir, "scored"), args.outdir, step_seconds)


if __name__ == "__main__":
    main()
