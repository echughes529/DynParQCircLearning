"""Figures for the three-way reset-implementation comparison.

Run:
    python -m src.diagnostics.plot_three_resets --indir outputs/three_resets

Consumes the CSVs written by bond_dim_requirement.py, reset_impl_bench.py,
collect_runs.py, score_saved_params.py and traj_estimator_stats.py, and writes
PNGs to docs/figures/. Every input is optional -- whichever CSVs are present
produce their figures and the rest are skipped, so this can be run while jobs
are still in flight.

Supersedes plot_traj_vs_purified.py, which knows only two arms.
"""

import argparse
import csv
import re
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One series per reset implementation. Validated as a 3-slot categorical palette
# under the strictest all-pairs rule on a light surface: worst CVD separation
# deltaE 11.3 (deutan) and 13.3 (tritan), worst normal-vision 22.5, all three
# above 3:1 contrast. The first two are unchanged from the two-arm figures so
# the old and new documents stay visually consistent; magenta is the addition,
# chosen because green and purple both collapse -- green against orange under
# protan, purple against blue under deutan.
TRAJ = "#2a78d6"
PUR = "#eb6834"
ENUM = "#b5379a"
COLOR = {"traj": TRAJ, "pur": PUR, "enum": ENUM}
LABEL = {
    "traj": "single trajectory (ancilla-free, sampled)",
    "pur": "purified (2 ancillas per reset)",
    "enum": "parallel trajectories (all $3^R$ branches)",
}
SHORT = {"traj": "trajectory", "pur": "purified", "enum": "parallel"}
# Every arm also gets a marker, so identity never rests on colour alone.
MARKER = {"traj": "o", "pur": "s", "enum": "D"}
ORDER = ("traj", "pur", "enum")

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


def save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def read_many(indir, pattern):
    rows = []
    if not os.path.isdir(indir):
        return rows
    for name in sorted(os.listdir(indir)):
        if pattern in name and name.endswith(".csv"):
            rows.extend(read_csv(os.path.join(indir, name)))
    return rows


def num(row, key, default=np.nan):
    """CSV values are strings; empty and 'None' both mean absent."""
    v = row.get(key)
    if v in (None, "", "None", "nan"):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def truthy(row, key):
    return str(row.get(key, "")).strip().lower() in ("true", "1", "yes")


def requested_gib(row):
    """How much the allocator was asked for when it refused, in GiB.

    XLA's RESOURCE_EXHAUSTED message names the allocation it could not satisfy
    ("Out of memory while trying to allocate 68.08GiB"). For a configuration
    that does not fit, that is the only quantitative thing available -- and it
    is the useful one, since it says how far past the card the configuration is
    rather than merely that it is past.
    """
    m = re.search(r"allocate\s+([\d.]+)\s*([KMG])iB", str(row.get("error", "")))
    if not m:
        return np.nan
    return float(m.group(1)) * {"K": 1 / 2 ** 20, "M": 1 / 2 ** 10, "G": 1.0}[m.group(2)]


# ----------------------------------------------------------------------------
# 1. Bond dimension required
# ----------------------------------------------------------------------------
def fig_bond_dim(rows, outdir, gate=1e-5):
    """Discarded weight against bond dimension, one panel per lattice.

    This is the headline: it says what each arm has to be given before its
    answer is right, which is the only fair basis for comparing what each costs.
    """
    rows = [r for r in rows if str(r.get("params", "init")) == "init"
            and str(r.get("use_optimal_ordering", "True")) == "True"]
    if not rows:
        return
    lattices = sorted({r["lattice"] for r in rows}, key=lambda s: [int(v) for v in s.split("x")])
    fig, axes = plt.subplots(1, len(lattices), figsize=(3.6 * len(lattices), 3.4), squeeze=False)

    # A floor so exactly-zero discarded weight (the exact-cap rungs) stays
    # plottable on a log axis instead of vanishing.
    FLOOR = 1e-16
    # The two ancilla-free arms share a chain and therefore truncate identically,
    # so their curves coincide exactly and one would hide the other. Draw the
    # trajectory arm wide and solid underneath and the enumerated arm narrow and
    # dashed on top: where they agree you see both, which is itself the point.
    STYLE = {
        "traj": dict(linewidth=3.2, linestyle="-", alpha=0.85, zorder=2, markersize=6),
        "pur": dict(linewidth=2.0, linestyle="-", zorder=3, markersize=5),
        "enum": dict(linewidth=1.5, linestyle=(0, (4, 2.5)), zorder=4, markersize=5.5,
                     markerfacecolor="white", markeredgewidth=1.6),
    }
    coincide = False
    for ax, lat in zip(axes[0], lattices):
        tidy(ax)
        curves = {}
        for mode in ORDER:
            pts = sorted([r for r in rows if r["lattice"] == lat and r["mode"] == mode],
                         key=lambda r: num(r, "bond_dim"))
            if not pts:
                continue
            x = [num(r, "bond_dim") for r in pts]
            y = [max(num(r, "discarded_weight"), FLOOR) for r in pts]
            curves[mode] = dict(zip(x, y))
            kw = dict(STYLE[mode])
            kw.setdefault("markeredgecolor", "white")
            kw.setdefault("markeredgewidth", 0.8)
            if mode == "enum":
                kw["markeredgecolor"] = COLOR[mode]
            ax.plot(x, y, marker=MARKER[mode], color=COLOR[mode], label=SHORT[mode], **kw)
            # Annotate the first rung that clears the gate: that is the number
            # the training runs are configured from. Staggered per arm so the
            # labels of coincident curves do not print on top of each other.
            passed = next((r for r in pts if num(r, "discarded_weight") <= gate), None)
            if passed is not None:
                bd = int(num(passed, "bond_dim"))
                dy = {"traj": 20, "pur": 8, "enum": 34}[mode]
                ax.annotate(f"{SHORT[mode][:4]} {bd}",
                            (bd, max(num(passed, "discarded_weight"), FLOOR)),
                            textcoords="offset points", xytext=(5, dy),
                            fontsize=7.5, color=COLOR[mode], fontweight="semibold")
        if "traj" in curves and "enum" in curves:
            shared = set(curves["traj"]) & set(curves["enum"])
            if shared and all(np.isclose(curves["traj"][b], curves["enum"][b], rtol=1e-6)
                              for b in shared):
                coincide = True
        ax.axhline(gate, color=INK_MUTED, linestyle="--", linewidth=1.0)
        ax.text(0.02, gate, f" gate {gate:g}", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=7, color=INK_MUTED)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 3, 3)
        ax.set_title(lat)
        ax.set_xlabel("bond dimension")
    axes[0][0].set_ylabel("discarded weight  $1-\\|\\psi\\|^2$")
    axes[0][-1].legend(loc="lower left", fontsize=7.5)
    note = ("trajectory and parallel share the bare $n_q$-site chain, so their curves coincide "
            "exactly (dashed over solid)" if coincide else "")
    fig.suptitle("Bond dimension each reset implementation needs", y=1.04, fontsize=11)
    if note:
        fig.text(0.5, 0.97, note, ha="center", fontsize=8, color=INK_2)
    save(fig, outdir, "bond_dim_requirement.png")


# ----------------------------------------------------------------------------
# 2. Cost anatomy
# ----------------------------------------------------------------------------
def fig_cost(rows, outdir):
    """Warm step cost and its attribution, grouped by arm at matched settings.

    Two panels rather than a dual axis: seconds and SVD counts are different
    quantities and putting them on one pair of axes would invite reading a
    crossing point that does not exist.
    """
    rows = [r for r in rows if not truthy(r, "oom") and r.get("failed_stage") in (None, "", "None")]
    if not rows:
        return
    # Group by (lattice, bond_dim, trials) so only genuinely comparable
    # configurations sit in one cluster.
    groups = defaultdict(dict)
    for r in rows:
        key = (r["lattice"], int(num(r, "bond_dim")), int(num(r, "trials")))
        groups[key][r["mode"]] = r
    keys = sorted(k for k, v in groups.items() if len(v) >= 2)
    if not keys:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    width = 0.26
    xs = np.arange(len(keys))
    for panel, (ax, field, ylabel, title) in enumerate([
        (axes[0], "warm_vg_per_trial_seconds", "seconds per trial per step",
         "Warm value-and-grad cost"),
        (axes[1], "svd_calls_per_step", "SVD calls per step",
         "Where the cost comes from"),
    ]):
        tidy(ax)
        for i, mode in enumerate(ORDER):
            vals = [num(groups[k].get(mode, {}), field) if mode in groups[k] else np.nan
                    for k in keys]
            ax.bar(xs + (i - 1) * width, vals, width * 0.92, color=COLOR[mode],
                   label=SHORT[mode] if panel == 0 else None,
                   edgecolor="white", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lat}\nbd {bd}, {t} trials" for lat, bd, t in keys], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(loc="upper left", ncol=3)
    fig.suptitle("Cost per optimisation step", y=1.03, fontsize=11)
    save(fig, outdir, "cost_anatomy.png")


# ----------------------------------------------------------------------------
# 3. Memory and the feasibility frontier
# ----------------------------------------------------------------------------
def fig_memory(rows, outdir, card_gib=44.4):
    """Peak device memory per arm, and where the enumerated arm stops fitting.

    Left: peak GiB at matched settings. Right: the enumerated arm's peak against
    trial count, with the card's capacity drawn in -- the 3^R branch factor is
    what pushes it over, and this is where it lands.
    """
    ok = [r for r in rows if not truthy(r, "oom") and r.get("failed_stage") in (None, "", "None")]
    if not ok:
        return
    # The right panel carries long row labels on its y-axis, so it needs both
    # more width and a wide gutter or they run back over the left panel's bars.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.0),
                             gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.42))

    # --- left: peak per arm at matched settings ---
    ax = axes[0]
    tidy(ax)
    groups = defaultdict(dict)
    for r in ok:
        groups[(r["lattice"], int(num(r, "bond_dim")), int(num(r, "trials")))][r["mode"]] = r
    keys = sorted(k for k, v in groups.items() if len(v) >= 2)
    xs = np.arange(len(keys))
    width = 0.26
    for i, mode in enumerate(ORDER):
        vals = [num(groups[k].get(mode, {}), "peak_bytes_after_vg") / 2 ** 30
                if mode in groups[k] else np.nan for k in keys]
        bars = ax.bar(xs + (i - 1) * width, vals, width * 0.92, color=COLOR[mode],
                      label=SHORT[mode], edgecolor="white", linewidth=1.0)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.axhline(card_gib, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(0.01, card_gib, " A40 capacity", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=7, color=INK_MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lat}\nbd {bd}, {t} trials" for lat, bd, t in keys], fontsize=7.5)
    ax.set_ylabel("peak device memory (GiB)")
    ax.set_title("Peak memory at matched settings")
    ax.set_ylim(0, card_gib * 1.18)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.20))

    # --- right: the enumerated feasibility frontier ---
    # A run that OOMs has no measured peak, but the allocator says how much it
    # was ASKING for when it gave up. That number is the informative one: it says
    # how far past the card a configuration is, not merely that it is past.
    ax = axes[1]
    tidy(ax)
    enum_rows = [r for r in rows if r["mode"] == "enum"]
    by_cfg = defaultdict(list)
    for r in enum_rows:
        chunk = r.get("branch_chunk_size") or ""
        tag = f"{r['lattice']} bd {int(num(r, 'bond_dim'))}"
        if chunk not in ("", "None"):
            tag += f" chunk {int(float(chunk))}"
        by_cfg[tag].append(r)

    # One row per configuration rather than a line against trials. The amount an
    # OOM'd run was asking for is NOT a smooth function of trial count -- a run
    # that dies compiling the forward pass asks for less than one that got as far
    # as the backward pass -- so joining these points would draw a trend that
    # does not exist. Rows, read directly, cannot imply one.
    entries = []
    for tag, rs in by_cfg.items():
        for r in sorted(rs, key=lambda r: -num(r, "trials")):
            trials = int(num(r, "trials"))
            if truthy(r, "oom"):
                gib, fits, stage = requested_gib(r), False, str(r.get("failed_stage", ""))
            else:
                gib, fits, stage = num(r, "peak_bytes_after_vg") / 2 ** 30, True, ""
            if np.isfinite(gib):
                entries.append((f"{tag}, {trials} trial{'s' if trials != 1 else ''}",
                                gib, fits, stage))
    if not entries:
        return
    entries.sort(key=lambda e: e[1])
    ys = np.arange(len(entries))
    for y, (lab, gib, fits, stage) in zip(ys, entries):
        ax.plot([0.5, gib], [y, y], color=COLOR["enum"], alpha=0.35, linewidth=1.4,
                zorder=1, solid_capstyle="butt")
        if fits:
            ax.plot([gib], [y], marker=MARKER["enum"], color=COLOR["enum"], markersize=8,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3)
            ax.annotate(f"{gib:.1f} GiB", (gib, y), textcoords="offset points",
                        xytext=(9, 0), va="center", fontsize=7, color=INK_2)
        else:
            ax.plot([gib], [y], marker="X", color=COLOR["enum"], markersize=9,
                    markerfacecolor="white", markeredgecolor=COLOR["enum"],
                    markeredgewidth=1.8, zorder=3)
            short_stage = {"compile_forward": "fwd", "compile_value_and_grad": "grad",
                           "process_died": "killed"}.get(stage, stage[:6])
            ax.annotate(f"{gib:.0f} GiB ({short_stage})", (gib, y),
                        textcoords="offset points", xytext=(9, 0), va="center",
                        fontsize=7, color=INK_MUTED)
    ax.axvline(card_gib, color=INK_MUTED, linestyle="--", linewidth=1.0)
    # Named in the title rather than beside the line: every row extends across
    # the capacity line, so an inline label collides with one of them whatever
    # height it is given.
    ax.set_xscale("log")
    ax.set_xlim(0.5, max(g for _, g, _, _ in entries) * 6)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries], fontsize=7.5)
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.set_xlabel("device memory (GiB, log scale)")
    ax.set_title(f"Diamond = measured peak;  X = allocation refused\n"
                 f"dashed line = A40 capacity ({card_gib:.0f} GiB)", fontsize=9.5)
    ax.grid(True, axis="x", alpha=0.7, linewidth=0.6)
    ax.grid(False, axis="y")
    fig.suptitle("Memory: the enumerated arm's binding constraint", y=1.03, fontsize=11)
    save(fig, outdir, "memory_frontier.png")


# ----------------------------------------------------------------------------
# 4. Accuracy against wall clock
# ----------------------------------------------------------------------------
def fig_accuracy(scored, runs, outdir):
    """Relative error against step, and against cumulative wall clock.

    Both panels use the common-yardstick energies from score_saved_params, never
    the energies each arm reported to itself -- those are not comparable across
    arms. The right panel is the one that decides which arm to use: a cheaper
    step is only worth having if it buys accuracy per second.
    """
    if not scored:
        return
    # seconds per step per arm, from the real training runs
    step_seconds = {}
    for r in runs:
        mode = r.get("mode")
        s = num(r, "warm_step_s")
        if mode in COLOR and np.isfinite(s):
            step_seconds.setdefault(mode, []).append(s)
    step_seconds = {m: float(np.median(v)) for m, v in step_seconds.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    by_arm_seed = defaultdict(lambda: defaultdict(list))
    for r in scored:
        mode = r.get("scored_mode") or ""
        if mode not in COLOR:
            continue
        by_arm_seed[mode][r.get("seed", "")].append(r)

    drew_wall = False
    for mode in ORDER:
        if mode not in by_arm_seed:
            continue
        # Median across seeds at each step: a single seed's curve is noise.
        per_step = defaultdict(list)
        for seed, rs in by_arm_seed[mode].items():
            best_at_step = defaultdict(list)
            for r in rs:
                best_at_step[num(r, "step")].append(num(r, "rel_error"))
            for step, errs in best_at_step.items():
                per_step[step].append(np.nanmin(errs))   # best trial of that seed
        steps = np.array(sorted(per_step))
        med = np.array([np.nanmedian(per_step[s]) for s in steps])
        lo = np.array([np.nanmin(per_step[s]) for s in steps])
        hi = np.array([np.nanmax(per_step[s]) for s in steps])

        axes[0].plot(steps, med, color=COLOR[mode], label=SHORT[mode])
        axes[0].fill_between(steps, lo, hi, color=COLOR[mode], alpha=0.15, linewidth=0)
        secs = step_seconds.get(mode)
        if secs:
            axes[1].plot(steps * secs / 3600.0, med, color=COLOR[mode], label=SHORT[mode])
            axes[1].fill_between(steps * secs / 3600.0, lo, hi, color=COLOR[mode],
                                 alpha=0.15, linewidth=0)
            drew_wall = True

    for ax, xlabel, title in ((axes[0], "optimisation step", "Accuracy per step"),
                              (axes[1], "wall clock (hours)", "Accuracy per hour of GPU")):
        tidy(ax)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
    axes[0].set_ylabel("relative error vs $E_0$  (common yardstick)")
    axes[0].axhline(1e-3, color=INK_MUTED, linestyle="--", linewidth=1.0)
    axes[0].text(0.02, 1e-3, " 0.1% convergence criterion",
                 transform=axes[0].get_yaxis_transform(), va="bottom",
                 fontsize=7, color=INK_MUTED)
    axes[0].legend(loc="upper right")
    if not drew_wall:
        axes[1].text(0.5, 0.5, "no per-step timings available", ha="center",
                     va="center", transform=axes[1].transAxes, color=INK_MUTED, fontsize=9)
    fig.suptitle("Band spans seeds; line is the median of each seed's best trial",
                 y=1.03, fontsize=9, color=INK_2)
    save(fig, outdir, "accuracy_per_walltime.png")


# ----------------------------------------------------------------------------
# 5. Convergence by seed
# ----------------------------------------------------------------------------
def fig_convergence(scored, outdir, threshold=1e-3):
    """Final relative error of every trial, split by arm and seed.

    Plotted as individual trials rather than a mean: what matters for a
    multi-restart optimiser is how many restarts land, and a mean over trials
    that mostly failed hides exactly that.
    """
    if not scored:
        return
    final = defaultdict(list)
    for r in scored:
        mode = r.get("scored_mode") or ""
        if mode not in COLOR:
            continue
        final[(mode, str(r.get("seed", "")))].append(r)
    if not final:
        return
    # keep only the last snapshot of each (arm, seed)
    points = {}
    for key, rs in final.items():
        last = max(num(r, "step") for r in rs)
        points[key] = [num(r, "rel_error") for r in rs if num(r, "step") == last]

    seeds = sorted({s for _, s in points})
    fig, ax = plt.subplots(figsize=(1.9 + 1.5 * max(len(seeds), 1), 3.6))
    tidy(ax)
    rng = np.random.default_rng(0)
    xticks, xlabels = [], []
    pos = 0
    for seed in seeds:
        for mode in ORDER:
            vals = points.get((mode, seed))
            if not vals:
                continue
            jitter = rng.uniform(-0.10, 0.10, size=len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=26,
                       color=COLOR[mode], alpha=0.75, edgecolor="white", linewidth=0.7,
                       marker=MARKER[mode],
                       label=SHORT[mode] if seed == seeds[0] else None)
            n_ok = int(np.sum(np.asarray(vals) <= threshold))
            ax.annotate(f"{n_ok}/{len(vals)}", (pos, min(vals)), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=7, color=COLOR[mode])
            xticks.append(pos)
            xlabels.append(SHORT[mode])
            pos += 1
        pos += 0.7
    ax.axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(0.01, threshold, " 0.1%", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=7, color=INK_MUTED)
    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("final relative error vs $E_0$")
    ax.set_title("Every trial's final accuracy, by arm and seed\n"
                 "(annotation: trials within 0.1%)", fontsize=9.5)
    ax.legend(loc="upper right", ncol=3)
    save(fig, outdir, "convergence_by_seed.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="outputs/three_resets")
    parser.add_argument("--outdir", default="docs/figures")
    args = parser.parse_args()

    style()
    bd_rows = read_many(args.indir, "bond_dim_requirement")
    bench_rows = read_many(args.indir, "bench") + read_many(args.indir, "probe")
    run_rows = read_many(args.indir, "runs")
    scored_rows = read_many(args.indir, "scored")

    print(f"inputs from {args.indir}: {len(bd_rows)} bond-dim rows, "
          f"{len(bench_rows)} bench rows, {len(run_rows)} run rows, "
          f"{len(scored_rows)} scored rows")

    fig_bond_dim(bd_rows, args.outdir)
    fig_cost(bench_rows, args.outdir)
    fig_memory(bench_rows, args.outdir)
    fig_accuracy(scored_rows, run_rows, args.outdir)
    fig_convergence(scored_rows, args.outdir)


if __name__ == "__main__":
    main()
