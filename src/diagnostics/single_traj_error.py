"""How much error does the single-trajectory approximation cause during training?

Run:
    python -m src.diagnostics.single_traj_error --lattice 2x2
    python -m src.diagnostics.single_traj_error --lattice 3x2 --steps 800 --seeds 5
    python -m src.diagnostics.single_traj_error --plot-only outputs/single_traj_error/2x2

Two optimisations start from the same parameters:

  run A  trains on the true energy E = sum_i p^(i) E^(i), evaluated over every
         trajectory i (use_enumerated_resets=True);
  run B  trains with the production single-trajectory estimator
         (use_trajectory_resets=True, n_trajectories=1). Each of its trials is the
         same starting point with its own trajectory draws, so the trials are
         independent repeats of one single-trajectory run.

At every step the full trajectory table {p^(i), E^(i)} is evaluated (forward only)
at the current parameters of both runs. From it:

  - the true energy E, so run B's true energy is known even though it only ever
    saw one trajectory per step;
  - the standard error of a single trajectory, SE_1 = sqrt(sum_i p^(i) (E^(i) - E)^2),
    which is the typical size of the per-step estimate error |E_hat - E|;
  - the path error E(run B) - E(run A): what the noisy updates cost in where the
    optimisation actually goes.

Both arms are built at a bond dimension where truncation is provably zero
(traj_estimator_stats.EXACT_BD), so every number is exact. Gradients are not
analysed here; traj_estimator_stats.py covers gradient noise at fixed points.
"""

import argparse
import os
import sys
import time
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.traj_estimator_stats import EXACT_BD, NLAYERS, RESET_LAYERS

K = tc.set_backend("jax")

E0 = {"2x2": -5.0, "3x2": -8.0, "3x3": -13.0}
# (bond_dim, use_optimal_ordering) per arm. 2x2/3x2 use the provably lossless
# figures with natural ordering; at 3x3 optimal ordering makes bd=64 lossless,
# which the recorded sum_i p^(i) (the "norm" column) checks at every step.
CONFIG = {lat: {"enum": (bd["enum"], False), "traj": (bd["traj"], False)}
          for lat, bd in EXACT_BD.items()}
CONFIG["3x3"] = {"enum": (64, True), "traj": (64, True)}
# Below this a trajectory has no state to speak of (note section 3.3.3), so its
# energy is left undefined rather than formed as 0/0.
P_FLOOR = 1e-12


def build_arms(lattice, n_seeds, init_seed, lr):
    Lx, Ly = (int(v) for v in lattice.split("x"))
    common = dict(Lx=Lx, Ly=Ly, nlayers=NLAYERS, reset_layers=RESET_LAYERS,
                  seed=init_seed, normalize_state=True,
                  sparse=False, use_mps=True, learning_rate=lr)
    (bd_e, ord_e), (bd_t, ord_t) = CONFIG[lattice]["enum"], CONFIG[lattice]["traj"]
    exact = ToricCodeAnsatz(use_trajectory_resets=False, use_enumerated_resets=True,
                            max_reset_branches=3 ** 8, trials=1, bond_dim=bd_e,
                            use_optimal_ordering=ord_e, **common)
    single = ToricCodeAnsatz(use_trajectory_resets=True, n_trajectories=1, trials=n_seeds,
                             bond_dim=bd_t, use_optimal_ordering=ord_t, **common)
    assert exact.nparams == single.nparams
    return exact, single


def run(args):
    exact, single = build_arms(args.lattice, args.seeds, args.init_seed, args.lr)
    R = exact.total_resets
    branches = np.asarray(exact._reset_branches)
    nb = len(branches)
    reset_idx = np.asarray(exact.reset_param_indices)
    print(f"{args.lattice}: R={R} resets -> {nb} trajectories, nparams={exact.nparams}, "
          f"chain bd={exact.bond_dim} (exact), run B repeats={args.seeds}")

    p0 = jnp.asarray(exact.initparams[0])
    params_a = p0[None, :]
    params_b = jnp.tile(p0, (args.seeds, 1))

    table = jax.jit(jax.vmap(exact.trajectory_table))

    T = args.steps
    rec = dict(
        branches=branches, reset_idx=reset_idx, lattice=args.lattice,
        E0=E0.get(args.lattice, np.nan), lr=args.lr, init_seed=args.init_seed,
        traj_seed=args.traj_seed,
        a_nums=np.zeros((T, nb)), a_dens=np.zeros((T, nb)), a_value=np.zeros(T),
        a_params=np.zeros((T, exact.nparams)),
        b_nums=np.zeros((T, args.seeds, nb)), b_dens=np.zeros((T, args.seeds, nb)),
        b_value=np.zeros((T, args.seeds)), b_params=np.zeros((T, args.seeds, exact.nparams)),
        b_finite=np.ones((T, args.seeds), dtype=bool),
    )

    opt = optax.adam(learning_rate=args.lr)
    opt_a, opt_b = opt.init(params_a), opt.init(params_b)

    # Run B mirrors VariationalAnsatz.optimize's trajectory branch: same status
    # draws, same primed EMA baseline, same freezing of non-finite trials.
    key = jax.random.PRNGKey(args.traj_seed)
    status_shape = (args.seeds, 1, R)
    key, sub = jax.random.split(key)
    baseline = single._costs_vmapped(params_b, jax.random.uniform(sub, status_shape),
                                     jnp.zeros(args.seeds))

    t0 = time.perf_counter()
    for t in range(T):
        rec["a_params"][t] = np.asarray(params_a[0])
        rec["b_params"][t] = np.asarray(params_b)
        na, da = table(params_a)
        nb_, db_ = table(params_b)
        rec["a_nums"][t], rec["a_dens"][t] = np.asarray(na[0]), np.asarray(da[0])
        rec["b_nums"][t], rec["b_dens"][t] = np.asarray(nb_), np.asarray(db_)

        va, ga = exact._cost_vvag(params_a)
        upd, opt_a = opt.update(ga, opt_a)
        params_a = optax.apply_updates(params_a, upd)
        rec["a_value"][t] = float(va[0])

        key, sub = jax.random.split(key)
        status = jax.random.uniform(sub, status_shape)
        vb, gb = single._cost_vvag(params_b, status, baseline)
        finite = jnp.isfinite(gb).all(axis=-1) & jnp.isfinite(vb)
        upd, opt_b = opt.update(jnp.where(finite[:, None], gb, 0.0), opt_b)
        params_b = optax.apply_updates(params_b, jnp.where(finite[:, None], upd, 0.0))
        baseline = jnp.where(finite, single.baseline_beta * baseline
                             + (1.0 - single.baseline_beta) * vb, baseline)
        rec["b_value"][t] = np.asarray(vb)
        rec["b_finite"][t] = np.asarray(finite)

        if t % 25 == 0 or t == T - 1:
            ea = rec["a_nums"][t].sum() / rec["a_dens"][t].sum()
            eb = rec["b_nums"][t].sum(-1) / rec["b_dens"][t].sum(-1)
            print(f"step {t:>5}  E_A {ea:+.5f}  E_B {np.array2string(eb, precision=4)}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if (t + 1) % args.save_every == 0 or t == T - 1:
            save(rec, args.out, upto=t + 1)

    rec["sampling_check"] = sampling_check(exact, single, rec, args)
    save(rec, args.out, upto=T)
    return rec


def sampling_check(exact, single, rec, args):
    """At a few frozen points of run B, draw many single trajectories and compare
    their sample mean and spread against E and SE_1 from the exact table."""
    M = args.check_samples
    steps = sorted({0, rec["b_value"].shape[0] // 2, rec["b_value"].shape[0] - 1})
    key = jax.random.PRNGKey(12345)
    out = []
    for t in steps:
        p = jnp.tile(jnp.asarray(rec["b_params"][t, 0]), (M, 1))
        key, sub = jax.random.split(key)
        vals = np.asarray(single._costs_vmapped(
            p, jax.random.uniform(sub, (M, 1, exact.total_resets)), jnp.zeros(M)))
        s = stats_from_table(rec["b_nums"][t, 0], rec["b_dens"][t, 0])
        out.append([t, vals.mean(), vals.std(ddof=1), s["E"], s["SE"], M])
        print(f"sampling check step {t}: mean {vals.mean():+.5f} vs E {s['E']:+.5f} "
              f"(SE/sqrt(M) = {s['SE'] / np.sqrt(M):.2e}); std {vals.std(ddof=1):.5f} vs SE_1 {s['SE']:.5f}")
    return np.asarray(out)


def stats_from_table(nums, dens):
    """E, SE_1 and MAD from one trajectory table (works over leading axes)."""
    dens = np.clip(dens, 0.0, None)
    total = dens.sum(-1, keepdims=True)
    p = dens / total
    alive = dens > P_FLOOR
    Ei = np.where(alive, nums / np.where(alive, dens, 1.0), np.nan)
    E = nums.sum(-1) / total[..., 0]
    dev = np.where(alive, Ei - E[..., None], 0.0)
    var = (p * dev ** 2).sum(-1)
    return dict(p=p, Ei=Ei, E=E, SE=np.sqrt(var), MAD=(p * np.abs(dev)).sum(-1),
                norm=total[..., 0])


def save(rec, out, upto):
    os.makedirs(out, exist_ok=True)
    trimmed = {k: (v[:upto] if isinstance(v, np.ndarray) and k.startswith(("a_", "b_")) else v)
               for k, v in rec.items()}
    np.savez_compressed(os.path.join(out, "run.npz"), **trimmed)


# --------------------------------------------------------------------------- analysis

def analyse(out):
    d = dict(np.load(os.path.join(out, "run.npz"), allow_pickle=True))
    A = stats_from_table(d["a_nums"], d["a_dens"])
    B = stats_from_table(d["b_nums"], d["b_dens"])
    reset_idx = d["reset_idx"]
    sig_a = d["a_params"][:, reset_idx]
    sig_b = d["b_params"][:, :, reset_idx]

    # Every logged single-trajectory value must be one of that step's E^(i).
    gap = np.nanmin(np.abs(B["Ei"] - d["b_value"][..., None]), axis=-1)
    # Exact energy from the table vs run A's own loss.
    loss_gap = np.abs(A["E"] - d["a_value"])

    summary = dict(
        steps=len(A["E"]), seeds=B["E"].shape[1], ntraj=d["a_nums"].shape[1],
        E0=float(d["E0"]),
        E_A_final=float(A["E"][-1]),
        E_B_final=B["E"][-1].tolist(),
        SE_A_final=float(A["SE"][-1]),
        SE_B_final=B["SE"][-1].tolist(),
        SE_A_max=float(A["SE"].max()), SE_B_max=float(B["SE"].max()),
        path_err_final=(B["E"][-1] - A["E"][-1]).tolist(),
        max_norm_err=float(max(np.abs(A["norm"] - 1).max(), np.abs(B["norm"] - 1).max())),
        max_logged_vs_table=float(np.nanmax(gap)),
        max_tableE_vs_loss=float(loss_gap.max()),
        preset_A_final=(np.sin(sig_a[-1] / 2) ** 2).tolist(),
        preset_B_final=(np.sin(sig_b[-1] / 2) ** 2).tolist(),
        nonfinite_B_steps=int((~d["b_finite"]).any(-1).sum()),
    )
    return d, A, B, sig_a, sig_b, summary


def trajectory_labels(branches):
    """Trajectory labels i = ab of the Kraus note from the code's branch indices.

    The code stores one index j_r in {0, 1, 2} per reset, meaning a_r b_r = 00
    (no reset), 10 (reset, Z value 0), 11 (reset, Z value 1). The note writes
    every a_r first, then every b_r: i = a_0 ... a_{R-1} b_0 ... b_{R-1}.
    """
    branches = np.asarray(branches)
    a, b = (branches > 0).astype(int), (branches == 2).astype(int)
    return ["".join(map(str, ar)) + "".join(map(str, br)) for ar, br in zip(a, b)]


def place_labels(fig, ax, xs, ys, texts, fontsize=6.5, gap_px=3):
    """Write each label vertically above its point, nudging it upwards until it
    clears every label already placed (greedy, in display pixels)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px, to_data = ax.transData, ax.transData.inverted()
    placed = []
    for k in np.argsort(xs):
        x_px, y_px = to_px.transform((xs[k], ys[k]))
        txt = ax.text(xs[k], ys[k], texts[k], rotation=90, fontsize=fontsize,
                      family="monospace", ha="center", va="bottom", color="0.15",
                      bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        bb = txt.get_window_extent(renderer)
        w, h = bb.width, bb.height
        y0 = y_px + 5
        moved = True
        while moved:
            moved = False
            for (px0, px1, py0, py1) in placed:
                if x_px - w / 2 < px1 + gap_px and x_px + w / 2 > px0 - gap_px \
                        and y0 < py1 + gap_px and y0 + h > py0 - gap_px:
                    y0 = py1 + gap_px
                    moved = True
        placed.append((x_px - w / 2, x_px + w / 2, y0, y0 + h))
        txt.set_position(to_data.transform((x_px, y0)))


def plot(out):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, A, B, sig_a, sig_b, summary = analyse(out)
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))

    steps = np.arange(len(A["E"]))
    S = B["E"].shape[1]
    ca, cb, ch = "#1f4e8c", "#2f7a55", "#c26a1b"

    # 1. the three energy curves
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, d["b_value"][:, 0], ".", ms=2.5, color=ch, alpha=0.6,
            label="run B, single-trajectory value")
    ax.fill_between(steps, B["E"][:, 0] - B["SE"][:, 0], B["E"][:, 0] + B["SE"][:, 0],
                    color=ch, alpha=0.15, lw=0, label=r"run B, $E \pm \mathrm{SD}[\hat{E}]$")
    for s in range(1, S):
        ax.plot(steps, B["E"][:, s], color=cb, lw=0.8, alpha=0.35)
    ax.plot(steps, B["E"][:, 0], color=cb, lw=1.6, label="run B, true energy")
    ax.plot(steps, A["E"], color=ca, lw=1.8, label="run A, true energy")
    ax.axhline(float(d["E0"]), color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("energy")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "energy_curves.png"), dpi=160)
    plt.close(fig)

    # 2. standard error and path error
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(steps, A["SE"], color=ca, lw=1.6, label=r"$\mathrm{SD}[\hat{E}]$ along run A")
    for s in range(S):
        ax.plot(steps, B["SE"][:, s], color=ch, lw=0.9, alpha=0.6,
                label=r"$\mathrm{SD}[\hat{E}]$ along run B" if s == 0 else None)
        ax.plot(steps, np.abs(B["E"][:, s] - A["E"]), color=cb, lw=0.9, alpha=0.6,
                label=r"path error $|E_B - E_A|$" if s == 0 else None)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, None)
    ax.set_xlabel("step")
    ax.set_ylabel("energy")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "errors.png"), dpi=160)
    plt.close(fig)

    # 3. reset probabilities
    R = sig_a.shape[1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for r in range(R):
        ax.plot(steps, np.sin(sig_a[:, r] / 2) ** 2, color=ca, lw=1.4,
                label="run A" if r == 0 else None)
        ax.plot(steps, np.sin(sig_b[:, 0, r] / 2) ** 2, color=ch, lw=1.0, ls="--",
                label="run B (repeat 0)" if r == 0 else None)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$p_\mathrm{reset} = \sin^2(\sigma_r/2)$")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reset_probabilities.png"), dpi=160)
    plt.close(fig)

    # 4. snapshots of the trajectory table along run B, repeat 0
    plot_snapshots(os.path.join(out, "snapshots.png"), B["p"][:, 0], B["Ei"][:, 0],
                   B["E"][:, 0], B["SE"][:, 0], sig_b[:, 0], d["branches"])
    return summary


def plot_snapshots(path, p_all, Ei_all, E_all, SE_all, sig_all, branches, title=None):
    """Stem plots of p^(i) against E^(i), each stem labelled with its trajectory.

    Arrays are one run's history, indexed by step first. Snapshots are weighted
    to early training: once the resets switch off every later table looks the
    same, so evenly spaced snapshots would mostly repeat it.
    """
    import matplotlib.pyplot as plt
    ca, ch = "#1f4e8c", "#c26a1b"
    T = len(E_all)
    snaps = [t for t in (0, 25, 50, 100, 200) if t < T] + [T - 1]
    labels = trajectory_labels(branches)
    shown = [np.where(np.isfinite(Ei_all[t]) & (p_all[t] > 1e-6))[0] for t in snaps]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey=True)
    for ax, t, ok in zip(axes.flat, snaps, shown):
        p, Ei, E, se = p_all[t], Ei_all[t], E_all[t], SE_all[t]
        ax.axvspan(E - se, E + se, color=ch, alpha=0.15, lw=0)
        ax.vlines(Ei[ok], 0, p[ok], color=ca, lw=1.5)
        ax.plot(Ei[ok], p[ok], "o", color=ca, ms=3.5)
        ax.axvline(E, color="k", ls="--", lw=0.9)
        pr = ", ".join(f"{v:.2f}" for v in np.sin(sig_all[t] / 2) ** 2)
        ax.set_title(f"step {t}   $p_\\mathrm{{reset}}$ = {pr}", fontsize=8.5)
        ax.set_xlabel(r"$E^{(\mathbf{i})}$", fontsize=9)
        ax.set_ylim(-0.03, 1.3)
        lo, hi = min(Ei[ok].min(), E - se), max(Ei[ok].max(), E + se)
        # A lone surviving trajectory would otherwise give a zero-width axis.
        pad = 0.08 * (hi - lo) if hi - lo > 1e-3 else 0.5
        ax.set_xlim(lo - pad, hi + pad)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p^{(\mathbf{i})}$")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    for ax, t, ok in zip(axes.flat, snaps, shown):
        place_labels(fig, ax, Ei_all[t][ok], p_all[t][ok], [labels[k] for k in ok])
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_run_a(out, xmax_run_a=400):
    """The reference run on its own: what a single trajectory would have cost at
    each point of the exact optimisation. Writes to <out>/run_a/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, A, _, sig_a, _, _ = analyse(out)
    dest = os.path.join(out, "run_a")
    os.makedirs(dest, exist_ok=True)
    steps = np.arange(len(A["E"]))
    ca, ch = "#1f4e8c", "#c26a1b"
    preset = np.sin(sig_a / 2) ** 2
    # Run A has settled long before the end; show only the part that changes.
    xmax = min(len(steps) - 1, xmax_run_a)
    lat = str(d["lattice"]).replace("x", "×")

    # 1. energy with +-SE_1, and the reset probabilities underneath
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1]))
    ax.fill_between(steps, A["E"] - A["SE"], A["E"] + A["SE"], color=ch, alpha=0.2,
                    lw=0, label=r"$E \pm \mathrm{SD}[\hat{E}]$")
    ax.plot(steps, A["E"], color=ca, lw=1.8, label=r"$E$")
    ax.axhline(float(d["E0"]), color="0.5", ls="--", lw=0.8)
    ax.set_ylabel("energy")
    ax.set_title(f"Energy and reset probabilities during training ({lat})")
    ax.legend(frameon=False, fontsize=8)
    for r in range(preset.shape[1]):
        ax2.plot(steps, preset[:, r], lw=1.4, label=f"reset {r}")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xlim(0, xmax)
    ax2.set_xlabel("step")
    ax2.set_ylabel(r"$p_\mathrm{reset}$")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(dest, "energy_and_resets.png"), dpi=160)
    plt.close(fig)

    # 2. size of the single-trajectory error along run A
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(steps, A["SE"], color=ca, lw=1.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, None)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\mathrm{SD}[\hat{E}]$")
    ax.set_title(f"Single-trajectory standard deviation during training ({lat})")
    fig.tight_layout()
    fig.savefig(os.path.join(dest, "standard_error.png"), dpi=160)
    plt.close(fig)

    # 3. trajectory snapshots
    plot_snapshots(os.path.join(dest, "snapshots.png"), A["p"], A["Ei"], A["E"],
                   A["SE"], sig_a, d["branches"],
                   title=f"Trajectory probabilities and energies during training ({lat})")
    print(f"wrote run A plots to {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", default="2x2", choices=sorted(CONFIG))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=5, help="independent repeats of run B")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--init-seed", type=int, default=4321)
    ap.add_argument("--traj-seed", type=int, default=99)
    ap.add_argument("--check-samples", type=int, default=2000)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot-only", default=None, metavar="DIR")
    ap.add_argument("--run-a-only", action="store_true",
                    help="with --plot-only: plot the reference run alone into DIR/run_a/")
    args = ap.parse_args()

    if args.plot_only:
        (plot_run_a if args.run_a_only else plot)(args.plot_only)
        return
    args.out = args.out or os.path.join("outputs", "single_traj_error", args.lattice)
    run(args)
    plot(args.out)


if __name__ == "__main__":
    main()
