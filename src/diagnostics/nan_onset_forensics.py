"""What is the SVD looking at on the step before a purified trial goes NaN?

Run:
    python -m src.diagnostics.nan_onset_forensics --lattice 3x3
    python -m src.diagnostics.nan_onset_forensics --lattice 3x3 --runs <ckpt1> <ckpt2>

THE QUESTION
------------
`optimize()` freezes any trial whose gradient goes non-finite, so a checkpoint's
`all_params` records, for each trial that died, the LAST FINITE parameter vector
it ever held. That is the only place the failure can be caught after the fact,
and it is a real measurement rather than a reconstruction: those parameters are
exactly what the forward pass was handed on the step it produced a NaN.

This module rebuilds the MPS at those parameters and asks which of three
mechanisms the spectrum is consistent with.

  1. DEGENERACY (what the Lorentzian is for).  jaxsvd_bwd's off-diagonal factor
     is F_ij = _safe_reciprocal(s_i^2 - s_j^2), and the toric code's flat
     spectrum makes near-degenerate pairs routine. The Lorentzian
     x/(x^2 + 1e-15) BOUNDS this at 1/(2 sqrt(eps)) = 1.58e7 -- large, but
     finite, and finite is the whole point. A degeneracy cannot produce a NaN.
     It can produce a huge-but-finite gradient.

  2. RANK DEFICIENCY.  Every split is forced to keep `max_singular_values`
     entries because jit needs static shapes. Once the true Schmidt rank falls
     below that number the remainder come back as structural zeros. In the
     BACKWARD pass these are benign -- _safe_reciprocal(0) = 0 exactly, and the
     u, vh columns they multiply are the arbitrary null-space directions the
     decomposition is free to return. The exposure is in the FORWARD pass: a
     rank-deficient matrix is where an iterative SVD kernel has no unique answer
     to converge to, which is the failure mode project memory records for
     cuSOLVER's batched Jacobi under vmap.

  3. ON-PEAK SINGULAR VALUES.  A kept s_i that is nonzero but of order
     sqrt(eps) = 3.16e-8 is the one case NO amount of Lorentzian broadening
     helps, because it sits exactly at the peak: it enters both Sinv = 1/s and,
     through s_i^2 - s_j^2 ~ -s_j^2, the off-diagonal F. This is the genuinely
     ill-conditioned case, distinct from both of the above.

The three are separated by looking at the KEPT block only, since that is what
the custom VJP differentiates, and by classifying each entry as zero /
on-peak / ordinary before any statistic is taken.

CONTROLS
--------
A number measured only at the failure point says nothing. Each dying trial is
compared against two matched controls from the same run:

    step 0            the same trial, at initialisation, before anything drifted
    surviving trials  a different trial of the SAME run at the SAME snapshot,
                      which did not die

If the failing trial's spectrum is indistinguishable from its controls, the
mechanism is not in the spectrum at all and the cause is elsewhere.
"""

import argparse
import glob
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np

from src.diagnostics.mps_complexity import build_ansatz
from src.diagnostics.reset_hotspot_profile import (
    DANGER, DANGER_HI, EXACT_ZERO, SVD_EPS, forward, numerics, profiling,
)


def snapshot_stride(z):
    """Steps between stored snapshots. `step_times` is per step; all_* per snapshot."""
    nsteps = int(z["step_times"].shape[0])
    nsnap = int(z["nsnapshots_filled"])
    return max(1, nsteps // nsnap)


def death_points(z):
    """(trial, first_bad_snapshot) for every trial that went non-finite."""
    E = z["all_energies"]
    n = int(z["nsnapshots_filled"])
    out = []
    for t in range(E.shape[0]):
        bad = np.where(~np.isfinite(E[t, :n]))[0]
        if bad.size:
            out.append((t, int(bad[0])))
    return out


def survivors(z):
    E = z["all_energies"]
    n = int(z["nsnapshots_filled"])
    return [t for t in range(E.shape[0]) if np.all(np.isfinite(E[t, :n]))]


def spectrum_summary(ansatz, params, mode):
    """Aggregate the whole forward pass's kept spectra into one row."""
    with profiling() as records:
        forward(ansatz, params, mode)
    kept = rank = zeros = onpeak = 0
    degen_splits = onpeak_splits = defic_splits = nsplits = 0
    worst_ratio = 1.0
    min_s = np.inf
    min_gap = np.inf
    max_amp = 0.0
    # Split the amplification by which term of jaxsvd_bwd it comes through.
    # This matters because the two are NOT both fixable the same way: the gap
    # channel is the degeneracy the Lorentzian exists for, the s channel is
    # 1/s and no broadening constant removes it -- raising eps only moves the
    # peak to a different s. Reporting only their max hides which is active.
    max_amp_s = 0.0
    max_amp_gap = 0.0
    for r in records:
        if r["kind"] != "svd":
            continue
        d = numerics(r)
        if d is None:
            continue
        nsplits += 1
        kept += d["n_kept"]
        rank += d["rank"]
        zeros += d["n_exact_zero"]
        onpeak += d["n_danger_s"]
        onpeak_splits += int(d["n_danger_s"] > 0)
        degen_splits += int(d["n_danger_gap_nz"] > 0)
        defic_splits += int(d["deficient"] > 0)
        if d["n_kept"]:
            worst_ratio = min(worst_ratio, d["rank"] / d["n_kept"])
        if d["smin_nonzero"] > 0:
            min_s = min(min_s, d["smin_nonzero"])
        if d["min_gap_nz"] > 0:
            min_gap = min(min_gap, d["min_gap_nz"])
        max_amp = max(max_amp, d["amp_gap_nz"], d["amp_s_nz"])
        max_amp_s = max(max_amp_s, d["amp_s_nz"])
        max_amp_gap = max(max_amp_gap, d["amp_gap_nz"])
    return {
        "splits": nsplits, "kept": kept, "rank": rank,
        "rank_frac": rank / kept if kept else float("nan"),
        "zeros": zeros, "onpeak": onpeak,
        "onpeak_splits": onpeak_splits, "degen_splits": degen_splits,
        "defic_splits": defic_splits, "worst_rank_ratio": worst_ratio,
        "min_s": min_s, "min_gap": min_gap, "max_amp": max_amp,
        "amp_s": max_amp_s, "amp_gap": max_amp_gap,
    }


def grad_finite(ansatz, params, mode):
    """Is the REAL training gradient finite at these parameters, on this host?

    Uses the same jitted vmapped callable optimize() drives, with the parameter
    vector tiled across the trial axis, so the batched kernel sees a batch --
    which is the condition project memory records as necessary for the GPU
    failure. A `True` here on CPU does not contradict a NaN on GPU; it localises
    the failure to the kernel rather than to the parameters.
    """
    import jax
    import jax.numpy as jnp

    batch = jnp.asarray(np.tile(np.asarray(params), (ansatz.trials, 1)))
    if mode in ("traj", "none"):
        key = jax.random.PRNGKey(0)
        status = jax.random.uniform(
            key, (ansatz.trials, ansatz.n_trajectories, ansatz.total_resets))
        v, g = ansatz._cost_vvag(batch, status, jnp.zeros(ansatz.trials))
    else:
        v, g = ansatz._cost_vvag(batch)
    g = np.asarray(jax.block_until_ready(g))
    v = np.asarray(v)
    return bool(np.all(np.isfinite(g))), float(np.max(np.abs(g))), v


HEADER = (f"    {'label':>26} {'splits':>7} {'kept':>6} {'rank':>6} {'rank%':>6} "
          f"{'zeros':>6} {'rank-def':>9} {'worst r/k':>10} "
          f"{'on-peak s':>10} {'degen':>7} {'min s>0':>10} "
          f"{'amp via 1/s':>12} {'amp via gap':>12}")


def show(label, d):
    print(f"    {label:>26} {d['splits']:>7} {d['kept']:>6} {d['rank']:>6} "
          f"{100 * d['rank_frac']:>5.0f}% {d['zeros']:>6} "
          f"{str(d['defic_splits']) + '/' + str(d['splits']):>9} "
          f"{d['worst_rank_ratio']:>10.2f} "
          f"{str(d['onpeak_splits']) + '/' + str(d['splits']):>10} "
          f"{str(d['degen_splits']) + '/' + str(d['splits']):>7} "
          f"{d['min_s']:>10.2e} {d['amp_s']:>12.2e} {d['amp_gap']:>12.2e}")


def timeseries(ansatz, z, mode, trial, upto, stride_snap, label):
    """Walk one trial's stored snapshots and print the spectrum row at each.

    A single measurement at the failure point cannot distinguish "this quantity
    caused the failure" from "this quantity is always like that". The time
    series can: a mechanism that drives the failure has to MOVE as the failure
    approaches, and a background property will sit flat.
    """
    P = z["all_params"]
    print(f"\n  --- {label}: trial {trial}, snapshots 0..{upto} "
          f"every {stride_snap} ---")
    print(f"    {'snapshot':>9} {'E':>10} {'rank%':>6} {'zeros':>6} "
          f"{'on-peak splits':>15} {'on-peak sv':>11} {'degen splits':>13} "
          f"{'min s>0':>10} {'amp via 1/s':>12} {'amp via gap':>12}")
    E = z["all_energies"]
    for snap in list(range(0, upto + 1, stride_snap)) + ([upto] if upto % stride_snap else []):
        d = spectrum_summary(ansatz, np.asarray(P[trial, snap]), mode)
        e = E[trial, snap]
        print(f"    {snap:>9} {e:>10.4f} {100 * d['rank_frac']:>5.0f}% "
              f"{d['zeros']:>6} "
              f"{str(d['onpeak_splits']) + '/' + str(d['splits']):>15} "
              f"{d['onpeak']:>11} "
              f"{str(d['degen_splits']) + '/' + str(d['splits']):>13} "
              f"{d['min_s']:>10.2e} {d['amp_s']:>12.2e} {d['amp_gap']:>12.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lattice", default="3x3")
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--reset-layers", default="[1]")
    p.add_argument("--runs", nargs="*", default=None,
                   help="checkpoint.npz paths; default = the 3-arm purified runs")
    p.add_argument("--max-trials", type=int, default=3,
                   help="how many dying trials to profile per run (each is a "
                        "full eager forward pass)")
    p.add_argument("--timeseries", action="store_true",
                   help="walk each dying trial's snapshots up to its death, plus "
                        "one survivor over the same range")
    p.add_argument("--ts-stride", type=int, default=3,
                   help="snapshots between timeseries rows")
    p.add_argument("--check-grad", action="store_true",
                   help="also evaluate the real jitted gradient at each point")
    args = p.parse_args()

    import ast
    reset_layers = ast.literal_eval(args.reset_layers)
    Lx, Ly = (int(v) for v in args.lattice.split("x"))

    runs = args.runs or sorted(glob.glob(
        "logs/2026-08-26_14-02-49_3x3_bd64_r1_pur_3armctl_*/plots/checkpoint.npz"))
    if not runs:
        raise SystemExit("no checkpoints found; pass --runs explicitly")

    print(f"Lorentzian eps={SVD_EPS:g}  peak at sqrt(eps)={DANGER:.2e}  "
          f"ceiling 1/(2 sqrt(eps))={1 / (2 * DANGER):.2e}")
    print(f"zero cut={EXACT_ZERO:g} (relative to each split's s_max), "
          f"on-peak band=(0, {DANGER_HI:g})")

    cache = {}
    for path in runs:
        z = np.load(path)
        mode = str(z["reset_mode"])
        seed = int(z["seed"])
        cap = int(z["bond_dim"])
        stride = snapshot_stride(z)
        deaths = death_points(z)
        alive = survivors(z)
        print(f"\n=== {path}")
        print(f"    arm={mode} seed={seed} bd={cap} chain={int(z['n_mps_qubits'])} "
              f"snapshot every {stride} steps; "
              f"{len(deaths)}/{z['all_energies'].shape[0]} trials went non-finite")
        if not deaths:
            print("    (no deaths in this run)")
            continue

        key = (Lx, Ly, mode, cap)
        if key not in cache:
            cache[key] = build_ansatz(Lx, Ly, mode, nlayers=args.nlayers,
                                      reset_layers=reset_layers, cap=cap)
        ansatz = cache[key]
        P = z["all_params"]

        print(HEADER)
        for trial, bad in deaths[:args.max_trials]:
            last_good = max(0, bad - 1)
            pts = [
                (f"T{trial} pre-NaN s{last_good}", P[trial, last_good]),
                (f"T{trial} init s0", P[trial, 0]),
            ]
            for ctrl in alive[:2]:
                pts.append((f"T{ctrl} SURVIVOR s{last_good}", P[ctrl, last_good]))
            for label, prm in pts:
                d = spectrum_summary(ansatz, np.asarray(prm), mode)
                show(label, d)
                if args.check_grad:
                    ok, gmax, _ = grad_finite(ansatz, np.asarray(prm), mode)
                    print(f"    {'':>26}   jitted grad finite on this host: "
                          f"{ok}   max|g|={gmax:.3e}")
            print()

        if args.timeseries:
            trial, bad = deaths[0]
            timeseries(ansatz, z, mode, trial, max(0, bad - 1), args.ts_stride,
                       "DIES")
            if alive:
                timeseries(ansatz, z, mode, alive[0], max(0, bad - 1),
                           args.ts_stride, "SURVIVES (control)")


if __name__ == "__main__":
    main()
