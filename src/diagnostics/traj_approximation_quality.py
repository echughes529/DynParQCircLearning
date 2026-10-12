"""How big an approximation is the single-trajectory estimator, really?

Run:
    python -m src.diagnostics.traj_approximation_quality \
        --lattice 3x3 \
        --checkpoint logs/<traj run>/plots/checkpoint.npz \
        --out outputs/three_resets/traj_approximation_3x3.csv

THE QUESTION
------------
The purified and enumerated arms evaluate the reset channel exactly. The
trajectory arm samples ONE of its 3^R Kraus branches per energy evaluation. That
looks like a severe approximation, yet the trajectory arm converges much like the
exact ones. Why?

The hypothesis this tests is that the approximation is not severe at all, because
the channel is dominated by a single branch. The three Kraus operators are

    K0 = cos(t/2) I,   K1 = sin(t/2)|0><0|,   K2 = sin(t/2)|0><1|

so a reset event fires with probability p = sin^2(t/2). If the optimiser drives
the reset angles t towards 0, then p -> 0, the all-K0 "nothing fired anywhere"
branch carries essentially all the weight, and sampling one branch is very nearly
the same as evaluating the channel. In that regime "the trajectory arm is a big
approximation" would simply be false for the states training actually visits,
even though it is true for a generic parameter vector.

TWO MEASUREMENTS
----------------
1. BRANCH OCCUPANCY. Pure post-processing of the saved parameters, essentially
   free. Per snapshot, read the reset angles and report p per reset event and
   P(no reset fires anywhere) = prod_r (1 - p_r). If that probability climbs
   towards 1 over training, the channel is collapsing onto one branch and the
   hypothesis holds. Reported alongside the exact branch-weight distribution, so
   "dominant" is a measured share rather than an inference from p alone.

2. TRAJECTORY-VS-EXACT ENERGY GAP. What the occupancy argument predicts, checked
   directly. At each snapshot, sample N trajectory energies and compare their mean
   against the EXACT channel energy at the same parameters, obtained by walking
   all 3^R Kraus branches:

       E = sum_j <phi_j|H|phi_j> / sum_j <phi_j|phi_j>

   with the branch states left unnormalised, so each one's surviving norm IS its
   probability (see VariationalAnsatz.enumerated_energy_from_params for the
   derivation). Branches are walked one at a time, not vmapped -- there are no
   gradients here, so only one branch is ever resident, and jax.vmap would not run
   on the NumPy backend anyway.

   Two numbers come out of this, and the second needs care.

   The SPREAD (standard deviation across draws) is how much noise a single draw
   feeds into the optimiser. If it collapses over training, late-stage steps are
   effectively exact.

   The BIAS (trajectory mean minus exact) is zero in expectation -- the estimator
   is unbiased by construction -- but a FINITE sample of it is not, and the
   discrepancy is largest exactly where the channel is most dominated. Once one
   branch carries 99.9% of the weight, every draw in a modest sample takes that
   branch, the spread reads exactly 0, and the sample mean is that branch's own
   energy rather than the channel average. The residual gap is then the
   contribution of the branches that were never drawn, and it shrinks alongside
   their weight. So a non-zero bias with a zero spread is not a bug: it is the
   measurement telling you the tail is unresolved at this sample size.
   `energy_top_branch` is recorded so this is checkable rather than argued --
   when the spread is 0, the trajectory mean equals it exactly.

WHAT THE ANSWER LOOKS LIKE
--------------------------
If P(no reset) -> 1 and the spread collapses, the answer to "why do they converge
alike" is that the trained circuit switches its own resets off, so the two arms
end up simulating nearly the same thing. If instead the spread stays large while
training still converges, the occupancy story is wrong and the answer has to be
the SGD one -- an unbiased gradient descends regardless of variance -- which is
what the estimator statistics in docs/three_reset_implementations.md section 6.1
already speak to.

The two are distinguishable here, which is the point of measuring both.
"""

import argparse
import itertools
import sys
import time
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np

from src.diagnostics.complexity_through_training import (
    load_checkpoint,
    verify_ordering,
    write_rows,
)
from src.diagnostics.mps_complexity import (
    build_ansatz,
    error_split_conf,
    numpy_mps_mode,
    rebuild,
    set_split_conf,
)


def branch_occupancy(thetas):
    """Per-event firing probabilities and the weight of the all-K0 branch.

    `thetas` is one snapshot's reset angles. p = sin^2(theta/2) is the probability
    that a given reset event fires; the events are independent given the
    parameters, so the "nothing fired anywhere" branch carries prod_r (1 - p_r).

    Angles are used as-is, not folded into [0, pi]. Adam is free to walk a reset
    angle slightly negative, and sin^2 is even, so a small negative angle means the
    same near-zero firing probability as its positive mirror.
    """
    thetas = np.asarray(thetas, dtype=float)
    p = np.sin(thetas / 2.0) ** 2
    return {
        "p_fire_min": float(p.min()),
        "p_fire_max": float(p.max()),
        "p_fire_mean": float(p.mean()),
        "p_no_reset_anywhere": float(np.prod(1.0 - p)),
        "theta_min": float(thetas.min()),
        "theta_max": float(thetas.max()),
        "theta_absmax": float(np.abs(thetas).max()),
    }


def exact_channel_energy(ansatz, params, branches):
    """Exact reset-channel energy, by walking every Kraus branch.

    Returns the energy, the branch weight distribution, the weight sum, and each
    branch's own normalised energy. The sum is 1 in exact arithmetic because the
    Kraus set is trace preserving, so its shortfall is the truncation error of the
    whole walk -- a free correctness check on every measurement rather than an
    assumption.

    The per-branch energies cost nothing extra (each branch is built anyway) and
    make the dominated regime legible: once one branch carries nearly all the
    weight, every trajectory draw returns that branch's energy, and having it in
    the same row is what distinguishes "the estimator is biased" from "the tail
    was never sampled".
    """
    num = 0.0
    weights = []
    branch_energies = []
    for branch in branches:
        qc, _ = rebuild(ansatz, params, "enum", branch=branch)
        # Unnormalised <phi|H|phi>; its weight is <phi|phi>, so the two sums
        # combine into the channel energy without any per-branch division.
        e_unnorm = float(np.real(ansatz._energy_of_circuit(qc)))
        w = float(np.real(np.asarray(qc.get_norm())) ** 2)
        num += e_unnorm
        weights.append(w)
        branch_energies.append(e_unnorm / w if w > 1e-14 else float("nan"))
    weights = np.asarray(weights)
    den = weights.sum()
    return ((float(num / den) if den > 1e-12 else float("nan")),
            weights, float(den), np.asarray(branch_energies))


def trajectory_energies(ansatz, params, rng, n):
    """N sampled single-trajectory energies at these parameters.

    Each draw normalises its own state before taking the expectation, which is
    what the training estimator does (dice_energy_from_params calls
    _normalize_mps_if_requested before _energy_of_circuit). Skipping that would
    read back the branch probability multiplied into the energy and inflate the
    apparent spread.
    """
    out = []
    for _ in range(n):
        qc, _ = rebuild(ansatz, params, "traj",
                        status=rng.uniform(size=ansatz.total_resets))
        norm2 = float(np.real(np.asarray(qc.get_norm())) ** 2)
        e = float(np.real(ansatz._energy_of_circuit(qc)))
        out.append(e / norm2 if norm2 > 1e-12 else float("nan"))
    return np.asarray(out)


def run(Lx, Ly, ck, args):
    rows = []
    R = None

    # Both arms are built under the normal JAX backend: __post_init__ constructs
    # vmapped value_and_grad helpers, which the NumPy backend refuses outright.
    # They are never called here.
    print("\nbuilding ansaetze:")
    t0 = time.perf_counter()
    traj = build_ansatz(Lx, Ly, "traj", nlayers=args.nlayers,
                        reset_layers=args.reset_layers, cap=args.cap)
    print(f"  traj: chain={traj.n_mps_qubits} R={traj.total_resets} "
          f"[{time.perf_counter() - t0:.1f}s]")
    t0 = time.perf_counter()
    enum = build_ansatz(Lx, Ly, "enum", nlayers=args.nlayers,
                        reset_layers=args.reset_layers, cap=args.cap)
    print(f"  enum: chain={enum.n_mps_qubits} R={enum.total_resets} "
          f"branches={3 ** enum.total_resets} [{time.perf_counter() - t0:.1f}s]")

    ok, detail = verify_ordering(traj, ck["path"])
    print(f"  ordering check (traj): "
          f"{ {True: 'OK', False: 'MISMATCH', None: 'unverifiable'}[ok] } -- {detail}")
    if ok is False:
        raise SystemExit("rebuilt chain ordering differs from the training run's; refusing to continue")
    # The enumerated arm shares the ancilla-free chain, so it must land on the
    # same ordering as the trajectory arm. If it does not, the exact reference is
    # being computed on a different chain from the thing it is a reference for.
    if traj.sys_mps_positions != enum.sys_mps_positions:
        raise SystemExit(
            "trajectory and enumerated arms produced different chain orderings; "
            "the exact reference would not be comparable to the sampled energies"
        )

    R = traj.total_resets
    branches = [np.asarray(b, dtype=int)
                for b in itertools.product(range(3), repeat=R)]
    print(f"  exact reference walks {len(branches)} branches per snapshot")

    all_params = ck["all_params"]
    reset_idx = traj.reset_param_indices
    snapshots = list(range(0, ck["filled"], args.snapshot_stride))
    if snapshots[-1] != ck["filled"] - 1:
        snapshots.append(ck["filled"] - 1)
    trials = list(range(min(args.trials, all_params.shape[0])))

    print(f"\n  {len(snapshots)} snapshots x {len(trials)} trials, "
          f"{args.traj_samples} trajectory draws each")
    hdr = (f"  {'step':>6s} {'tr':>3s} {'p_fire':>9s} {'P(no rst)':>10s} "
           f"{'E exact':>12s} {'E traj':>12s} {'bias':>10s} {'spread':>10s} "
           f"{'top wt':>8s} {'sec':>6s}")

    with numpy_mps_mode():
        set_split_conf(traj, error_split_conf(args.tolerance, args.cap))
        set_split_conf(enum, error_split_conf(args.tolerance, args.cap))
        for si, snapshot in enumerate(snapshots):
            if si % 15 == 0:
                print(hdr, flush=True)
            for trial in trials:
                params = np.asarray(all_params[trial, snapshot, :])
                t0 = time.perf_counter()

                occ = branch_occupancy(params[reset_idx])
                e_exact, weights, wsum, branch_energies = exact_channel_energy(
                    enum, params, branches)
                rng = np.random.default_rng(args.traj_seed + 1000 * snapshot + trial)
                e_traj = trajectory_energies(traj, params, rng, args.traj_samples)
                dt = time.perf_counter() - t0

                order = np.argsort(weights)[::-1]
                mean = float(np.mean(e_traj))
                std = float(np.std(e_traj, ddof=1)) if len(e_traj) > 1 else 0.0
                row = dict(
                    lattice=f"{Lx}x{Ly}", param_mode=ck["reset_mode"],
                    param_seed=ck["seed"], param_checkpoint=ck["path"],
                    snapshot=snapshot, step=snapshot * ck["stride"], trial=trial,
                    total_resets=R, n_branches=len(branches),
                    tolerance=args.tolerance,
                    training_energy=(float(ck["all_energies"][trial, snapshot])
                                     if ck["all_energies"] is not None else float("nan")),
                    energy_exact=e_exact,
                    energy_traj_mean=mean,
                    energy_traj_std=std,
                    # Standard error of the mean: what the bias check has to clear
                    # before a non-zero bias means anything.
                    energy_traj_sem=(std / np.sqrt(len(e_traj)) if len(e_traj) > 1 else 0.0),
                    energy_traj_min=float(e_traj.min()),
                    energy_traj_max=float(e_traj.max()),
                    bias=mean - e_exact,
                    n_draws=len(e_traj),
                    # Shortfall from 1 is the truncation error of the branch walk.
                    branch_weight_sum=wsum,
                    top_branch_weight=float(weights[order[0]] / wsum),
                    top2_branch_weight=float(weights[order[:2]].sum() / wsum),
                    # The dominant branch's own energy. When the spread is 0 every
                    # draw took this branch, so energy_traj_mean equals it and the
                    # gap to energy_exact is the unsampled tail, not estimator bias.
                    energy_top_branch=float(branch_energies[order[0]]),
                    # Participation ratio: the effective number of branches
                    # actually carrying weight. 1 means fully dominated by one.
                    branch_participation=float(
                        (weights.sum() ** 2) / max((weights ** 2).sum(), 1e-300)),
                    seconds=dt, **occ,
                )
                rows.append(row)
                write_rows(rows, args.out)
                print(f"  {row['step']:>6d} {trial:>3d} {occ['p_fire_mean']:>9.2e} "
                      f"{occ['p_no_reset_anywhere']:>10.6f} {e_exact:>12.6f} "
                      f"{mean:>12.6f} {row['bias']:>10.2e} {std:>10.2e} "
                      f"{row['top_branch_weight']:>8.5f} {dt:>6.2f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", default="3x3")
    parser.add_argument("--checkpoint", required=True,
                        help="a trajectory run's checkpoint.npz (needs track_params on)")
    parser.add_argument("--tolerance", type=float, default=1e-8,
                        help="truncation tolerance for both the sampled and exact states; "
                             "tight, because this measures an energy difference and must "
                             "not confound it with truncation")
    parser.add_argument("--cap", type=int, default=512)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--snapshot-stride", type=int, default=1)
    parser.add_argument("--traj-samples", type=int, default=32)
    parser.add_argument("--traj-seed", type=int, default=20260901)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--reset-layers", default="[1]")
    parser.add_argument("--out", default="outputs/three_resets/traj_approximation.csv")
    args = parser.parse_args()

    import ast
    args.reset_layers = ast.literal_eval(args.reset_layers)
    Lx, Ly = (int(v) for v in args.lattice.split("x"))

    ck = load_checkpoint(args.checkpoint)
    if ck["reset_mode"] not in ("traj", "unknown"):
        print(f"WARNING: --checkpoint is a {ck['reset_mode']!r} run, not a trajectory one. "
              "The occupancy and gap numbers still describe those parameters, but they "
              "are not the parameters the trajectory estimator actually visited.")
    print(f"loaded: {args.checkpoint}\n  seed={ck['seed']} bd={ck['bond_dim']} "
          f"chain={ck['n_mps_qubits']} R={ck['total_resets']} "
          f"snapshots={ck['filled']} stride={ck['stride']}")

    t0 = time.perf_counter()
    rows = run(Lx, Ly, ck, args)
    write_rows(rows, args.out, final=True)
    print(f"total {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
