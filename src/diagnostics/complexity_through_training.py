"""How complex is each arm's MPS at every point along a training run?

Run:
    python -m src.diagnostics.complexity_through_training \
        --lattice 3x3 \
        --traj-checkpoint logs/<traj run>/plots/checkpoint.npz \
        --pur-checkpoint  logs/<pur run>/plots/checkpoint.npz \
        --out outputs/three_resets/complexity_through_training_3x3.csv

WHAT THIS MEASURES
------------------
The training loop stores the parameter vector every `howoften_tosave` steps (150
snapshots over 1500 steps at the production config). This walks those snapshots
and, at each one, rebuilds the state from scratch on the NumPy backend with
*error-tolerance* truncation -- see mps_complexity for why that cannot be done
inside the training loop itself -- and records how big the resulting MPS is.

The result is a complexity curve through training, rather than the two-point
ladder measurement in docs/three_reset_implementations.md section 2.

TWO COMPARISONS, AND WHY BOTH ARE NEEDED
----------------------------------------
`as-trained` rebuilds each arm's own parameter history in its own mode. This is
the literal question -- what did each simulation actually have to carry? -- but it
conflates two things, because after step 0 the two runs' parameters diverge. An
as-trained difference mixes "this reset implementation is cheaper to represent"
with "these two runs ended up in different parts of parameter space".

`same-params` takes ONE parameter history and rebuilds it in BOTH modes. Because
nparams is identical across the three reset implementations (ansatz_classes.py,
`self.nparams` does not depend on the mode), the same vector is meaningful in
either. This is the controlled version: the parameters are held fixed, so any
difference is attributable to the reset implementation alone.

READING THE OUTPUT
------------------
One CSV row per (comparison, param source, rebuild mode, snapshot, trial,
tolerance). The columns that matter:

  max_bond_dim    widest bond. Compare WITHIN an arm across training. Comparing
                  it BETWEEN arms is unsound: the trajectory chain is 12 sites at
                  3x3 and the purified chain is 20, so it is not the same cut.
  mps_entries     total complex entries stored. THIS is the cross-arm number --
                  it is the real memory footprint and does not care how long the
                  chain is.
  max_entropy_vn  physical entanglement rather than representational cost. For
                  the purified arm this is where the ancilla which-path record
                  shows up.
  discarded_wt    1 - ||psi||^2, what truncation actually threw away. If this is
                  not near the tolerance, the cap bound instead.
  cap_bound       true means the safety cap bound and the row is a CEILING, not a
                  measurement. Do not quote those bond dimensions.

Trajectory rebuilds are stochastic (one sampled branch of the reset channel per
build), so each snapshot is measured over several draws and reported as max /
median / min. The max is the honest one for a capacity question: the bond
dimension has to serve every trajectory the optimiser might draw, not the average.

The CSV is rewritten after every row, so a job killed at walltime keeps
everything it had measured.
"""

import argparse
import csv
import os
import sys
import time
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np

from src.diagnostics.mps_complexity import (
    ANCILLA_FREE,
    build_ansatz,
    measure,
    numpy_mps_mode,
)

DEFAULT_TOLERANCES = (1e-4, 1e-6, 1e-8)


def load_checkpoint(path):
    """Parameter history plus the run identity needed to rebuild it faithfully."""
    d = np.load(path, allow_pickle=True)

    def scalar(key, default=None):
        if key not in d.files:
            return default
        v = d[key]
        return v.item() if v.shape == () else v

    all_params = d["all_params"]
    filled = int(scalar("nsnapshots_filled", all_params.shape[1]))
    if all_params.size == 0 or not np.any(all_params):
        raise SystemExit(
            f"{path}: all_params is empty or all zeros -- that run had track_params off, "
            "so there is no parameter history to rebuild. Re-run with TRACK_PARAMS=true."
        )
    last_step = int(scalar("step", 0))
    # howoften_tosave is not stored directly, but snapshot k is step k*stride and
    # the last filled snapshot is at `step`, so the stride follows.
    stride = int(round(last_step / max(filled - 1, 1))) if filled > 1 else 1

    return {
        "path": path,
        "all_params": all_params,
        "all_energies": d["all_energies"] if "all_energies" in d.files else None,
        "filled": filled,
        "stride": stride,
        "reset_mode": str(scalar("reset_mode", "unknown")),
        "seed": int(scalar("seed", -1)),
        "bond_dim": int(scalar("bond_dim", -1)),
        "n_mps_qubits": int(scalar("n_mps_qubits", -1)),
        "total_resets": int(scalar("total_resets", -1)),
        "trials": int(scalar("trials", all_params.shape[0])),
    }


def verify_ordering(ansatz, checkpoint_path):
    """Check the rebuilt chain ordering against the one the training run printed.

    The ordering search is seeded (`_rng.seed(42)`), so this should always agree --
    which is exactly why it is worth asserting. If it ever stops agreeing, every
    bond dimension measured here is of a different chain from the one that was
    trained, and silently so.

    Only meaningful when the ansatz being rebuilt is the same arm as the run that
    wrote the checkpoint: the ordering search optimises a different objective for
    purified (it reserves two chain slots per reset event), so the arms
    legitimately get different permutations and a cross-mode comparison would
    always "fail". The caller is responsible for only asking when the modes match.

    Returns (ok, detail). A missing stdout.out is not a failure, just unverifiable.
    """
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    stdout_path = os.path.join(run_dir, "stdout.out")
    if not os.path.exists(stdout_path):
        return None, f"no stdout.out beside {checkpoint_path}"
    want = None
    with open(stdout_path, "r", errors="replace") as fh:
        for line in fh:
            if "System qubit order:" in line:
                want = line.split("System qubit order:")[1].strip()
                break
    if want is None:
        return None, "stdout.out has no 'System qubit order:' line"
    got = str([q for q, _ in sorted(ansatz.sys_to_mps.items(), key=lambda kv: kv[1])])
    return (got == want), f"rebuilt {got} vs logged {want}"


def reset_theta_stats(ansatz, all_params, snapshot):
    """Range of the reset thetas at one snapshot, across trials.

    Recorded on every row so a result can be attributed to the reset-theta
    initialisation it came from. The runs reused here initialised near pi/2; the
    working tree now initialises in [0.05, 0.1], which is a different experiment
    and would otherwise be indistinguishable in the output.
    """
    idx = ansatz.reset_param_indices
    if idx is None or not len(idx):
        return {"reset_theta_min": float("nan"), "reset_theta_max": float("nan")}
    thetas = np.asarray(all_params[:, snapshot, :][:, idx], dtype=float)
    return {"reset_theta_min": float(thetas.min()), "reset_theta_max": float(thetas.max())}


def _aggregate(samples):
    """Collapse several trajectory draws of the same snapshot into one row.

    Bond dimension and entries take max / median / min: the max is what a
    simulation would actually have to provision. Energy and discarded weight take
    the mean, since those are estimates of a channel average.
    """
    def col(key):
        return np.asarray([s[key] for s in samples], dtype=float)

    worst = max(samples, key=lambda s: s["max_bond_dim"])
    out = dict(worst)                      # profiles/entropies from the worst draw
    out.update({
        "max_bond_dim": int(col("max_bond_dim").max()),
        "max_bond_dim_median": float(np.median(col("max_bond_dim"))),
        "max_bond_dim_min": int(col("max_bond_dim").min()),
        "mps_entries": int(col("mps_entries").max()),
        "mps_entries_median": float(np.median(col("mps_entries"))),
        "energy": float(col("energy").mean()),
        "energy_std": float(col("energy").std()),
        "energy_normalised": float(col("energy_normalised").mean()),
        "discarded_weight": float(col("discarded_weight").max()),
        "max_entropy_vn": float(col("max_entropy_vn").max()),
        "max_entropy_renyi2": float(col("max_entropy_renyi2").max()),
        "n_draws": len(samples),
        "cap_bound": bool(any(s["cap_bound"] for s in samples)),
    })
    return out


def measure_snapshot(ansatz, params, mode, tol, cap, rng, traj_samples):
    """One (snapshot, trial, mode, tolerance) measurement, sampling if stochastic."""
    if mode == "enum":
        # The enumerated arm has no single representative state -- the mixed state
        # is carried as the ensemble over 3^R Kraus branches, so "the bond
        # dimension of its MPS" is not a well-posed question here. Its branch walk
        # lives in traj_approximation_quality, where it is the exact reference
        # rather than a complexity measurement.
        raise ValueError(
            "mode='enum' cannot be rebuilt as one state; drop it from the rebuild "
            "modes (its checkpoint is still usable as a parameter source)."
        )
    if mode not in ANCILLA_FREE:
        return _aggregate([measure(ansatz, params, mode, tol, cap)])
    # traj / none: `none` has no reset events, so its status vector is empty and
    # every draw builds the identical circuit -- sampling it would cost N times as
    # much for N copies of the same number.
    n = traj_samples if ansatz.total_resets else 1
    samples = [
        measure(ansatz, params, mode, tol, cap,
                status=rng.uniform(size=ansatz.total_resets))
        for _ in range(n)
    ]
    return _aggregate(samples)


def flatten(row):
    """Per-cut lists become pipe-joined strings so the CSV stays one row per measurement."""
    out = {}
    for k, v in row.items():
        out[k] = "|".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in v) \
            if isinstance(v, (list, tuple)) else v
    return out


def write_rows(rows, out, final=False):
    """Rewrite the CSV from scratch after every measurement.

    A full 3x3 sweep is hours of purified rebuilds, so a walltime kill part way
    through must not lose the lot. Rewriting a few thousand rows costs nothing
    next to one rebuild. os.replace is atomic on POSIX, so a kill mid-write leaves
    the previous CSV intact rather than a truncated one. Same pattern as
    bond_dim_requirement.write_rows.
    """
    if not rows:
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    tmp = out + ".partial"
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, out)
    if final:
        print(f"\nWrote {len(rows)} rows to {out}")


def run_sweep(Lx, Ly, checkpoints, args):
    """Every (comparison, param source, rebuild mode) pass, into one row list."""
    rows = []

    # Which passes to run. as-trained: each arm's params in its own mode.
    # same-params: the trajectory run's params in BOTH modes.
    passes = []
    for mode, ck in checkpoints.items():
        passes.append(("as_trained", ck, mode))
    if args.same_params and "traj" in checkpoints:
        for mode in ("traj", "pur"):
            passes.append(("same_params", checkpoints["traj"], mode))

    # Ansatz construction happens under the NORMAL (JAX) backend, before the
    # measurement loop swaps to NumPy. __post_init__ builds the vmapped
    # value_and_grad helpers the optimiser would use, and the NumPy backend
    # refuses outright ("numpy backend doesn't support AD"). Those helpers are
    # never called here; only the structural fields (ordering, claws, reset
    # sites, split_conf) matter, and none of them are backend-specific.
    ansatze = {}
    print("\nbuilding ansaetze (ordering search runs once per mode):")
    for mode in dict.fromkeys(m for _, _, m in passes):
        t0 = time.perf_counter()
        a = build_ansatz(Lx, Ly, mode, nlayers=args.nlayers,
                         reset_layers=args.reset_layers, seed=args.seed,
                         cap=args.cap)
        print(f"  {mode}: chain={a.n_mps_qubits} R={a.total_resets} "
              f"nparams={a.nparams} exact cap={2 ** (a.n_mps_qubits // 2)} "
              f"[{time.perf_counter() - t0:.1f}s ordering search]")
        ansatze[mode] = a

    for comparison, ck, mode in passes:
        ansatz = ansatze[mode]
        all_params = ck["all_params"]

        if all_params.shape[-1] != ansatz.nparams:
            print(f"  SKIP {comparison}/{ck['reset_mode']}->{mode}: checkpoint has "
                  f"{all_params.shape[-1]} parameters, this arm takes {ansatz.nparams}")
            continue

        # Only comparable when the rebuild arm matches the arm that wrote the
        # checkpoint, which on an as_trained pass it does by construction. On the
        # same_params pass the purified ansatz is deliberately being fed the
        # trajectory run's parameters, and the two arms have genuinely different
        # orderings, so there is nothing to check.
        if comparison == "as_trained":
            ok, detail = verify_ordering(ansatz, ck["path"])
            status = {True: "OK", False: "MISMATCH", None: "unverifiable"}[ok]
            print(f"  ordering check ({mode}): {status} -- {detail}")
            if ok is False:
                raise SystemExit(
                    "Rebuilt chain ordering differs from the one the training run used, so "
                    "every bond dimension measured here would be of a different chain. "
                    "Refusing to continue."
                )

        snapshots = list(range(0, ck["filled"], args.snapshot_stride))
        if snapshots[-1] != ck["filled"] - 1:
            snapshots.append(ck["filled"] - 1)   # always include the converged end
        trials = list(range(min(args.trials, all_params.shape[0])))

        print(f"\n  === {comparison}: params from {ck['reset_mode']} (seed {ck['seed']}) "
              f"rebuilt as {mode} | {len(snapshots)} snapshots x {len(trials)} trials "
              f"x {len(args.tolerances)} tolerances ===")
        hdr = (f"  {'step':>6s} {'tr':>3s} {'tol':>8s} {'bd':>5s} {'entries':>9s} "
               f"{'S_vn':>7s} {'disc wt':>10s} {'E':>12s} {'sec':>6s}")

        t_pass = time.perf_counter()
        # Only the rebuilds run on NumPy. Entering per pass rather than around the
        # whole sweep keeps ansatz construction (which needs JAX) outside it.
        with numpy_mps_mode():
            measure_pass(rows, ansatz, ck, comparison, mode, snapshots, trials,
                         hdr, f"{Lx}x{Ly}", args)
        print(f"  pass done in {(time.perf_counter() - t_pass) / 60:.1f} min")

    return rows


def measure_pass(rows, ansatz, ck, comparison, mode, snapshots, trials, hdr,
                 lattice, args):
    """Measure one (comparison, param source, rebuild mode) pass, appending rows.

    Must be called inside numpy_mps_mode. Appends to `rows` and rewrites the CSV
    after each measurement rather than returning at the end, so a walltime kill
    keeps everything measured so far.
    """
    all_params = ck["all_params"]
    for si, snapshot in enumerate(snapshots):
        if si % 10 == 0:
            print(hdr, flush=True)
        theta_stats = reset_theta_stats(ansatz, all_params, snapshot)
        for trial in trials:
            params = np.asarray(all_params[trial, snapshot, :])
            for tol in args.tolerances:
                # Seeded per (snapshot, trial) so the trajectory draws are
                # reproducible and, importantly, are the SAME draws for every
                # tolerance -- otherwise a tolerance comparison would be
                # confounded by having sampled different branches.
                rng = np.random.default_rng(args.traj_seed + 1000 * snapshot + trial)
                t0 = time.perf_counter()
                m = measure_snapshot(ansatz, params, mode, tol, args.cap,
                                     rng, args.traj_samples)
                dt = time.perf_counter() - t0
                row = dict(
                    comparison=comparison,
                    param_mode=ck["reset_mode"], param_seed=ck["seed"],
                    param_checkpoint=ck["path"], param_bond_dim=ck["bond_dim"],
                    rebuild_mode=mode, lattice=lattice,
                    chain=ansatz.n_mps_qubits, total_resets=ansatz.total_resets,
                    exact_cap=2 ** (ansatz.n_mps_qubits // 2),
                    snapshot=snapshot, step=snapshot * ck["stride"], trial=trial,
                    training_energy=(float(ck["all_energies"][trial, snapshot])
                                     if ck["all_energies"] is not None else float("nan")),
                    seconds=dt, **theta_stats, **m,
                )
                rows.append(flatten(row))
                write_rows(rows, args.out)
                print(f"  {row['step']:>6d} {trial:>3d} {tol:>8.0e} "
                      f"{m['max_bond_dim']:>5d} {m['mps_entries']:>9d} "
                      f"{m['max_entropy_vn']:>7.3f} {m['discarded_weight']:>10.2e} "
                      f"{m['energy']:>12.6f} {dt:>6.2f}"
                      + ("  CAP" if m["cap_bound"] else ""), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", default="3x3")
    parser.add_argument("--traj-checkpoint", default=None)
    parser.add_argument("--pur-checkpoint", default=None)
    parser.add_argument("--enum-checkpoint", default=None)
    parser.add_argument("--tolerances", default=",".join(f"{t:g}" for t in DEFAULT_TOLERANCES))
    parser.add_argument("--cap", type=int, default=512,
                        help="safety cap on bond dimension; rows where it binds are "
                             "flagged cap_bound and are ceilings, not measurements")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--snapshot-stride", type=int, default=1)
    parser.add_argument("--traj-samples", type=int, default=8,
                        help="trajectory draws per snapshot; the reported bond dimension "
                             "is the worst of them")
    parser.add_argument("--traj-seed", type=int, default=20260901)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--reset-layers", default="[1]")
    parser.add_argument("--seed", type=int, default=2026,
                        help="ansatz construction seed; only affects initparams, which "
                             "this script does not use -- the ordering search is "
                             "separately seeded and is what has to match")
    parser.add_argument("--no-same-params", dest="same_params", action="store_false",
                        help="skip the controlled comparison (traj params rebuilt in both modes)")
    parser.add_argument("--out", default="outputs/three_resets/complexity_through_training.csv")
    args = parser.parse_args()

    import ast
    args.reset_layers = ast.literal_eval(args.reset_layers)
    args.tolerances = [float(t) for t in args.tolerances.split(",") if t.strip()]
    Lx, Ly = (int(v) for v in args.lattice.split("x"))

    checkpoints = {}
    for mode, path in (("traj", args.traj_checkpoint), ("pur", args.pur_checkpoint),
                       ("enum", args.enum_checkpoint)):
        if path:
            ck = load_checkpoint(path)
            if ck["reset_mode"] == "unknown":
                # Checkpoints written before reset_mode was recorded. The flag it
                # was passed under is the only statement of which arm it is, so
                # adopt that -- otherwise the ordering check and every row's
                # param_mode label would read "unknown" and the check would be
                # skipped exactly on the runs that most need it.
                print(f"  note: checkpoint predates reset_mode being recorded; "
                      f"taking it to be a {mode!r} run, as passed")
                ck["reset_mode"] = mode
            elif ck["reset_mode"] != mode:
                print(f"WARNING: --{mode}-checkpoint points at a run whose "
                      f"reset_mode is {ck['reset_mode']!r}")
            checkpoints[mode] = ck
            print(f"loaded {mode}: {path}\n  seed={ck['seed']} bd={ck['bond_dim']} "
                  f"chain={ck['n_mps_qubits']} R={ck['total_resets']} "
                  f"snapshots={ck['filled']} stride={ck['stride']} "
                  f"params={ck['all_params'].shape}")
    if not checkpoints:
        parser.error("pass at least one of --traj-checkpoint / --pur-checkpoint / --enum-checkpoint")

    print(f"\ntolerances: {args.tolerances} | safety cap: {args.cap} | "
          f"trials: {args.trials} | snapshot stride: {args.snapshot_stride} | "
          f"traj draws: {args.traj_samples}")

    t0 = time.perf_counter()
    rows = run_sweep(Lx, Ly, checkpoints, args)
    write_rows(rows, args.out, final=True)
    print(f"total {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
