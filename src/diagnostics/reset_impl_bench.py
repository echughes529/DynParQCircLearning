"""Cost and memory anatomy of the three reset implementations.

Run:
    python -m src.diagnostics.reset_impl_bench --configs 3x3:traj:64,3x3:pur:104,3x3:enum:64
    python -m src.diagnostics.reset_impl_bench --configs 4x3:enum:128 --trial-ladder 10,4,2,1

Companion to bond_dim_requirement.py. That script answers "what bond dimension
does each path need"; this one answers "what does that setting cost per
optimisation step and in peak memory, and where does the cost come from".

Each config is given as LxxLy:mode:bond_dim, so the arms can be benchmarked at
DIFFERENT bond dimensions -- which is the whole point. Comparing all three at one
bond dimension would flatter whichever is being under-resourced.

Reported per config:
  - build time (dominated by the random ordering search)
  - first forward and first value_and_grad, i.e. JIT compilation. This matters
    on its own for the enumerated arm, which compiles a 3^R-wide vmap.
  - warm forward and warm value_and_grad times, median over repeats
  - peak device bytes after the forward pass and after value_and_grad; the
    difference is what the backward pass's tape costs
  - real SVD/QR call counts through TensorNetwork's MPS gate-splitting path
  - chain length, realised max bond dimension, parameter count, branch count
  - energy, gradient norm, and whether anything came out non-finite

WHY EACH CONFIG RUNS IN ITS OWN PROCESS
---------------------------------------
`peak_bytes_in_use` is a monotone high-water mark over the whole process: it
never falls when memory is freed. Benchmarking the enumerated arm (tens of GB)
and then the trajectory arm (about one GB) in one process would report the
enumerated peak for both. So by default the parent re-executes itself once per
config and collects the rows. --no-isolate keeps everything in one process,
which is faster but only safe for a single config or in ascending memory order.

Isolation also makes out-of-memory a recordable outcome rather than a crash:
the child dies, the parent writes a row with oom=True and the stage it reached,
and the sweep carries on. That is exactly what locates the feasibility frontier
of the enumerated arm, whose cost is 3^R.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax
import jax.numpy as jnp

from src.find_gs import _device_memory_bytes, _host_peak_rss_bytes
from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.svd_call_counting import tap_backends

NLAYERS = 2
RESET_LAYERS = [1]
SEED = 2026
REPEATS = 5
MODES = ("traj", "pur", "enum")


def time_call(fn, *args):
    """Wall-clock of one call, blocking on the result.

    jax dispatches asynchronously: without block_until_ready this would time the
    enqueue, not the computation.
    """
    start = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, time.perf_counter() - start


def build_ansatz(Lx, Ly, mode, bond_dim, trials, n_trajectories, branch_chunk_size=None):
    return ToricCodeAnsatz(
        Lx=Lx, Ly=Ly, nlayers=NLAYERS, reset_layers=RESET_LAYERS,
        trials=trials, seed=SEED, bond_dim=bond_dim,
        use_optimal_ordering=True,
        use_trajectory_resets=(mode == "traj"),
        use_enumerated_resets=(mode == "enum"),
        # 3^R: 81 at 3x3, 729 at 4x3. The default guard of 512 would refuse 4x3,
        # and refusing it is the wrong answer here -- whether it fits is the
        # measurement.
        max_reset_branches=3 ** 8,
        branch_chunk_size=branch_chunk_size,
        n_trajectories=n_trajectories, normalize_state=True,
        sparse=False, use_mps=True,
    )


def bench(Lx, Ly, mode, bond_dim, trials, n_trajectories, branch_chunk_size=None):
    """Measure one config in this process. Returns a row dict."""
    print(f"\n{'=' * 78}")
    print(f"{Lx}x{Ly}  mode={mode}  bond_dim={bond_dim}  trials={trials}")
    print("=" * 78)

    row = dict(lattice=f"{Lx}x{Ly}", mode=mode, bond_dim=bond_dim, trials=trials,
               n_trajectories=n_trajectories, branch_chunk_size=branch_chunk_size,
               oom=False, failed_stage="", error="")
    stage = "build"

    t0 = time.perf_counter()
    ansatz = build_ansatz(Lx, Ly, mode, bond_dim, trials, n_trajectories, branch_chunk_size)
    row["build_seconds"] = time.perf_counter() - t0
    row["chain"] = ansatz.n_mps_qubits
    row["nparams"] = ansatz.nparams
    row["total_resets"] = ansatz.total_resets
    row["nancillas"] = ansatz.nancillas
    row["nbranches"] = int(3 ** ansatz.total_resets) if mode == "enum" else 1
    row["ordering_search_seconds"] = float(getattr(ansatz, "ordering_search_seconds", np.nan))

    params = jnp.asarray(ansatz.initparams)
    rng = np.random.default_rng(7)

    # All three arms are driven through the same vmapped-and-jitted callables the
    # optimiser itself uses, so these are the real per-step costs rather than a
    # hand-built approximation. Only the trajectory arm takes extra positional
    # arguments; enumerated and purified share the plain single-argument form.
    if mode == "traj":
        status_shape = (trials, n_trajectories, ansatz.total_resets)
        baseline = jnp.zeros(trials)

        def fwd(p):
            return ansatz._costs_vmapped(p, jnp.asarray(rng.uniform(size=status_shape)), baseline)

        def vg(p):
            return ansatz._cost_vvag(p, jnp.asarray(rng.uniform(size=status_shape)), baseline)
    else:
        def fwd(p):
            return ansatz._costs_vmapped(p)

        def vg(p):
            return ansatz._cost_vvag(p)

    try:
        stage = "compile_forward"
        _, row["compile_forward_seconds"] = time_call(fwd, params)
        _, row["peak_bytes_after_forward"] = _device_memory_bytes()

        stage = "warm_forward"
        fwd_times = [time_call(fwd, params)[1] for _ in range(REPEATS)]
        row["warm_forward_seconds"] = float(np.median(fwd_times))

        stage = "compile_value_and_grad"
        (value, grad), row["compile_value_and_grad_seconds"] = time_call(vg, params)
        _, row["peak_bytes_after_vg"] = _device_memory_bytes()

        stage = "warm_value_and_grad"
        vg_times = [time_call(vg, params)[1] for _ in range(REPEATS)]
        row["warm_value_and_grad_seconds"] = float(np.median(vg_times))
        row["warm_vg_per_trial_seconds"] = row["warm_value_and_grad_seconds"] / trials
    except Exception as exc:
        # Device OOM surfaces as an XlaRuntimeError carrying RESOURCE_EXHAUSTED.
        # It is data, not a failure: it is where this arm stops being usable.
        text = f"{type(exc).__name__}: {exc}"
        row["oom"] = "RESOURCE_EXHAUSTED" in text or "out of memory" in text.lower()
        row["failed_stage"] = stage
        row["error"] = text[:400]
        row["host_rss_bytes"] = _host_peak_rss_bytes()
        print(f"  FAILED at {stage}: {text[:400]}")
        return row

    # Count SVD/QR through a single un-jitted circuit build: inside a compiled
    # program the calls happen at trace time, so counting there would report the
    # traced graph, not the work per step.
    called, called_tn, restore = tap_backends()
    try:
        if mode == "traj":
            qc, _ = ansatz._circuit_traj(params[0], jnp.asarray(rng.uniform(size=ansatz.total_resets)))
        elif mode == "enum":
            # One branch, not all 3^R: this counts the per-branch gate-splitting
            # work, and the branch multiplicity is reported separately as
            # nbranches. Multiplying the two gives the per-step total, which is
            # the honest way to attribute the cost.
            qc = ansatz._circuit_branch(params[0], ansatz._reset_branches[0])
        else:
            qc = ansatz._circuit(params[0])
        row["max_bond_dim"] = int(np.max(np.asarray(qc.get_bond_dimensions())))
    finally:
        restore()

    row["svd_calls"] = called_tn["svd"]
    row["qr_calls"] = called_tn["qr"]
    row["svd_calls_tcbackend"] = called["svd"]
    row["qr_calls_tcbackend"] = called["qr"]
    row["svd_calls_per_step"] = called_tn["svd"] * row["nbranches"]
    row["host_rss_bytes"] = _host_peak_rss_bytes()

    grad_norms = np.linalg.norm(np.asarray(grad), axis=-1)
    row["mean_energy"] = float(np.mean(np.asarray(value)))
    row["mean_grad_norm"] = float(np.mean(grad_norms))
    row["all_finite"] = bool(np.all(np.isfinite(np.asarray(value)))
                             and np.all(np.isfinite(np.asarray(grad))))
    row["device"] = str(jax.devices())

    gib = 2 ** 30
    print(f"  chain={row['chain']} sites ({row['nancillas']} ancilla), "
          f"max bd reached={row['max_bond_dim']}, nparams={row['nparams']}, "
          f"branches={row['nbranches']}")
    print(f"  build {row['build_seconds']:.1f}s "
          f"(ordering search {row['ordering_search_seconds']:.1f}s)")
    print(f"  compile: forward {row['compile_forward_seconds']:.1f}s, "
          f"value_and_grad {row['compile_value_and_grad_seconds']:.1f}s")
    print(f"  warm:    forward {row['warm_forward_seconds']:.4f}s, "
          f"value_and_grad {row['warm_value_and_grad_seconds']:.4f}s "
          f"({row['warm_vg_per_trial_seconds']:.4f}s per trial)")
    print(f"  peak device: after forward {row['peak_bytes_after_forward'] / gib:.2f} GiB, "
          f"after value_and_grad {row['peak_bytes_after_vg'] / gib:.2f} GiB "
          f"(backward tape adds "
          f"{(row['peak_bytes_after_vg'] - row['peak_bytes_after_forward']) / gib:.2f} GiB)")
    print(f"  host RSS peak: {row['host_rss_bytes'] / gib:.2f} GiB")
    print(f"  SVD calls per branch (TN backend): {row['svd_calls']}, "
          f"per step: {row['svd_calls_per_step']}, QR: {row['qr_calls']}")
    print(f"  energy {row['mean_energy']:.6f}, |grad| {row['mean_grad_norm']:.4e}, "
          f"all finite: {row['all_finite']}")
    return row


def parse_config(spec):
    lattice, mode, bd = spec.strip().split(":")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    Lx, Ly = (int(v) for v in lattice.split("x"))
    return Lx, Ly, mode, int(bd)


def run_isolated(spec, args):
    """Run one config in a fresh process and return its row.

    A dead child is a result, not an error: a host-OOM kill shows up as a
    negative return code and a device OOM as a non-zero one, and either way the
    row records how far this config got.
    """
    row_path = f"{args.out}.row.json"
    cmd = [sys.executable, "-m", "src.diagnostics.reset_impl_bench",
           "--single", spec, "--row-out", row_path,
           "--trials", str(args.trials),
           "--n-trajectories", str(args.n_trajectories),
           "--out", args.out]
    if args.branch_chunk_size is not None:
        cmd += ["--branch-chunk-size", str(args.branch_chunk_size)]
    if os.path.exists(row_path):
        os.remove(row_path)

    print(f"\n>>> isolated run: {spec} (trials={args.trials})", flush=True)
    proc = subprocess.run(cmd, cwd=os.getcwd())
    if os.path.exists(row_path):
        with open(row_path) as fh:
            row = json.load(fh)
        os.remove(row_path)
        return row

    Lx, Ly, mode, bd = parse_config(spec)
    killed_by_signal = proc.returncode < 0
    print(f"  child exited {proc.returncode} without writing a row "
          f"({'killed by signal -- host OOM is the usual cause' if killed_by_signal else 'device OOM or crash'})")
    return dict(lattice=f"{Lx}x{Ly}", mode=mode, bond_dim=bd, trials=args.trials,
                n_trajectories=args.n_trajectories,
                branch_chunk_size=args.branch_chunk_size,
                oom=True, failed_stage="process_died",
                error=f"child returncode {proc.returncode}"
                      f"{' (signal)' if killed_by_signal else ''}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs",
                        help="comma-separated LxxLy:mode:bond_dim, "
                             "e.g. 3x3:traj:64,3x3:pur:104,3x3:enum:64")
    parser.add_argument("--single", help=argparse.SUPPRESS)
    parser.add_argument("--row-out", help=argparse.SUPPRESS)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--trial-ladder",
                        help="comma-separated trial counts to try per config, "
                             "descending, stopping at the first that fits. This is "
                             "the feasibility probe for the enumerated arm.")
    parser.add_argument("--n-trajectories", type=int, default=1)
    parser.add_argument("--branch-chunk-size", type=int, default=None)
    parser.add_argument("--no-isolate", action="store_true",
                        help="run every config in this process. Faster, but peak "
                             "memory is a process-wide high-water mark, so only "
                             "safe for one config or in ascending memory order.")
    parser.add_argument("--out", default="outputs/three_resets/bench.csv")
    args = parser.parse_args()

    # Child mode: measure exactly one config and hand the row back as JSON.
    if args.single:
        Lx, Ly, mode, bd = parse_config(args.single)
        row = bench(Lx, Ly, mode, bd, args.trials, args.n_trajectories,
                    args.branch_chunk_size)
        if args.row_out:
            with open(args.row_out, "w") as fh:
                json.dump(row, fh)
        return

    if not args.configs:
        parser.error("--configs is required")

    print(f"jax devices: {jax.devices()}")
    print(f"XLA_PYTHON_CLIENT_PREALLOCATE="
          f"{os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'unset')}")

    ladder = ([int(v) for v in args.trial_ladder.split(",")]
              if args.trial_ladder else [args.trials])

    rows = []
    for spec in args.configs.split(","):
        for trials in ladder:
            args.trials = trials
            if args.no_isolate:
                Lx, Ly, mode, bd = parse_config(spec)
                row = bench(Lx, Ly, mode, bd, trials, args.n_trajectories,
                            args.branch_chunk_size)
            else:
                row = run_isolated(spec, args)
            rows.append(row)
            if not row.get("oom") and not row.get("failed_stage"):
                # Ladder is descending: the first size that fits is the largest
                # that fits, so there is nothing to learn from smaller ones.
                break
            print(f"  {spec} did not fit at trials={trials}"
                  + (f"; retrying smaller" if trials != ladder[-1] else "; ladder exhausted"))

    gib = 2 ** 30
    print(f"\n{'=' * 96}\nSUMMARY\n{'=' * 96}")
    print(f"{'config':>18} {'trials':>6} {'chain':>6} {'bd':>5} {'branch':>7} "
          f"{'warm vg':>9} {'/trial':>8} {'peak GiB':>9} {'SVD/step':>9}")
    for r in rows:
        tag = f"{r['lattice']} {r['mode']}"
        if r.get("oom") or r.get("failed_stage"):
            print(f"{tag:>18} {r['trials']:>6} {'':>6} {r['bond_dim']:>5} {'':>7} "
                  f"{'OOM at ' + r['failed_stage']:>9}")
            continue
        print(f"{tag:>18} {r['trials']:>6} {r['chain']:>6} {r['bond_dim']:>5} "
              f"{r['nbranches']:>7} {r['warm_value_and_grad_seconds']:>9.4f} "
              f"{r['warm_vg_per_trial_seconds']:>8.4f} "
              f"{r['peak_bytes_after_vg'] / gib:>9.2f} {r['svd_calls_per_step']:>9}")

    import csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
