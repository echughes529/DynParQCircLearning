"""Cost anatomy of the two reset implementations, at the bond dimension each needs.

Run:
    python -m src.diagnostics.traj_vs_purified_bench --configs 3x3:traj:64,3x3:pur:104

Companion to bond_dim_requirement.py. That script answers "what bond dimension
does each path need"; this one answers "what does that setting cost per
optimisation step, and where does the cost come from".

Each config is given as LxxLy:mode:bond_dim, so the two arms can be benchmarked
at DIFFERENT bond dimensions -- which is the whole point. Comparing both paths at
one bond dimension would flatter whichever path is being under-resourced.

Reported per config:
  - build time (dominated by the random ordering search)
  - first value_and_grad, i.e. JIT compilation
  - warm forward and warm value_and_grad times, median over repeats
  - real SVD/QR call counts through TensorNetwork's MPS gate-splitting path
  - chain length, realised max bond dimension, parameter count
  - energy, gradient norm, and whether anything came out non-finite

The SVD counts are what attribute the cost difference: a shorter chain and the
absence of Toffolis are two separate savings, and only the counts distinguish
them. Read them as a measurement, not a confirmation of a predicted number --
apply_MPO's n-qubit path costs (k-1) SVDs per k-site gate span, so the totals
depend on the actual qubit ordering.
"""

import argparse
import sys
import time
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax
import jax.numpy as jnp

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.svd_call_counting import tap_backends

NLAYERS = 2
RESET_LAYERS = [1]
SEED = 2026
REPEATS = 5


def time_call(fn, *args):
    """Wall-clock of one call, blocking on the result.

    jax dispatches asynchronously: without block_until_ready this would time the
    enqueue, not the computation.
    """
    start = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, time.perf_counter() - start


def bench(Lx, Ly, mode, bond_dim, trials, n_trajectories):
    use_traj = mode == "traj"
    print(f"\n{'=' * 78}")
    print(f"{Lx}x{Ly}  mode={mode}  bond_dim={bond_dim}  trials={trials}")
    print("=" * 78)

    t0 = time.perf_counter()
    ansatz = ToricCodeAnsatz(
        Lx=Lx, Ly=Ly, nlayers=NLAYERS, reset_layers=RESET_LAYERS,
        trials=trials, seed=SEED, bond_dim=bond_dim,
        use_optimal_ordering=True, use_trajectory_resets=use_traj,
        n_trajectories=n_trajectories, normalize_state=True,
        sparse=False, use_mps=True,
    )
    build_s = time.perf_counter() - t0

    params = jnp.asarray(ansatz.initparams)
    rng = np.random.default_rng(7)

    # Both arms are driven through the same vmapped-and-jitted callables the
    # optimiser itself uses, so these numbers are the real per-step cost rather
    # than a hand-built approximation of it.
    if use_traj:
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

    _, compile_fwd_s = time_call(fwd, params)
    (value, grad), compile_vg_s = time_call(vg, params)

    fwd_times, vg_times = [], []
    for _ in range(REPEATS):
        _, dt = time_call(fwd, params)
        fwd_times.append(dt)
        _, dt = time_call(vg, params)
        vg_times.append(dt)

    # Count SVD/QR through a single un-jitted circuit build: inside a compiled
    # program the calls happen at trace time, so counting there would report the
    # traced graph, not the work per step.
    called, called_tn, restore = tap_backends()
    try:
        if use_traj:
            qc, _ = ansatz._circuit_traj(params[0], jnp.asarray(rng.uniform(size=ansatz.total_resets)))
        else:
            qc = ansatz._circuit(params[0])
        max_bd = int(np.max(np.asarray(qc.get_bond_dimensions())))
    finally:
        restore()

    grad_norms = np.linalg.norm(np.asarray(grad), axis=-1)
    row = dict(
        lattice=f"{Lx}x{Ly}", mode=mode, bond_dim=bond_dim, trials=trials,
        n_trajectories=n_trajectories,
        chain=ansatz.n_mps_qubits, nparams=ansatz.nparams,
        total_resets=ansatz.total_resets, max_bond_dim=max_bd,
        build_seconds=build_s,
        ordering_search_seconds=getattr(ansatz, "ordering_search_seconds", float("nan")),
        compile_forward_seconds=compile_fwd_s,
        compile_value_and_grad_seconds=compile_vg_s,
        warm_forward_seconds=float(np.median(fwd_times)),
        warm_value_and_grad_seconds=float(np.median(vg_times)),
        warm_vg_per_trial_seconds=float(np.median(vg_times)) / trials,
        svd_calls=called_tn["svd"], qr_calls=called_tn["qr"],
        svd_calls_tcbackend=called["svd"], qr_calls_tcbackend=called["qr"],
        mean_energy=float(np.mean(np.asarray(value))),
        mean_grad_norm=float(np.mean(grad_norms)),
        all_finite=bool(np.all(np.isfinite(np.asarray(value))) and np.all(np.isfinite(np.asarray(grad)))),
        device=str(jax.devices()),
    )

    print(f"  chain={row['chain']} sites, max bd reached={max_bd}, nparams={row['nparams']}")
    print(f"  build {build_s:.1f}s (ordering search {row['ordering_search_seconds']:.1f}s)")
    print(f"  compile: forward {compile_fwd_s:.1f}s, value_and_grad {compile_vg_s:.1f}s")
    print(f"  warm:    forward {row['warm_forward_seconds']:.4f}s, "
          f"value_and_grad {row['warm_value_and_grad_seconds']:.4f}s "
          f"({row['warm_vg_per_trial_seconds']:.4f}s per trial)")
    print(f"  SVD calls (TN backend): {called_tn['svd']}, QR calls: {called_tn['qr']}")
    print(f"  energy {row['mean_energy']:.6f}, |grad| {row['mean_grad_norm']:.4e}, "
          f"all finite: {row['all_finite']}")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", required=True,
                        help="comma-separated LxxLy:mode:bond_dim, "
                             "e.g. 3x3:traj:64,3x3:pur:104")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--n-trajectories", type=int, default=1)
    parser.add_argument("--out", default="outputs/traj_vs_purified/bench.csv")
    args = parser.parse_args()

    print(f"jax devices: {jax.devices()}")
    rows = []
    for spec in args.configs.split(","):
        lattice, mode, bd = spec.strip().split(":")
        Lx, Ly = (int(v) for v in lattice.split("x"))
        rows.append(bench(Lx, Ly, mode, int(bd), args.trials, args.n_trajectories))

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"{'config':>20} {'chain':>6} {'bd':>5} {'warm vg/step':>13} {'per trial':>11} {'SVDs':>6}")
    for r in rows:
        print(f"{r['lattice'] + ' ' + r['mode']:>20} {r['chain']:>6} {r['bond_dim']:>5} "
              f"{r['warm_value_and_grad_seconds']:>13.4f} {r['warm_vg_per_trial_seconds']:>11.4f} "
              f"{r['svd_calls']:>6}")

    import csv
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
