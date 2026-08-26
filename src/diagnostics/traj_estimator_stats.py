"""How noisy is the trajectory gradient, and does the noise matter?

Run:
    python -m src.diagnostics.traj_estimator_stats
    python -m src.diagnostics.traj_estimator_stats --lattice 3x2 --samples 400

The trajectory path buys a shorter MPS chain by sampling one branch of the reset
channel, which makes the energy and gradient stochastic. bond_dim_requirement.py
measures what that buys; this measures what it costs.

Everything is anchored to an EXACT reference: at 2x2 and 3x2 both paths fit in a
bond dimension where truncation is provably zero, so the purified path's gradient
is the true channel gradient with no approximation anywhere. The trajectory
estimator is then compared against it.

Reported, at each training stage and each n_trajectories:
  - bias:      does the sample mean match the exact gradient (it must -- the
               estimator is unbiased, and a failure here is a bug, not noise)
  - SNR:       |exact gradient| / per-sample standard deviation, per parameter,
               called out separately for the reset thetas, which only have a
               gradient at all because of the DiCE score term
  - direction: cosine similarity between a single-sample gradient and the exact
               one. This is what Adam actually consumes, and it is the number
               that decides whether training still descends
  - variance vs n_trajectories: should fall as 1/n; anything slower means paying
               n times the cost for less than n times the noise reduction
  - baseline:  whether subtracting the control variate actually helps

The interesting failure mode this is built to catch: SNR collapsing near
convergence, where the true gradient is small but the sampling noise is not. If
that happens, "more variance is fine as long as it gets there" stops being true
at exactly the point that matters.
"""

import argparse
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax
import jax.numpy as jnp
import optax
import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz

K = tc.set_backend("jax")

NLAYERS = 2
RESET_LAYERS = [1]
SEED = 4321
# Bond dimensions at which truncation is provably zero (2^floor(n_sites/2)), so
# the reference gradient is exact rather than approximately exact. The enumerated
# arm shares the trajectory arm's bare nq-site chain, hence the same figure.
EXACT_BD = {
    "2x2": {"traj": 4, "enum": 4, "pur": 8},
    "3x2": {"traj": 8, "enum": 8, "pur": 32},
}


def make_arms(lattice):
    """One ansatz per reset implementation, over a shared parameter layout.

    All three are built at a bond dimension where truncation is provably zero,
    so the two exact arms really are exact and any disagreement between them
    would be a bug rather than a truncation artefact.
    """
    Lx, Ly = (int(v) for v in lattice.split("x"))
    common = dict(Lx=Lx, Ly=Ly, nlayers=NLAYERS, reset_layers=RESET_LAYERS,
                  trials=1, seed=SEED, use_optimal_ordering=False,
                  normalize_state=True, sparse=False, use_mps=True)
    arms = {
        "pur": ToricCodeAnsatz(use_trajectory_resets=False,
                               bond_dim=EXACT_BD[lattice]["pur"], **common),
        "traj": ToricCodeAnsatz(use_trajectory_resets=True,
                                bond_dim=EXACT_BD[lattice]["traj"], **common),
        "enum": ToricCodeAnsatz(use_trajectory_resets=False, use_enumerated_resets=True,
                                max_reset_branches=3 ** 8,
                                bond_dim=EXACT_BD[lattice]["enum"], **common),
    }
    counts = {k: a.nparams for k, a in arms.items()}
    assert len(set(counts.values())) == 1, f"parameter layouts must match: {counts}"
    return arms


def make_pair(lattice):
    """Backwards-compatible (purified, trajectory) accessor."""
    arms = make_arms(lattice)
    return arms["pur"], arms["traj"]


def exact_grad_fn(exact_arm):
    """Energy and gradient of the exact reset channel (no sampling involved).

    Works for either exact arm -- purified or enumerated. They compute the same
    function by different routes, which is what cross_check_exact_arms verifies.
    """
    return jax.jit(jax.value_and_grad(exact_arm.energy_from_params))


def cross_check_exact_arms(arms, params):
    """Do the two exact arms agree, in energy AND gradient, at exact bond dim?

    This is the load-bearing assumption of the whole three-way comparison: the
    purified circuit records which branch was taken in two ancillas, while the
    enumerated one keeps the branches as separate states and sums their
    contributions. Those are the same channel, so at a bond dimension where
    neither truncates they must agree to floating-point precision. If they do
    not, no downstream cost or accuracy number means anything.
    """
    out = {}
    for name in ("pur", "enum"):
        e, g = exact_grad_fn(arms[name])(params)
        out[name] = (float(e), np.asarray(g))
    de = abs(out["pur"][0] - out["enum"][0])
    gp, ge = out["pur"][1], out["enum"][1]
    dg = float(np.max(np.abs(gp - ge)))
    scale = max(float(np.max(np.abs(gp))), 1e-30)
    cos = float(np.dot(gp, ge) / (np.linalg.norm(gp) * np.linalg.norm(ge) + 1e-300))
    print(f"\n  exact-arm cross-check (purified vs enumerated, both at exact bond dim):")
    print(f"    |dE|                  {de:.3e}")
    print(f"    max |d(grad)|         {dg:.3e}  ({dg / scale:.2e} relative to |grad|_max)")
    print(f"    cosine(grad_p, grad_e) {cos:.12f}")
    ok = de < 1e-9 and dg / scale < 1e-7
    print(f"    -> {'AGREE' if ok else 'DISAGREE -- investigate before trusting anything else'}")
    return dict(cross_check_dE=de, cross_check_dgrad=dg,
                cross_check_dgrad_rel=dg / scale, cross_check_cos=cos,
                cross_check_ok=bool(ok))


def sample_estimator(traj, params, n_samples, n_trajectories, baseline, seed):
    """Draw n_samples independent estimates of the energy and its gradient."""
    traj.n_trajectories = n_trajectories
    # Built once, outside the loop: jax.jit caches on the function object, and a
    # bound method is a fresh object on every attribute access, so building it
    # inside would recompile on every sample.
    fn = jax.jit(jax.value_and_grad(traj.dice_energy_from_params, argnums=0))
    key = jax.random.PRNGKey(seed)
    values, grads = [], []
    for _ in range(n_samples):
        key, sub = jax.random.split(key)
        status = jax.random.uniform(sub, (n_trajectories, traj.total_resets))
        v, g = fn(params, status, baseline)
        values.append(float(v))
        grads.append(np.asarray(g))
    return np.asarray(values), np.asarray(grads)


def summarise(exact_g, grads, reset_idx, label, exact_e, values, rows, **meta):
    mean_g = grads.mean(axis=0)
    std_g = grads.std(axis=0, ddof=1)
    n = len(grads)
    # Standard error of the mean: the yardstick for whether an apparent bias is
    # real or just the finite sample talking.
    sem = std_g / np.sqrt(n)

    # Late in training the reset angle saturates (p = sin^2(theta/2) -> 0 or 1)
    # and every sample takes the same branch, so the spread collapses to
    # round-off. Dividing by it would report z-scores of 1e12 and SNRs of 1e14,
    # which say nothing except "the denominator was zero". Treat that case as
    # what it is -- a degenerate sampler -- rather than as spectacular accuracy.
    DEGENERATE = 1e-12
    degenerate = float(np.linalg.norm(std_g)) < DEGENERATE
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sem > DEGENERATE, np.abs(mean_g - exact_g) / sem, np.nan)
        snr = np.where(std_g > DEGENERATE, np.abs(exact_g) / std_g, np.nan)

    exact_norm = np.linalg.norm(exact_g)
    cos = grads @ exact_g / (np.linalg.norm(grads, axis=1) * exact_norm)

    reset_mask = np.zeros(len(exact_g), dtype=bool)
    reset_mask[reset_idx] = True

    e_sem = values.std(ddof=1) / np.sqrt(n)
    finite_snr = snr[np.isfinite(snr)]

    # A sampler whose typical draw is noiseless can still be wrong rarely and
    # badly: once the branch probability saturates, the off-branch is drawn only
    # occasionally but carries a large deviation. Standard deviation alone
    # averages that away, so record the tail directly.
    e_dev = np.abs(values - np.median(values))
    row = dict(
        label=label, n_samples=n, degenerate_sampler=degenerate,
        energy_exact=float(exact_e), energy_mean=float(values.mean()),
        energy_sem=float(e_sem),
        energy_z=float(abs(values.mean() - exact_e) / e_sem) if e_sem > DEGENERATE else float("nan"),
        energy_frac_outliers=float(np.mean(e_dev > 10 * np.median(e_dev) + 1e-9)),
        energy_max_dev=float(e_dev.max()),
        grad_worst_z=float(np.nanmax(z)) if np.any(np.isfinite(z)) else float("nan"),
        grad_norm_exact=float(exact_norm),
        grad_noise_norm=float(np.linalg.norm(std_g)),
        snr_median=float(np.median(finite_snr)) if finite_snr.size else float("nan"),
        snr_reset_median=float(np.nanmedian(snr[reset_mask])),
        snr_cartan_median=float(np.nanmedian(snr[~reset_mask])),
        cos_mean=float(cos.mean()), cos_std=float(cos.std(ddof=1)),
        cos_p10=float(np.percentile(cos, 10)),
        frac_cos_positive=float(np.mean(cos > 0)),
        **meta,
    )
    rows.append(row)

    note = "   [sampler degenerate: every draw took the same branch]" if degenerate else ""
    print(f"    {label:<28} E {values.mean():>9.5f} vs {exact_e:>9.5f} "
          f"(|z|={row['energy_z']:.2f})  |g|={exact_norm:.3e} noise={row['grad_noise_norm']:.3e}{note}")
    print(f"    {'':<28} SNR med={row['snr_median']:.3f} (reset {row['snr_reset_median']:.3f}, "
          f"cartan {row['snr_cartan_median']:.3f})  cos={cos.mean():.3f}+-{cos.std(ddof=1):.3f} "
          f"p10={row['cos_p10']:.3f}  outlier frac={row['energy_frac_outliers']:.3f}")
    return row


def train_snapshots(traj, stages, lr=0.01, seed=99):
    """Trajectory-mode training, returning parameters at the requested steps.

    Gradient noise is not a fixed property of the estimator -- it depends on
    where in parameter space the optimiser is -- so the statistics have to be
    taken at states training actually visits, not only at a random start.
    """
    params = jnp.asarray(traj.initparams)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)
    key = jax.random.PRNGKey(seed)
    status_shape = (traj.trials, traj.n_trajectories, traj.total_resets)
    baseline = jnp.zeros(traj.trials)

    snapshots = {}
    for step in range(max(stages) + 1):
        if step in stages:
            snapshots[step] = params[0]
        key, sub = jax.random.split(key)
        status = jax.random.uniform(sub, status_shape)
        value, grad = traj._cost_vvag(params, status, baseline)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        baseline = 0.9 * baseline + 0.1 * value
    return snapshots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", default="2x2", choices=sorted(EXACT_BD))
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--stages", default="0,100,400",
                        help="training steps at which to take the statistics")
    parser.add_argument("--n-trajectories", default="1,2,4,8",
                        help="comma-separated values to compare")
    parser.add_argument("--ref-mode", default="pur", choices=("pur", "enum"),
                        help="which exact arm supplies the reference gradient")
    parser.add_argument("--out", default="outputs/three_resets/estimator_stats.csv")
    args = parser.parse_args()

    stages = [int(v) for v in args.stages.split(",")]
    n_trajs = [int(v) for v in args.n_trajectories.split(",")]

    print(f"jax devices: {jax.devices()}")
    arms = make_arms(args.lattice)
    purified, traj, enum = arms["pur"], arms["traj"], arms["enum"]
    print(f"\n{args.lattice}: traj/enum chain={traj.n_mps_qubits} at bd={traj.bond_dim} (exact), "
          f"purified chain={purified.n_mps_qubits} at bd={purified.bond_dim} (exact), "
          f"resets={traj.total_resets} -> {3 ** traj.total_resets} branches, "
          f"nparams={traj.nparams}")

    reset_idx = np.asarray(traj.reset_param_indices)
    # Which exact arm supplies the reference. They are the same function -- the
    # cross-check below is what establishes that -- so this only changes the
    # route taken to it, not the answer.
    grad_fn = exact_grad_fn(arms[args.ref_mode])
    cross = cross_check_exact_arms(arms, jnp.asarray(traj.initparams[0]))

    print(f"\nTraining to collect snapshots at steps {stages} ...")
    traj.n_trajectories = 1
    snapshots = train_snapshots(traj, set(stages))

    rows = []
    for step in stages:
        params = snapshots[step]
        exact_e, exact_g = grad_fn(params)
        exact_e, exact_g = float(exact_e), np.asarray(exact_g)
        # The reset probabilities explain the sampler's behaviour directly: as
        # p approaches 0 or 1 the branch stops being random and the estimator
        # stops being noisy, whatever the sample count says.
        thetas = np.asarray(params)[reset_idx]
        p_reset = np.sin(thetas / 2.0) ** 2
        print(f"\n  === step {step}: exact channel energy {exact_e:.6f} | "
              f"reset thetas {np.array2string(thetas, precision=4)} "
              f"-> p {np.array2string(p_reset, precision=4)} ===")

        for n_traj in n_trajs:
            values, grads = sample_estimator(traj, params, args.samples, n_traj,
                                             baseline=jnp.asarray(exact_e), seed=1000 + step)
            summarise(exact_g, grads, reset_idx, f"n_traj={n_traj}, baseline=E",
                      exact_e, values, rows, step=step, n_trajectories=n_traj,
                      baseline="exact_energy", lattice=args.lattice,
                      ref_mode=args.ref_mode, **cross)

        # Baseline ablation: the control variate should shrink the variance
        # without moving the mean. Setting it to 0 scales the score term by the
        # full energy instead of the deviation from it.
        values, grads = sample_estimator(traj, params, args.samples, 1,
                                         baseline=jnp.asarray(0.0), seed=1000 + step)
        summarise(exact_g, grads, reset_idx, "n_traj=1, baseline=0",
                  exact_e, values, rows, step=step, n_trajectories=1,
                  baseline="zero", lattice=args.lattice,
                  ref_mode=args.ref_mode, **cross)

    print(f"\n{'=' * 78}\nVARIANCE SCALING IN n_trajectories\n{'=' * 78}")
    print(f"{'step':>6} {'n_traj':>7} {'noise norm':>12} {'vs n=1':>9} {'ideal 1/sqrt(n)':>16} {'cos mean':>9}")
    for step in stages:
        base = next(r for r in rows if r["step"] == step and r["n_trajectories"] == 1
                    and r["baseline"] == "exact_energy")
        for n_traj in n_trajs:
            r = next(x for x in rows if x["step"] == step and x["n_trajectories"] == n_traj
                     and x["baseline"] == "exact_energy")
            ratio = r["grad_noise_norm"] / base["grad_noise_norm"]
            print(f"{step:>6} {n_traj:>7} {r['grad_noise_norm']:>12.4e} {ratio:>9.3f} "
                  f"{1 / np.sqrt(n_traj):>16.3f} {r['cos_mean']:>9.3f}")

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
