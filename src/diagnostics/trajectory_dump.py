"""Dump the complete trajectory distribution at a few points during training.

Run:
    python -m src.diagnostics.trajectory_dump --lattice 2x2
    python -m src.diagnostics.trajectory_dump --lattice 3x3 --reset-layers 0

One ToricCodeAnsatz trains with enumerated resets (every trajectory evaluated
exactly at every step). At 0, 20, 40, 60, 80 and 100% of training a read-only
callback writes, for every trial, one row per trajectory i = ab (notation of the
"Trajectory Kraus Operators" note):

    a, |a|, b, p^(a), p^(b|a), p^(ab), E^(ab)

to <out>/reset_layers_<k>/<lattice>/trial<t>_step<s>.npz, with run metadata in a matching .json.
"step s" means the parameters at the start of step s; step maxiter is the final
parameters.

How the numbers are obtained. trajectory_table returns <phi|phi> = p^(ab) and
<phi|H|phi> = p^(ab) E^(ab), with sigma built into each reset. Dividing by p^(a)
to get p^(b|a) fails once sigma -> 0, where p^(a) of a fired pattern can be
~1e-16. But p^(b|a) and E^(ab) do not depend on sigma (note eq. 3.9), so the
table is evaluated a second time on a copy of the parameters with sigma = pi/2,
where p^(a) = 2^-R for every pattern:

    p^(b|a) = <phi|phi> * 2^R,    E^(ab) = <phi|H|phi> / <phi|phi>

p^(a) then comes from the formula at the real sigma and p^(ab) = p^(a) p^(b|a).
The real-sigma table is the independent reference for the checks.

Checks (the script raises at the first failure, after writing that dump's files):
    1. sum p^(ab) = 1, and sum_b p^(b|a) = 1 for every a
    2. p^(a) from the formula matches sum_b of the real-sigma table; p^(ab)
       matches the real-sigma table row by row (over trajectories with
       p^(a) >= P_KEEP; the rest are skipped at the real sigma, see P_KEEP)
    3. 2^|a| trajectories per pattern, 3^R in total
    4. sum p^(ab) E^(ab) matches the real-sigma energy, within the skipped
       weight times E_UB
    5. after training: 500 single-trajectory samples at the saved point with the
       largest variance reproduce its mean and variance, and every sample is one
       of the saved E^(ab)
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
import tensorcircuit as tc

from src.examples import run_config as cfg
from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.single_traj_error import trajectory_labels

K = tc.set_backend("jax")

# Lossless caps for the ancilla-free chain: 2^floor(n/2) for the 4- and 7-qubit
# chains; 64 at 3x3 with optimal ordering.
BOND_DIM = {"2x2": 4, "3x2": 8, "3x3": 64}
FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Below this a trajectory has no state to speak of (note section 3.3.3).
Q_FLOOR = 1e-12
# The real-sigma reference evaluation skips every trajectory whose pattern
# probability p^(a) is below this. Once sigma ~ 1e-17 the fired-reset
# trajectories carry sin(sigma/2) ~ 1e-17 in their Kraus operator, so their MPS
# tensors are almost zero; splitting them NaN'd the GPU evaluation (job 3692122,
# 3x3 step 600). Their total weight is computed exactly, recorded, and allowed
# for in checks 2 and 4. The saved numbers never touch them: they come from the
# sigma = pi/2 evaluation and the p^(a) formula.
P_KEEP = 1e-12
TOL_P = 1e-10
TOL_E = 1e-8
N_SIGMA = 4.0


def git_state():
    def run(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
    return run("git", "rev-parse", "HEAD"), bool(run("git", "status", "--porcelain", "--untracked-files=no"))


def build_ansatz(lattice, enumerated, trials, maxiter, seed, reset_layers):
    Lx, Ly = (int(v) for v in lattice.split("x"))
    return ToricCodeAnsatz(
        Lx=Lx, Ly=Ly, nlayers=cfg.nlayers, howoften_toreset=cfg.howoften_toreset, h=0,
        trials=trials, maxiter=maxiter, howoften_tosave=cfg.howoften_tosave,
        unitary=cfg.unitary, sparse=cfg.sparse, perform_noisy_simulations=False,
        use_prob_resets_ansatz=True, reset_layers=reset_layers, use_mps=True,
        bond_dim=BOND_DIM[lattice], use_optimal_ordering=cfg.use_optimal_ordering,
        use_trajectory_resets=not enumerated, n_trajectories=1,
        use_enumerated_resets=enumerated, max_reset_branches=cfg.max_reset_branches,
        shared_reset_theta=True, seed=seed,
    )


def p_pattern(sigma, abs_a, R):
    """p^(a) = p_reset^|a| (1 - p_reset)^(R - |a|), eq. 3.7."""
    s2, c2 = np.sin(sigma / 2) ** 2, np.cos(sigma / 2) ** 2
    return s2 ** abs_a * c2 ** (R - abs_a)


class Dumper:
    """The step_callback handed to optimize(). Only reads params."""

    def __init__(self, ansatz, lattice, out, dump_steps, meta):
        self.ansatz, self.lattice, self.out = ansatz, lattice, out
        self.dump_steps = set(dump_steps)
        self.meta = meta
        self.R = ansatz.total_resets
        self.sig_idx = int(np.asarray(ansatz.reset_param_indices)[0])
        branches = np.asarray(ansatz._reset_branches)
        self.a = (branches > 0).astype(np.int8)
        self.b = (branches == 2).astype(np.int8)
        self.abs_a = self.a.sum(1)
        self.labels = np.asarray(trajectory_labels(branches))
        # Rows sharing a reset pattern, for the per-pattern sums.
        _, self.pattern_id = np.unique(self.a, axis=0, return_inverse=True)
        self.pattern_id = self.pattern_id.reshape(-1)
        self.branches = branches
        self.table = jax.jit(jax.vmap(ansatz.trajectory_table))

        def one_branch(params, branch):
            # Same quantities as VariationalAnsatz.trajectory_table's one_branch.
            qc = ansatz._circuit_branch(params, branch)
            centre = qc._mps.tensors[qc._mps.center_position]
            return ansatz._energy_of_circuit(qc), K.real(K.sum(centre * K.conj(centre)))

        # (trials, n) params x (trials, 3^R, R) branch lists. A fixed shape, so
        # it compiles once however many trajectories each trial skips.
        self.table_subset = jax.jit(jax.vmap(jax.vmap(one_branch, in_axes=(None, 0))))
        self.saved = {}          # (step, trial) -> dict, for the sampling check
        self.dump_seconds = 0.0

    def __call__(self, step, params):
        if step not in self.dump_steps:
            return
        t0 = time.perf_counter()
        params = np.asarray(params)
        half = params.copy()
        half[:, self.sig_idx] = np.pi / 2
        nums2, dens2 = (np.asarray(x) for x in self.table(jnp.asarray(half)))

        # Real-sigma reference over the kept trajectories only. A skipped slot is
        # filled with that trial's first kept trajectory and masked out after.
        sig = params[:, self.sig_idx]
        p_keep = self.meta["reference_p_keep"]
        keep = np.stack([p_pattern(s, self.abs_a, self.R) >= p_keep for s in sig])
        filler = self.branches[np.argmax(keep, axis=1)]
        branch_lists = np.where(keep[:, :, None], self.branches[None], filler[:, None, :])
        nums1, dens1 = (np.asarray(x) for x in
                        self.table_subset(jnp.asarray(params), jnp.asarray(branch_lists)))
        nums1, dens1 = np.where(keep, nums1, 0.0), np.where(keep, dens1, 0.0)

        failures = []
        for t in range(params.shape[0]):
            row, checks = self._one_trial(step, t, params[t], nums1[t], dens1[t],
                                          nums2[t], dens2[t], keep[t])
            failures += [f"step {step} trial {t}: {c}" for c, ok in checks.items() if not ok["pass"]]
        self.dump_seconds += time.perf_counter() - t0
        print(f"[dump] step {step}: {params.shape[0]} trials written "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        if failures:
            raise RuntimeError("trajectory dump check failed:\n  " + "\n  ".join(failures))

    def _one_trial(self, step, t, theta, nums1, dens1, nums2, dens2, keep):
        R, abs_a = self.R, self.abs_a
        sigma = float(theta[self.sig_idx])
        p_b_given_a = dens2 * 2.0 ** R
        alive = p_b_given_a > Q_FLOOR
        E_ab = np.where(alive, nums2 / np.where(alive, dens2, 1.0), np.nan)
        p_a = p_pattern(sigma, abs_a, R)
        p_ab = p_a * p_b_given_a
        # Weight of the trajectories skipped in the real-sigma reference, and the
        # most they can move its energy (|E^(ab)| <= E_UB).
        dropped = float(p_ab[~keep].sum())
        E_slack = dropped * self.meta["E_UB"]
        # Exact energy up to the skipped weight: sum of p^(ab) E^(ab) over the
        # kept trajectories at the real sigma.
        E_exact = float(nums1.sum())

        checks = {}

        def record(name, value, ok, detail=""):
            checks[name] = dict(value=float(value), **{"pass": bool(ok)})
            print(f"  step {step:>5} trial {t} {name:<28} {value:.3e}  "
                  f"{'ok' if ok else 'FAIL'} {detail}", flush=True)

        # 1. normalisation
        err = abs(p_ab.sum() - 1.0)
        record("sum p(ab) - 1", err, err < TOL_P)
        per_pattern = np.bincount(self.pattern_id, weights=p_b_given_a)
        err = np.abs(per_pattern - 1.0).max()
        record("max_a |sum_b p(b|a) - 1|", err, err < TOL_P)
        # 2. p^(a) and p^(ab) against the real-sigma table
        kept_pattern = np.bincount(self.pattern_id, weights=keep) > 0
        p_a_table = np.bincount(self.pattern_id, weights=dens1)
        p_a_formula = np.bincount(self.pattern_id, weights=p_a) / np.bincount(self.pattern_id)
        err = np.abs(p_a_table - p_a_formula)[kept_pattern].max()
        record("max |p(a) - table| (kept)", err, err < TOL_P + 1e-8 * p_a_formula.max())
        err = np.abs(p_ab - dens1)[keep].max()
        record("max |p(ab) - table| (kept)", err, err < TOL_P)
        # 3. trajectory counts
        counts = np.bincount(self.pattern_id)
        pattern_abs = np.bincount(self.pattern_id, weights=abs_a) / counts
        ok = len(p_ab) == 3 ** R and np.array_equal(counts, 2 ** pattern_abs.astype(int))
        record("count (3^R, 2^|a| each)", len(p_ab), ok)
        # 4. energy
        E_mix = np.nansum(p_ab * E_ab)
        err = abs(E_mix - E_exact)
        record("|sum p E - E_exact|", err, err < TOL_E + E_slack,
               f"(skipped weight {dropped:.1e})" if dropped else "")

        mu = E_mix
        var = np.nansum(p_ab * (E_ab - mu) ** 2)
        mu4 = np.nansum(p_ab * (E_ab - mu) ** 4)
        self.saved[(step, t)] = dict(theta=theta, E=mu, var=var, mu4=mu4, E_ab=E_ab[alive])

        stem = os.path.join(self.out, f"trial{t}_step{step}")
        np.savez_compressed(stem + ".npz", label=self.labels, a=self.a, abs_a=abs_a,
                            b=self.b, p_a=p_a, p_b_given_a=p_b_given_a, p_ab=p_ab, E_ab=E_ab)
        meta = dict(
            self.meta, trial=t, step=step, step_fraction=step / self.meta["maxiter"],
            R=R, sigma=sigma, sigma_index=self.sig_idx,
            p_reset_per_location=[float(np.sin(sigma / 2) ** 2)] * R,
            theta=theta.tolist(), E_exact=E_exact, E_variance=float(var),
            reference_kept_trajectories=int(keep.sum()), reference_skipped_weight=dropped,
            n_trajectories=int(len(p_ab)), n_undefined_E=int((~alive).sum()),
            nonfinite_steps_so_far=int(getattr(self.ansatz, "n_nonfinite_steps", 0)),
            checks=checks,
        )
        with open(stem + ".json", "w") as fh:
            json.dump(meta, fh, indent=2)
        return meta, checks


def sampling_check(lattice, dumper, n_samples, seed, trials, maxiter):
    """Check 5: single-trajectory samples against the saved distribution."""
    (step, t), s = max(dumper.saved.items(), key=lambda kv: kv[1]["var"])
    single = build_ansatz(lattice, enumerated=False, trials=1, maxiter=maxiter, seed=seed,
                          reset_layers=dumper.meta["reset_layers"])
    assert single.nparams == dumper.ansatz.nparams, "parameter layouts differ"
    params = jnp.tile(jnp.asarray(s["theta"]), (n_samples, 1))
    status = jax.random.uniform(jax.random.PRNGKey(seed + 1), (n_samples, 1, dumper.R))
    vals = np.asarray(single._costs_vmapped(params, status, jnp.zeros(n_samples)))

    mean, svar = vals.mean(), vals.var(ddof=1)
    se_mean = np.sqrt(s["var"] / n_samples)
    se_var = np.sqrt(max(s["mu4"] - s["var"] ** 2, 0.0) / n_samples)
    nearest = np.abs(vals[:, None] - s["E_ab"][None, :]).min(1).max()
    res = dict(
        lattice=lattice, step=step, trial=t, n_samples=n_samples,
        sample_mean=float(mean), exact_mean=float(s["E"]), se_mean=float(se_mean),
        sample_var=float(svar), exact_var=float(s["var"]), se_var=float(se_var),
        max_dist_to_saved_E_ab=float(nearest),
        mean_pass=bool(abs(mean - s["E"]) <= N_SIGMA * se_mean + TOL_E),
        var_pass=bool(abs(svar - s["var"]) <= N_SIGMA * se_var + TOL_E),
        values_pass=bool(nearest < TOL_E),
    )
    print(f"[sampling check] {lattice} step {step} trial {t}, {n_samples} samples:\n"
          f"  mean {mean:+.6f} vs exact {s['E']:+.6f}  (|diff| {abs(mean - s['E']):.2e}, "
          f"SE {se_mean:.2e})  {'ok' if res['mean_pass'] else 'FAIL'}\n"
          f"  var  {svar:.6f} vs exact {s['var']:.6f}  (|diff| {abs(svar - s['var']):.2e}, "
          f"SE {se_var:.2e})  {'ok' if res['var_pass'] else 'FAIL'}\n"
          f"  every sample is a saved E^(ab): max distance {nearest:.2e}  "
          f"{'ok' if res['values_pass'] else 'FAIL'}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lattice", required=True, choices=sorted(BOND_DIM))
    ap.add_argument("--maxiter", type=int, default=cfg.maxiter)
    ap.add_argument("--trials", type=int, default=cfg.trials)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--reset-layers", type=int, nargs="+", default=list(cfg.reset_layers),
                    help="layers carrying the probabilistic resets (zero-based)")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--out", default=os.path.join("outputs", "trajectory_dump"))
    ap.add_argument("--init-p-reset", type=float, default=None,
                    help="start every trial at this p_reset (sigma = 2 asin(sqrt(p))); "
                         "default keeps the ansatz's random reset-angle init")
    ap.add_argument("--p-keep", type=float, default=P_KEEP,
                    help="skip trajectories with p^(a) below this in the real-sigma reference")
    ap.add_argument("--dump-steps", type=int, nargs="+", default=None,
                    help="explicit steps to dump (default: 0, 20, ..., 100%% of --maxiter)")
    args = ap.parse_args()

    rl_tag = "reset_layers_" + "_".join(map(str, args.reset_layers))
    out = os.path.join(args.out, rl_tag, args.lattice)
    os.makedirs(out, exist_ok=True)
    commit, dirty = git_state()
    if dirty:
        print("WARNING: tracked files have uncommitted changes; git_commit does not "
              "fully identify this code.", flush=True)

    t_start = time.perf_counter()
    ansatz = build_ansatz(args.lattice, enumerated=True, trials=args.trials,
                          maxiter=args.maxiter, seed=args.seed, reset_layers=args.reset_layers)
    if args.init_p_reset is not None:
        # Only the shared reset angle changes; every other parameter keeps the
        # seed's draw, so runs are comparable with the default-init ones.
        sigma0 = 2.0 * np.arcsin(np.sqrt(args.init_p_reset))
        sig_idx = int(np.asarray(ansatz.reset_param_indices)[0])
        ansatz.initparams = jnp.asarray(ansatz.initparams).at[:, sig_idx].set(sigma0)
    dump_steps = sorted(set(args.dump_steps) if args.dump_steps else
                        {int(round(f * args.maxiter)) for f in FRACTIONS})
    coeffs = np.array([c for _, c in ansatz._hamiltonian_terms()])
    meta = dict(
        lattice=args.lattice, maxiter=args.maxiter, trials=args.trials,
        E_LB=float(-np.abs(coeffs).sum()), E_UB=float(np.abs(coeffs).sum()),
        seed=int(ansatz.seed), git_commit=commit, git_dirty=dirty,
        bond_dim=BOND_DIM[args.lattice], use_optimal_ordering=bool(cfg.use_optimal_ordering),
        nlayers=int(cfg.nlayers), reset_layers=args.reset_layers, reset_mode=ansatz.reset_mode,
        learning_rate=float(ansatz.learning_rate), shared_reset_theta=True,
        init_p_reset=args.init_p_reset, reference_p_keep=args.p_keep,
        svd_fwd_alg=os.environ.get("DPQC_SVD_FWD_ALG") or "default",
    )
    print(f"{args.lattice}: R={ansatz.total_resets}, {3 ** ansatz.total_resets} trajectories, "
          f"nparams={ansatz.nparams}, sigma index={list(np.asarray(ansatz.reset_param_indices))}, "
          f"dump steps {dump_steps}, E_LB/E_UB = {meta['E_LB']}/{meta['E_UB']}", flush=True)

    dumper = Dumper(ansatz, args.lattice, out, dump_steps, meta)
    t_train = time.perf_counter()
    ansatz.optimize(step_callback=dumper)
    train_seconds = time.perf_counter() - t_train - dumper.dump_seconds

    check = sampling_check(args.lattice, dumper, args.samples, args.seed,
                           args.trials, args.maxiter)
    with open(os.path.join(out, "sampling_check.json"), "w") as fh:
        json.dump(check, fh, indent=2)

    files = [f for f in os.listdir(out) if f.startswith("trial")]
    sizes = [os.path.getsize(os.path.join(out, f)) for f in files]
    summary = dict(
        lattice=args.lattice, total_seconds=time.perf_counter() - t_start,
        training_seconds=train_seconds, dump_seconds=dumper.dump_seconds,
        n_files=len(files), total_bytes=int(sum(sizes)),
        max_file_bytes=int(max(sizes)), n_nonfinite_steps=int(getattr(ansatz, "n_nonfinite_steps", 0)),
    )
    with open(os.path.join(out, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    if not (check["mean_pass"] and check["var_pass"] and check["values_pass"]):
        raise RuntimeError(f"sampling check failed: {check}")


if __name__ == "__main__":
    main()
