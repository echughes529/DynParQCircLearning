"""Experiment 3: does skipping normalization actually produce NaNs in training?

Runs the real Adam loop (same optimizer, same vmapped value-and-grad helper as
find_gs.optimize) for each of the three normalization regimes on matched
seeds, and records the step at which the first non-finite energy or gradient
appears, plus per-step diagnostics of the quantities the other experiments
predict should drive it: ||psi||^2, gradient norm, smallest singular value and
smallest squared-singular-value gap actually fed to the SVD backward pass.

Each (mode, bond_dim) pair trains `trials` independent parameter starts at
once through the same vmap the production code uses, so "how often does a run
NaN" is measured per trial, not per job.

Usage:
    python -m src.diagnostics.norm_training_experiment \
        --Lx 3 --Ly 2 --bond-dim 8 --seeds 1 2 3 4 --maxiter 300
"""
import argparse
import json
import sys
import time
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from src.diagnostics.normalization_study import (
    NORM_MODES,
    SAFE_RECIPROCAL_KNEE,
    make_ansatz,
)
from src.utilities.generate_ansatz import get_singular_values_per_cut


def true_energy(ansatz, params_row):
    """<psi|H|psi>/<psi|psi> for the current parameters.

    The loss value each mode reports is NOT comparable across modes -- with
    norm_mode="off" it is ||psi||^2 * E, which is not an expectation value of
    anything. Every energy compared across modes in this experiment goes
    through here instead.
    """
    return float(np.asarray(ansatz.rayleigh_from_params(params_row)))


def probe_state(ansatz, params_row):
    """Forward-only look at the state the loss was just evaluated on."""
    qc = ansatz._circuit(params_row)
    norm_sq = float(np.abs(np.asarray(qc.get_norm()))) ** 2
    if ansatz.norm_mode != "off":
        qc.normalize()
    try:
        spectra = get_singular_values_per_cut(qc)
    except Exception:
        return norm_sq, np.nan, np.nan
    min_sv = np.inf
    min_gap = np.inf
    for s in spectra:
        s = np.asarray(s, dtype=float)
        s = s[s > 0]
        if s.size:
            min_sv = min(min_sv, float(s.min()))
        if s.size > 1:
            sq = np.sort(s ** 2)[::-1]
            d = -np.diff(sq)
            d = d[d > 0]
            if d.size:
                min_gap = min(min_gap, float(d.min()))
    return norm_sq, (min_sv if np.isfinite(min_sv) else np.nan), \
        (min_gap if np.isfinite(min_gap) else np.nan)


def run_one(mode, bond_dim, seed, args):
    reset_layers = ([args.nlayers - 1] if args.reset_layers is None
                    else args.reset_layers)
    a = make_ansatz(mode, Lx=args.Lx, Ly=args.Ly, nlayers=args.nlayers,
                    howoften_toreset=7, reset_layers=reset_layers,
                    trials=args.trials, bond_dim=bond_dim, seed=seed,
                    use_optimal_ordering=not args.natural_ordering,
                    cartan_mode=args.cartan_mode,
                    toffoli_mode=args.toffoli_mode,
                    learning_rate=args.learning_rate)
    params = jnp.array(a.initparams)
    optimizer = optax.adam(learning_rate=a.learning_rate)
    opt_state = optimizer.init(params)

    history = []
    first_nan_step = None
    nan_trials = []
    t0 = time.time()

    for step in range(args.maxiter):
        value, grad = a._cost_vvag(params)
        v = np.asarray(value)
        g = np.asarray(grad)
        bad = ~np.isfinite(v) | (~np.isfinite(g)).any(axis=-1)

        if bad.any() and first_nan_step is None:
            first_nan_step = step
            nan_trials = np.where(bad)[0].tolist()

        if step % args.print_every == 0 or (first_nan_step == step):
            trial = int(np.where(bad)[0][0]) if bad.any() else 0
            norm_sq, min_sv, min_gap = probe_state(a, params[trial])
            # Mode-independent yardstick: the true Rayleigh quotient, best
            # over trials, so "off" and "layer" curves can be plotted together.
            e_true = [true_energy(a, params[t]) for t in range(args.trials)]
            rec = dict(step=step,
                       loss=float(np.nanmean(np.real(v))),
                       energy_true_mean=float(np.nanmean(e_true)),
                       energy_true_best=float(np.nanmin(e_true)),
                       grad_norm=float(np.linalg.norm(g[np.isfinite(g).all(axis=-1)]))
                       if np.isfinite(g).all(axis=-1).any() else np.nan,
                       grad_norm_max=float(np.max(np.linalg.norm(
                           np.where(np.isfinite(g), g, 0.0), axis=-1))),
                       n_bad_trials=int(bad.sum()),
                       norm_sq=norm_sq, min_sv=min_sv, min_sq_gap=min_gap,
                       gap_below_knee=bool(np.isfinite(min_gap) and
                                           min_gap < SAFE_RECIPROCAL_KNEE))
            history.append(rec)
            print(f"    [{mode:<5} bd={bond_dim:<3} seed={seed}] step {step:>4}: "
                  f"loss={rec['loss']:+.6f}  E_true={rec['energy_true_mean']:+.6f}  "
                  f"best={rec['energy_true_best']:+.6f}  |g|={rec['grad_norm']:.3e}  "
                  f"||psi||^2={norm_sq:.6f}  min_sv={min_sv:.2e}  "
                  f"min_sq_gap={min_gap:.2e}  bad={rec['n_bad_trials']}", flush=True)

        if bad.all():
            print(f"    [{mode:<5} bd={bond_dim:<3} seed={seed}] all trials NaN at "
                  f"step {step} -- stopping", flush=True)
            break

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

    final = np.asarray(a._cost_vvag(params)[0])
    final_true = [true_energy(a, params[t]) for t in range(args.trials)]
    return dict(mode=mode, bond_dim=bond_dim, seed=seed, trials=args.trials,
                maxiter=args.maxiter, Lx=args.Lx, Ly=args.Ly,
                nlayers=args.nlayers,
                first_nan_step=first_nan_step, nan_trials=nan_trials,
                n_trials_nan_at_end=int((~np.isfinite(final)).sum()),
                final_loss_mean=float(np.nanmean(np.real(final))),
                final_energy_mean=float(np.nanmean(final_true)),
                final_energy_best=float(np.nanmin(final_true))
                if np.isfinite(final_true).any() else np.nan,
                wall_seconds=time.time() - t0,
                history=history)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=3)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--bond-dims", type=int, nargs="+", default=[8])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--modes", nargs="+", default=list(NORM_MODES))
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--cartan-mode", default="fused")
    p.add_argument("--toffoli-mode", default="direct")
    p.add_argument("--reset-layers", type=int, nargs="+", default=None)
    p.add_argument("--natural-ordering", action="store_true",
                   help="use_optimal_ordering=False, i.e. the pre-abf5c85 default "
                        "under which the historical 3x3 bd=64 NaNs were observed")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = []
    for bond_dim in args.bond_dims:
        for seed in args.seeds:
            for mode in args.modes:
                results.append(run_one(mode, bond_dim, seed, args))
                r = results[-1]
                print(f"  => {mode:<5} bd={bond_dim:<3} seed={seed}: "
                      f"first_nan_step={r['first_nan_step']}  "
                      f"nan_trials_at_end={r['n_trials_nan_at_end']}/{args.trials}  "
                      f"E_true_best={r['final_energy_best']:+.6f}  "
                      f"({r['wall_seconds']:.0f}s)", flush=True)
                if args.out:
                    with open(args.out, "w") as f:
                        json.dump(results, f, indent=1, default=float)

    print("\n================ SUMMARY ================")
    for mode in args.modes:
        sel = [r for r in results if r["mode"] == mode]
        n_runs_nan = sum(1 for r in sel if r["first_nan_step"] is not None)
        n_trials_nan = sum(r["n_trials_nan_at_end"] for r in sel)
        n_trials_tot = sum(r["trials"] for r in sel)
        best = [r["final_energy_best"] for r in sel if np.isfinite(r["final_energy_best"])]
        print(f"  {mode:<5}: {n_runs_nan}/{len(sel)} runs hit NaN, "
              f"{n_trials_nan}/{n_trials_tot} trials NaN at end, "
              f"best E = {min(best) if best else float('nan'):+.6f}")


if __name__ == "__main__":
    main()
