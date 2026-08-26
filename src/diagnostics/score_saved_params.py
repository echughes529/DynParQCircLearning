"""Re-score a training run's saved parameters against one common energy yardstick.

Run:
    python -m src.diagnostics.score_saved_params \
        --checkpoint logs/<run>/checkpoint.npz --lattice 3x3 --ref-bond-dim 128

WHY THIS EXISTS
---------------
The three reset implementations do not report comparable energies during
training. Purified and enumerated both evaluate the reset channel exactly, so
their reported value is the channel energy -- but at different chain lengths and
therefore different truncation errors for the same bond dimension. The
trajectory path reports the energy of ONE sampled branch, an unbiased estimate
of that same quantity carrying the sampling noise of a single draw. Plotting any
of these against the others and calling the difference "accuracy" would be
comparing a quantity against a differently-biased estimate of itself.

So every accuracy number in this investigation comes from here instead: take the
parameters a run actually reached, and evaluate them with ONE reference circuit
at a bond dimension high enough to be converged. That is a single deterministic
function applied to all three arms, which makes the comparison meaningful.

CHOICE OF REFERENCE (--ref-mode)
--------------------------------
Purified is the default and the historical choice. Enumerated is the interesting
alternative: it is equally exact, but runs on the bare nq-site chain rather than
nq + 2R sites, so at 3x3 it is converged on 12 sites where purified needs 20.
For a fixed compute budget that is a better-converged yardstick. It costs 3^R
forward evaluations, which is affordable here because scoring needs no gradients.

Scoring the same parameters under both references and finding the same answer is
the strongest available check that the yardstick itself is sound -- if they
disagree, at least one reference is under-resourced.

Scoring happens offline, from parameters checkpointed during training, so the
training jobs never pay for the high-bond-dimension reference evaluation.

It also recovers PURITY for trajectory runs. purity_from_params deliberately
refuses to run in trajectory mode -- each trajectory is a pure state, so the
ancilla-cut purity would trivially read 1.0 and the mixed state only exists as
the ensemble over trajectories. Evaluated on the purified circuit at the same
parameters, the quantity is well defined again.
"""

import argparse
import csv
import os
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import h5py
import numpy as np
import jax
import jax.numpy as jnp

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.find_gs import purity_vec

NLAYERS = 2
RESET_LAYERS = [1]


def E0(Lx, Ly):
    """Exact ground-state energy at h=0: minus (number of stars + plaquettes)."""
    return -(Lx * Ly + (Lx - 1) * (Ly - 1))


def load_checkpoint(path):
    """Snapshots from an in-flight or finished run's checkpoint file."""
    d = np.load(path, allow_pickle=True)
    out = {
        "all_params": d["all_params"],          # (trials, snapshots, nparams)
        "all_energies": d["all_energies"],      # (trials, snapshots)
        "step_times": d["step_times"],
        "filled": int(d["nsnapshots_filled"]),
        "final_params": d["params"],
    }
    # Self-describing checkpoints carry which arm produced them, so the label
    # does not have to be supplied by hand and cannot be got wrong.
    out["reset_mode"] = str(d["reset_mode"]) if "reset_mode" in d.files else None
    out["bond_dim"] = int(d["bond_dim"]) if "bond_dim" in d.files else None
    out["seed"] = int(d["seed"]) if "seed" in d.files else None
    return out


def load_hdf5(path, run_id=None):
    with h5py.File(path, "r") as f:
        runs = sorted(k for k in f if k.startswith("run_"))
        run_id = run_id or runs[-1]
        res = f[run_id]["results"]
        out = {
            "all_params": res["all_params"][:],
            "all_energies": res["all_energies"][:],
            "final_params": res["final_parameters"][:],
            "filled": res["all_energies"].shape[1],
            "step_times": res["step_times"][:] if "step_times" in res else np.array([]),
        }
        print(f"  loaded {run_id} from {path}")
        return out


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="checkpoint.npz written during training")
    src.add_argument("--hdf5", help="results/*.h5 written after training")
    parser.add_argument("--run-id", default=None, help="run group inside --hdf5")
    parser.add_argument("--lattice", required=True, help="e.g. 3x3")
    parser.add_argument("--ref-bond-dim", type=int, required=True,
                        help="bond dimension for the reference evaluation; must be at or "
                             "above the reference arm's requirement as measured by "
                             "bond_dim_requirement")
    parser.add_argument("--ref-mode", default="pur", choices=("pur", "enum"),
                        help="which exact arm to use as the yardstick. pur is the default; "
                             "enum is equally exact on a shorter chain (nq vs nq+2R sites), "
                             "so it is better converged for the same bond dimension.")
    parser.add_argument("--howoften-tosave", type=int, default=10,
                        help="snapshot cadence of the run being scored, to recover step numbers")
    parser.add_argument("--every", type=int, default=1,
                        help="score every Nth snapshot (the reference evaluation is not cheap)")
    parser.add_argument("--track-purity", action="store_true")
    parser.add_argument("--label", default="", help="free-text tag copied into every output row")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.track_purity and args.ref_mode == "enum":
        # Purity is read off the ancilla cut, and the enumerated reference has no
        # ancillas -- purity_from_params refuses for exactly that reason. The
        # purified reference is the one that can recover it.
        parser.error("--track-purity needs --ref-mode pur: the enumerated reference "
                     "is ancilla-free, so there is no cut to trace out.")

    Lx, Ly = (int(v) for v in args.lattice.split("x"))
    data = load_checkpoint(args.checkpoint) if args.checkpoint else load_hdf5(args.hdf5, args.run_id)

    all_params = data["all_params"]
    trials, nsnapshots, nparams = all_params.shape
    filled = data["filled"]
    if not np.any(all_params):
        raise SystemExit("all_params is all zeros -- the run was not launched with track_params=True, "
                         "so there is nothing to score.")

    # The arm being scored, taken from the checkpoint so it cannot be mislabelled.
    scored_mode = data.get("reset_mode")
    label = args.label or (f"{scored_mode}_{data.get('seed')}" if scored_mode else "")

    print(f"jax devices: {jax.devices()}")
    print(f"scoring {args.lattice} {scored_mode or '(arm unknown)'} "
          f"(trained at bond_dim={data.get('bond_dim')}, seed={data.get('seed')}): "
          f"{trials} trials x {filled} filled snapshots")
    print(f"reference: {args.ref_mode} at bond_dim={args.ref_bond_dim}")

    # The reference is an EXACT arm -- purified or enumerated -- evaluated with
    # no sampling, so it is the same deterministic function applied to all three
    # arms' parameters.
    ref = ToricCodeAnsatz(
        Lx=Lx, Ly=Ly, nlayers=NLAYERS, reset_layers=RESET_LAYERS,
        trials=trials, seed=1, bond_dim=args.ref_bond_dim,
        use_optimal_ordering=True, use_trajectory_resets=False,
        use_enumerated_resets=(args.ref_mode == "enum"),
        max_reset_branches=3 ** 8,
        normalize_state=True, sparse=False, use_mps=True,
    )
    if ref.nparams != nparams:
        raise SystemExit(f"parameter count mismatch: run has {nparams}, "
                         f"{args.lattice} reference expects {ref.nparams}")

    exact = E0(Lx, Ly)
    rows = []
    snapshot_indices = list(range(0, filled, args.every))
    if filled - 1 not in snapshot_indices:
        snapshot_indices.append(filled - 1)

    for si in snapshot_indices:
        params = jnp.asarray(all_params[:, si, :])
        # One vmapped call scores every trial at once, the same way the
        # optimiser evaluates them.
        energies = np.asarray(jax.block_until_ready(ref._costs_vmapped(params)))
        purities = np.asarray(purity_vec(ref, params)) if args.track_purity else np.full(trials, np.nan)
        step = si * args.howoften_tosave
        for t in range(trials):
            rows.append(dict(
                label=label, lattice=args.lattice, trial=t, snapshot=si, step=step,
                scored_mode=scored_mode, ref_mode=args.ref_mode,
                trained_bond_dim=data.get('bond_dim'), seed=data.get('seed'),
                energy_exact=float(energies[t]),
                energy_reported=float(data["all_energies"][t, si]),
                rel_error=float(abs(energies[t] - exact) / abs(exact)),
                purity=float(purities[t]),
                E0=exact, ref_bond_dim=args.ref_bond_dim,
            ))
        best = energies.min()
        print(f"  step {step:>5}: best exact energy {best:>10.6f} of {exact} "
              f"(rel err {abs(best - exact) / abs(exact):.3e}), "
              f"reported {data['all_energies'][:, si].min():>10.6f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    final = np.asarray([r["energy_exact"] for r in rows if r["snapshot"] == snapshot_indices[-1]])
    print(f"final: best {final.min():.6f}, {int(np.sum(np.abs(final - exact) / abs(exact) < 1e-3))}/{trials} "
          f"trials within 1e-3 of E0={exact}")


if __name__ == "__main__":
    main()
