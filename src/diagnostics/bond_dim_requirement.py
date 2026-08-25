"""How much bond dimension does each reset implementation actually need?

Run:
    python -m src.diagnostics.bond_dim_requirement --lattices 2x2,3x2,3x3
    python -m src.diagnostics.bond_dim_requirement --lattices 4x3 --out results.csv

WHY THIS IS THE HEADLINE MEASUREMENT
------------------------------------
The purified reset path spends two ancillas per reset event, so its MPS chain is
nq + 2*R sites; the trajectory path samples one branch and stays at nq sites. At
3x3 that is 20 sites against 12. A longer chain is not automatically worse, but
it does have to carry the ancilla entanglement in its bond dimension, and bond
dimension is the thing that actually blocks larger lattices.

So the question is not "who is faster at bond_dim=64". It is "what bond
dimension does each path need before its answer is right", and only then "what
does that bond dimension cost in seconds".

HOW "NEEDED" IS MEASURED
------------------------
Build the circuit with normalize_state=False and read the norm. Every MPS
truncation throws away some weight, and with normalization off that loss
accumulates into the state norm, so

    discarded weight = 1 - ||psi||^2

is a direct readout of how much of the state the bond dimension could not hold.
The fractional error in the unnormalised energy equals this quantity exactly.
Note that normalize_state=True -- the production setting -- makes ||psi||^2 read
1.0 even when 7% of the state has been discarded, which is precisely why this
diagnostic has to turn it off.

The pass gate is 1e-5, two orders below the 1e-3 relative-energy criterion the
plotting script uses to call a trial converged.

An MPS on n sites with bond dimension 2^floor(n/2) is the full Hilbert space, so
truncation is provably zero there. Those caps are printed alongside the results:
if a measured requirement comes out above the cap, the measurement is wrong.
"""

import argparse
import sys
import time
import types

# TensorFlow is only needed by unrelated legacy constructors in generate_ansatz;
# stub it unconditionally -- some environments have a TF/sklearn build compiled
# against a different NumPy ABI than the installed NumPy, which crashes hard on
# import rather than raising cleanly.
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import h5py
import numpy as np
import jax
import jax.numpy as jnp

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.generate_ansatz import make_split_conf

# Pass gate on discarded weight, two orders below the 1e-3 convergence criterion.
GATE = 1e-5
# Trajectories sampled per bond dimension. Each is a different pure state, so
# the requirement is the worst case over them, not the average.
N_TRAJ_SAMPLES = 20

NLAYERS = 2
RESET_LAYERS = [1]
SEED = 2026          # matches the converged 3x3 run stored in results/toriccode.h5

LADDERS = {
    ("2x2", "traj"): [2, 3, 4],
    ("2x2", "pur"): [4, 6, 8],
    ("3x2", "traj"): [4, 6, 8],
    ("3x2", "pur"): [8, 16, 24, 32],
    ("3x3", "traj"): [8, 16, 24, 32, 48, 64],
    ("3x3", "pur"): [32, 48, 64, 80, 96, 112, 128],
    ("4x3", "traj"): [16, 32, 48, 64, 96, 128, 192, 256],
    ("4x3", "pur"): [64, 96, 128, 160, 192, 256],
    ("4x4", "traj"): [32, 64, 96, 128, 192, 256],
    ("4x4", "pur"): [64, 128, 192, 256],
}


def lattice_facts(Lx, Ly):
    """Chain lengths, exact bond-dimension caps and E0, from the lattice alone."""
    nq = 2 * Lx * Ly - Lx - Ly
    R = (Lx - 1) * (Ly - 1)                  # total_resets for reset_layers=[1]
    stars, plaquettes = Lx * Ly, (Lx - 1) * (Ly - 1)
    return {
        "nq": nq,
        "R": R,
        "traj_chain": nq,
        "pur_chain": nq + 2 * R,
        "traj_cap": 2 ** (nq // 2),
        "pur_cap": 2 ** ((nq + 2 * R) // 2),
        "E0": -(stars + plaquettes),
    }


def build(Lx, Ly, mode, bond_dim, use_optimal_ordering=True):
    """One ansatz per (lattice, mode). Bond dimension is varied afterwards.

    The ordering search runs up to 2M random permutations inside __post_init__,
    which would be wasted work if repeated per rung of the ladder.
    """
    return ToricCodeAnsatz(
        Lx=Lx, Ly=Ly, nlayers=NLAYERS, reset_layers=RESET_LAYERS,
        trials=1, seed=SEED, bond_dim=bond_dim,
        use_optimal_ordering=use_optimal_ordering,
        use_trajectory_resets=(mode == "traj"),
        # The whole point of this diagnostic: with normalization on, the norm is
        # forced back to 1 after every layer and the discarded weight vanishes
        # from view.
        normalize_state=False,
        sparse=False, use_mps=True,
    )


def set_bond_dim(ansatz, bond_dim):
    """Re-point an already-built ansatz at a different bond dimension.

    The circuit builders read `self.split_conf` when they are called rather than
    capturing it at construction time, so overwriting it here changes the
    truncation of the next circuit built -- and skips repeating the multi-minute
    ordering search that __post_init__ would otherwise redo.
    """
    ansatz.bond_dim = bond_dim
    ansatz.split_conf = make_split_conf(bond_dim)


def measure(ansatz, params, mode, rng):
    """Discarded weight, energy and realised max bond dimension at one setting."""
    if mode == "traj":
        weights, energies, maxbds = [], [], []
        for _ in range(N_TRAJ_SAMPLES):
            status = jnp.asarray(rng.uniform(size=ansatz.total_resets))
            qc, _ = ansatz._circuit_traj(params, status)
            weights.append(discarded_weight(qc))
            energies.append(float(ansatz._energy_of_circuit(qc)))
            maxbds.append(int(np.max(np.asarray(qc.get_bond_dimensions()))))
        # Worst case over trajectories: each is a different pure state, and the
        # bond dimension has to be enough for all of them, not the average one.
        return {
            "discarded_weight": float(np.max(weights)),
            "discarded_weight_mean": float(np.mean(weights)),
            # Energies here are unnormalised on purpose -- rescaling by the norm
            # would hide exactly the error being measured.
            "energy": float(np.mean(energies)),
            "energy_std": float(np.std(energies)),
            "max_bond_dim": int(np.max(maxbds)),
        }

    qc = ansatz._circuit(params)
    return {
        "discarded_weight": discarded_weight(qc),
        "discarded_weight_mean": discarded_weight(qc),
        "energy": float(ansatz._energy_of_circuit(qc)),
        "energy_std": 0.0,
        "max_bond_dim": int(np.max(np.asarray(qc.get_bond_dimensions()))),
    }


def discarded_weight(qc):
    """1 - ||psi||^2, the fraction of the state truncation threw away."""
    return float(1.0 - np.asarray(qc.get_norm()) ** 2)


def converged_params_3x3():
    """Best converged 3x3 parameter vector from the stored bd=96 run, or None.

    Required bond dimension is state-dependent, so measuring only at a random
    initialisation would answer the wrong question. This run reached exactly
    -13 on 6 of 20 trials, so its best trial is a genuine ground state.
    """
    try:
        with h5py.File("results/toriccode.h5", "r") as f:
            for name in sorted(f.keys(), reverse=True):
                s = f[name]["settings"].attrs
                if int(s.get("Lx", 0)) == 3 and int(s.get("Ly", 0)) == 3 and int(s.get("nlayers", 0)) == NLAYERS:
                    energies = f[name]["results"]["final_energies"][:]
                    params = f[name]["results"]["final_parameters"][:]
                    best = int(np.argmin(energies))
                    print(f"  loaded converged 3x3 params from {name} "
                          f"(trial {best}, E={energies[best]:.6f}, trained at bd={s.get('bond_dim')})")
                    return jnp.asarray(params[best])
    except (OSError, KeyError) as exc:
        print(f"  no stored 3x3 parameters available ({exc})")
    return None


def checkpoint_param_sets(path, n_stages=4, trial=0):
    """Parameter vectors from several points along a real training run.

    Random initialisations and converged ground states are the two endpoints,
    and they can want very different bond dimensions. What a run actually needs
    is the worst of everything it passes through, which only a real trajectory
    can supply.
    """
    d = np.load(path)
    all_params = d["all_params"]
    filled = int(d["nsnapshots_filled"])
    if not np.any(all_params):
        print(f"  {path}: all_params is all zeros (run had track_params off); skipping")
        return {}
    idx = np.unique(np.linspace(0, max(filled - 1, 0), n_stages).astype(int))
    return {f"step{int(i)}": jnp.asarray(all_params[trial, i, :]) for i in idx}


def run_lattice(Lx, Ly, rows, use_optimal_ordering=True, extra_params=None):
    key = f"{Lx}x{Ly}"
    facts = lattice_facts(Lx, Ly)
    print(f"\n{'=' * 78}")
    print(f"{key}: nq={facts['nq']} R={facts['R']} | chain traj={facts['traj_chain']} "
          f"pur={facts['pur_chain']} | exact caps traj={facts['traj_cap']} pur={facts['pur_cap']} "
          f"| E0={facts['E0']}")
    print("=" * 78)

    converged = converged_params_3x3() if key == "3x3" else None

    for mode in ("traj", "pur"):
        ladder = [bd for bd in LADDERS[(key, mode)]]
        cap = facts["traj_cap"] if mode == "traj" else facts["pur_cap"]
        t0 = time.perf_counter()
        ansatz = build(Lx, Ly, mode, ladder[-1], use_optimal_ordering)
        build_s = time.perf_counter() - t0

        param_sets = {"init": jnp.asarray(ansatz.initparams[0])}
        if converged is not None:
            param_sets["converged"] = converged
        param_sets.update(extra_params or {})

        for pname, params in param_sets.items():
            print(f"\n  --- {key} {mode} params={pname} (build {build_s:.1f}s, "
                  f"chain={ansatz.n_mps_qubits}, exact cap={cap}) ---")
            print(f"  {'bd':>5} {'discarded wt':>14} {'energy':>14} {'max bd':>7} {'sec':>7}")
            required = None
            for bd in ladder:
                if bd > cap:
                    continue
                set_bond_dim(ansatz, bd)
                t1 = time.perf_counter()
                m = measure(ansatz, params, mode, np.random.default_rng(12345))
                dt = time.perf_counter() - t1
                flag = ""
                if required is None and m["discarded_weight"] <= GATE:
                    required = bd
                    flag = "  <-- gate passed"
                if bd >= cap:
                    flag += "  (exact: full Hilbert space)"
                print(f"  {bd:>5} {m['discarded_weight']:>14.3e} {m['energy']:>14.6f} "
                      f"{m['max_bond_dim']:>7} {dt:>7.2f}{flag}")
                rows.append(dict(lattice=key, mode=mode, params=pname, bond_dim=bd,
                                 chain=ansatz.n_mps_qubits, exact_cap=cap,
                                 E0=facts["E0"], seconds=dt,
                                 use_optimal_ordering=use_optimal_ordering, **m))
            if required is None:
                print(f"  REQUIRED bd: not reached within the ladder (max {ladder[-1]})")
            else:
                # Headroom above the exact cap is padding with zeros, not extra
                # capacity: at 2^floor(n/2) the MPS already is the full Hilbert
                # space and there is nothing left to represent.
                headroom = min(required + 8, cap)
                extra = "" if headroom > required else "  (already exact -- no headroom is meaningful)"
                print(f"  REQUIRED bd: {required}  ->  long-run bd = {headroom}{extra}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattices", default="2x2,3x2,3x3",
                        help="comma-separated, e.g. 2x2,3x2,3x3,4x3")
    parser.add_argument("--out", default="outputs/traj_vs_purified/bond_dim_requirement.csv")
    parser.add_argument("--natural-ordering", action="store_true",
                        help="control run with use_optimal_ordering=False, so both paths "
                             "share the natural system-qubit order")
    parser.add_argument("--from-checkpoint", default=None,
                        help="also measure at parameters sampled along a real training run "
                             "(a checkpoint.npz written by optimize())")
    args = parser.parse_args()

    print(f"jax devices: {jax.devices()}")
    print(f"gate: discarded weight <= {GATE:g}; trajectory samples per rung: {N_TRAJ_SAMPLES}")

    extra = checkpoint_param_sets(args.from_checkpoint) if args.from_checkpoint else None

    rows = []
    for spec in args.lattices.split(","):
        Lx, Ly = (int(v) for v in spec.strip().split("x"))
        run_lattice(Lx, Ly, rows, use_optimal_ordering=not args.natural_ordering,
                    extra_params=extra)

    if rows:
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
