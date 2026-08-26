"""How much bond dimension does each reset implementation actually need?

Run:
    python -m src.diagnostics.bond_dim_requirement --lattices 2x2,3x2,3x3
    python -m src.diagnostics.bond_dim_requirement --lattices 4x3 --modes traj,enum

WHY THIS IS THE HEADLINE MEASUREMENT
------------------------------------
The purified reset path spends two ancillas per reset event, so its MPS chain is
nq + 2*R sites; the trajectory and enumerated paths are ancilla-free and stay at
nq sites. At 3x3 that is 20 sites against 12. A longer chain is not automatically
worse, but it does have to carry the ancilla entanglement in its bond dimension,
and bond dimension is the thing that actually blocks larger lattices.

So the question is not "who is faster at bond_dim=64". It is "what bond
dimension does each path need before its answer is right", and only then "what
does that bond dimension cost in seconds". Comparing all three at one bond
dimension would flatter whichever is being under-resourced.

HOW "NEEDED" IS MEASURED
------------------------
Every MPS truncation throws away some weight, and with normalisation off that
loss accumulates into the state norm, so

    discarded weight = 1 - ||psi||^2

is a direct readout of how much of the state the bond dimension could not hold.
The fractional error in the unnormalised energy equals this quantity exactly.
normalize_state=True -- the production setting -- makes ||psi||^2 read 1.0 even
when 7% of the state has been discarded, which is why this diagnostic turns it
off for the two arms that have it.

The three arms read it slightly differently:

  purified     one state; 1 - ||psi||^2 directly.
  trajectory   one state per sampled branch; the worst case over N_TRAJ_SAMPLES,
               since the bond dimension has to be enough for every trajectory
               the optimiser might draw, not the average one.
  enumerated   1 - sum_j ||phi_j||^2 over all 3^R branches. No normalize_state
               trick is needed: the branch Kraus set is exactly trace
               preserving, so that sum is identically 1 in exact arithmetic and
               the shortfall is the whole channel's discarded weight. See
               measure_enum.

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

MODES = ("traj", "pur", "enum")

# The enumerated arm runs on the same nq-site chain as the trajectory arm, and
# _branch_reset applies its 2x2 Kraus factor at the orthogonality centre, which
# costs no SVD and no truncation. So its requirement should land near the
# trajectory one -- the open question this measures is whether the WORST of the
# 3^R branches is meaningfully worse than the worst of 20 sampled trajectories.
LADDERS = {
    ("2x2", "traj"): [2, 3, 4],
    ("2x2", "enum"): [2, 3, 4],
    ("2x2", "pur"): [4, 6, 8],
    ("3x2", "traj"): [4, 6, 8],
    ("3x2", "enum"): [4, 6, 8],
    ("3x2", "pur"): [8, 16, 24, 32],
    ("3x3", "traj"): [8, 16, 24, 32, 48, 64],
    ("3x3", "enum"): [8, 16, 24, 32, 48, 64],
    ("3x3", "pur"): [32, 48, 64, 80, 96, 112, 128],
    ("4x3", "traj"): [16, 32, 48, 64, 96, 128, 192, 256],
    ("4x3", "enum"): [16, 32, 48, 64, 96, 128, 192, 256],
    ("4x3", "pur"): [64, 96, 128, 160, 192, 256, 384, 512],
    ("4x4", "traj"): [32, 64, 96, 128, 192, 256],
    ("4x4", "enum"): [32, 64, 96, 128, 192, 256],
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
        "nbranches": 3 ** R,
        # Both ancilla-free arms share the bare system chain; only the purified
        # one pays 2 sites per reset event.
        "traj_chain": nq,
        "enum_chain": nq,
        "pur_chain": nq + 2 * R,
        "traj_cap": 2 ** (nq // 2),
        "enum_cap": 2 ** (nq // 2),
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
        use_enumerated_resets=(mode == "enum"),
        # 3^R branches: 81 at 3x3 but 729 at 4x3, above the default guard of 512.
        # Affordable here because this diagnostic walks the branches one at a
        # time with no autodiff, so only one branch is ever resident.
        max_reset_branches=3 ** 8,
        # The whole point of this diagnostic: with normalization on, the norm is
        # forced back to 1 after every layer and the discarded weight vanishes
        # from view. (The branch builder ignores this flag -- it never normalises
        # at all, because the surviving norm IS the branch probability.)
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
    """Discarded weight, energy and realised max bond dimension at one setting.

    Every arm reports both an unnormalised and a normalised energy. The
    unnormalised one is the raw readout, where the fractional error equals the
    discarded weight exactly; the normalised one is what each arm's estimator
    actually returns to the optimiser. They differ per arm, and that difference
    is itself a result: the enumerated estimator divides by the branch-norm sum
    as part of its definition, so it self-corrects for truncation drift, whereas
    the other two only do so when normalize_state is on.
    """
    if mode == "enum":
        return measure_enum(ansatz, params)

    if mode == "traj":
        weights, energies, norm_energies, maxbds = [], [], [], []
        for _ in range(N_TRAJ_SAMPLES):
            status = jnp.asarray(rng.uniform(size=ansatz.total_resets))
            qc, _ = ansatz._circuit_traj(params, status)
            w = discarded_weight(qc)
            e = float(ansatz._energy_of_circuit(qc))
            weights.append(w)
            energies.append(e)
            norm_energies.append(e / (1.0 - w))
            maxbds.append(int(np.max(np.asarray(qc.get_bond_dimensions()))))
        # Worst case over trajectories: each is a different pure state, and the
        # bond dimension has to be enough for all of them, not the average one.
        return {
            "discarded_weight": float(np.max(weights)),
            "discarded_weight_mean": float(np.mean(weights)),
            # Energies here are unnormalised on purpose -- rescaling by the norm
            # would hide exactly the error being measured.
            "energy": float(np.mean(energies)),
            "energy_normalised": float(np.mean(norm_energies)),
            "energy_std": float(np.std(energies)),
            "max_bond_dim": int(np.max(maxbds)),
            "n_states": N_TRAJ_SAMPLES,
        }

    qc = ansatz._circuit(params)
    w = discarded_weight(qc)
    e = float(ansatz._energy_of_circuit(qc))
    return {
        "discarded_weight": w,
        "discarded_weight_mean": w,
        "energy": e,
        "energy_normalised": e / (1.0 - w),
        "energy_std": 0.0,
        "max_bond_dim": int(np.max(np.asarray(qc.get_bond_dimensions()))),
        "n_states": 1,
    }


def measure_enum(ansatz, params):
    """Discarded weight of the enumerated arm, from the branch norms.

    This arm needs no normalize_state trick and no sampling, because its Kraus
    set is exactly trace preserving. With

        K0 = cos(t/2) I,  K1 = sin(t/2)|0><0|,  K2 = sin(t/2)|0><1|

    we have sum_j Kj^dag Kj = cos^2(t/2) I + sin^2(t/2)(|0><0| + |1><1|) = I, and
    the branch builder never normalises, so in exact arithmetic

        sum_j ||phi_j||^2 = 1

    identically. Whatever is missing from that sum is exactly what truncation
    discarded -- across the whole channel, not one sampled branch of it. It is
    also free: that sum is already the denominator of
    enumerated_energy_from_params.

    Branches are walked one at a time rather than vmapped. There is no autodiff
    here, so only one branch is ever resident, which keeps a 729-branch 4x3
    measurement to the memory cost of a single circuit.
    """
    branches = np.asarray(ansatz._reset_branches)
    num = 0.0
    den = 0.0
    maxbds = []
    weights = []
    for branch in branches:
        qc = ansatz._circuit_branch(params, jnp.asarray(branch))
        w = float(np.real(np.asarray(qc.get_norm())) ** 2)
        num += float(ansatz._energy_of_circuit(qc))
        den += w
        weights.append(w)
        maxbds.append(int(np.max(np.asarray(qc.get_bond_dimensions()))))
    weights = np.asarray(weights)
    # Guard the ratio: at a hopeless bond dimension the norms can collapse.
    energy_normalised = float(num / den) if den > 1e-12 else float("nan")
    return {
        "discarded_weight": float(1.0 - den),
        "discarded_weight_mean": float(1.0 - den),
        "energy": float(num),
        "energy_normalised": energy_normalised,
        # Spread of branch probabilities, not of energies: it says how much of
        # the channel rides on the few branches that carry real weight.
        "energy_std": float(np.std(weights)),
        "max_bond_dim": int(np.max(maxbds)),
        "n_states": int(len(branches)),
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


def run_lattice(Lx, Ly, rows, use_optimal_ordering=True, extra_params=None,
                modes=MODES):
    key = f"{Lx}x{Ly}"
    facts = lattice_facts(Lx, Ly)
    print(f"\n{'=' * 78}")
    print(f"{key}: nq={facts['nq']} R={facts['R']} | chain traj/enum={facts['traj_chain']} "
          f"pur={facts['pur_chain']} | exact caps traj/enum={facts['traj_cap']} "
          f"pur={facts['pur_cap']} | enum branches={facts['nbranches']} | E0={facts['E0']}")
    print("=" * 78)

    converged = converged_params_3x3() if key == "3x3" else None

    for mode in modes:
        if (key, mode) not in LADDERS:
            print(f"\n  --- {key} {mode}: no ladder defined, skipping ---")
            continue
        ladder = [bd for bd in LADDERS[(key, mode)]]
        cap = facts[f"{mode}_cap"]
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
            print(f"  {'bd':>5} {'discarded wt':>14} {'E (raw)':>14} {'E (norm)':>14} "
                  f"{'max bd':>7} {'sec':>7}")
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
                      f"{m['energy_normalised']:>14.6f} {m['max_bond_dim']:>7} {dt:>7.2f}{flag}")
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
    parser.add_argument("--modes", default=",".join(MODES),
                        help=f"comma-separated subset of {','.join(MODES)}")
    parser.add_argument("--out", default="outputs/three_resets/bond_dim_requirement.csv")
    parser.add_argument("--natural-ordering", action="store_true",
                        help="control run with use_optimal_ordering=False, so all three paths "
                             "share the natural system-qubit order")
    parser.add_argument("--from-checkpoint", default=None,
                        help="also measure at parameters sampled along a real training run "
                             "(a checkpoint.npz written by optimize())")
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        parser.error(f"unknown mode(s) {unknown}; choose from {MODES}")

    print(f"jax devices: {jax.devices()}")
    print(f"gate: discarded weight <= {GATE:g}; trajectory samples per rung: {N_TRAJ_SAMPLES}")
    print(f"modes: {', '.join(modes)}")

    extra = checkpoint_param_sets(args.from_checkpoint) if args.from_checkpoint else None

    rows = []
    for spec in args.lattices.split(","):
        Lx, Ly = (int(v) for v in spec.strip().split("x"))
        run_lattice(Lx, Ly, rows, use_optimal_ordering=not args.natural_ordering,
                    extra_params=extra, modes=modes)

    if rows:
        import csv
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        # Rows from different arms carry the same keys, but write the union
        # defensively so a future arm-specific column cannot silently truncate.
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
