"""Count the SWAP gates an MPS circuit issues, per reset implementation.

Run:
    python -m src.diagnostics.swap_counting --lattice 3x3
    python -m src.diagnostics.swap_counting --lattice 3x3 --natural-ordering
    python -m src.diagnostics.swap_counting --lattice 3x3 --cartan-mode separate
    python -m src.diagnostics.swap_counting --lattice 3x3 --modes traj --demo-jit

WHERE SWAPS COME FROM
---------------------
An MPS can only apply a two-qubit gate to ADJACENT sites. When a gate's two
qubits sit further apart, MPSCircuit.apply_double_gate walks one of them along
the chain, applies the gate, and walks it back:

    consecutive_swap(index1, index2 - 1)      # d - 1 SWAP gates in
    apply_adjacent_double_gate(gate, ...)     # the gate itself
    consecutive_swap(index2 - 1, index1)      # d - 1 SWAP gates back out

so a gate spanning distance d = |index1 - index2| costs 2*(d-1) SWAPs, and each
SWAP is itself a two-site gate application with its own SVD and its own
truncation. That is why the qubit ordering search in
ansatz_classes._build_interleaved_ordering minimises the maximum claw distance:
it is minimising this.

Toffolis do NOT swap. With toffoli_mode="direct" (the default) a 3-qubit gate
goes through apply_nqubit_gate, which builds an MPO spanning the qubits and
applies it in place -- no walking. With toffoli_mode="decomposed" the CCX becomes
a sequence of 2-qubit gates, which do swap. Both are counted here, separately.

THE THING THAT TRIPS PEOPLE UP
------------------------------
The swap count is a STATIC property of the circuit, fixed when the program is
traced. The Python that decides "are these adjacent, how far apart are they" runs
during tracing, not during execution: jit compiles the resulting fixed sequence of
tensor operations once, and every later step replays it. So a counter wrapped
around consecutive_swap increments on the FIRST call to a jitted function (the
compile) and never again, no matter how many optimisation steps run.

That is not a limitation of the counting -- it is the answer. "SWAPs per training
step" is the same number every step, and it equals what one traced forward pass
issues. This module demonstrates that rather than asserting it: --demo-jit calls
the real jitted value_and_grad three times and shows the count moving only once.

vmap over trials does not multiply it either. All trials execute the same program
on batched tensors, so the gate sequence is issued once regardless of `trials`.
The per-step COST scales with trials; the per-step gate COUNT does not.

WHAT TO COMPARE
---------------
  swap_gates            total SWAPs issued in one forward pass -- the headline
  two_site_applications  every adjacent two-site gate, SWAPs included. Each is one
                        SVD, so this tracks the real splitting work
  non_swap_two_site     the same minus SWAPs: the "useful" two-site gates
  swap_fraction         swap_gates / two_site_applications -- how much of the
                        splitting work is spent merely moving qubits around
  nqubit_gates          MPO applications (Toffolis in direct mode); these span
                        several sites and cost (k-1) SVDs each, but no SWAPs
  nqubit_site_span      total sites those MPOs cover. Read WITH nqubit_gates:
                        8 gates over 24 sites means every Toffoli is tight on 3
                        adjacent sites; over 72 means they are stretched

MEASURED AT 3x3 (nlayers=2, reset_layers=[1], fused Cartan, direct Toffoli)
--------------------------------------------------------------------------
                     SWAPs   MPO span
  traj  optimised       64          0
  traj  natural        144          0
  pur   optimised      148         24
  pur   natural        144         72

Two things fall out of this that are not obvious.

Purified issues 2.3x the SWAPs of trajectory to apply the SAME 24 Cartan gates.
Both arms do identical useful work; the difference is purely that purified's
gates span further, because its chain is 20 sites rather than 12 and its ordering
has to reserve two slots per reset event.

And the ordering search is doing something different for each arm. For trajectory
there are no Toffolis, so it is pure claw-distance minimisation and it wins big:
144 -> 64. For purified it makes the SWAP count slightly WORSE (144 -> 148) while
cutting the Toffoli MPO span by 3x (72 -> 24). It is buying tight Toffolis at the
price of stretched Cartan blocks -- which is the right trade, since an MPO across
k sites costs (k-1) splits at full bond dimension, and it is consistent with
docs/three_reset_implementations.md section 2.1 finding that without the search
the purified arm never clears its accuracy gate at all.
"""

import argparse
import contextlib
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np

from src.diagnostics.mps_complexity import MODES, build_ansatz


def tap_swaps():
    """Wrap MPSCircuit's gate-application methods to count what they issue.

    Returns (counts, restore). Call restore() to undo the patch -- always in a
    finally block, since leaving these wrappers installed would silently slow and
    miscount every later circuit build in the process.

    Patching the CLASS rather than an instance is deliberate: the circuit objects
    are created inside the ansatz builders, so there is no instance to reach in
    from out here. Same approach as svd_call_counting.tap_backends.
    """
    from tensorcircuit.mpscircuit import MPSCircuit

    counts = {
        "swap_gates": 0,
        "consecutive_swap_calls": 0,
        "two_site_applications": 0,
        "nqubit_gates": 0,
        "nqubit_site_span": 0,
        "single_site_gates": 0,
        "max_swap_distance": 0,
    }

    orig_cs = MPSCircuit.consecutive_swap
    orig_adg = MPSCircuit.apply_adjacent_double_gate
    orig_nq = MPSCircuit.apply_nqubit_gate
    orig_asg = MPSCircuit.apply_single_gate

    def consecutive_swap(self, index_from, index_to, split=None):
        # consecutive_swap walks one site at a time, so it issues exactly
        # |index_to - index_from| SWAP gates. Each of those goes on to call
        # apply_adjacent_double_gate, which the wrapper below counts separately --
        # so two_site_applications already includes them and must not be
        # incremented here as well.
        d = abs(int(index_to) - int(index_from))
        counts["consecutive_swap_calls"] += 1
        counts["swap_gates"] += d
        counts["max_swap_distance"] = max(counts["max_swap_distance"], d)
        return orig_cs(self, index_from, index_to, split=split)

    def apply_adjacent_double_gate(self, gate, index1, index2, center_position=None, split=None):
        counts["two_site_applications"] += 1
        return orig_adg(self, gate, index1, index2,
                        center_position=center_position, split=split)

    def apply_nqubit_gate(self, gate, *index, split=None):
        # Count ONLY the terminal call. apply_nqubit_gate calls ITSELF when the
        # gate's qubit indices are not already sorted ascending: it transposes
        # the gate, sorts the indices and re-enters, then returns. Counting on
        # entry therefore scores every out-of-order gate twice -- which is what
        # reported 16 Toffolis at 3x3 for the 8 that are actually issued. The
        # non-recursing call is the one whose indices are already ascending,
        # which is exactly the test apply_nqubit_gate itself makes.
        if np.all(np.diff(index) > 0):
            counts["nqubit_gates"] += 1
            # Span in sites. An MPO across k sites costs (k-1) two-site splits,
            # so this is what to weigh against the SWAP count -- but note it is
            # NOT swaps: apply_MPO moves the orthogonality centre through
            # reduce_dimension and never calls consecutive_swap.
            counts["nqubit_site_span"] += int(max(index) - min(index) + 1)
        return orig_nq(self, gate, *index, split=split)

    def apply_single_gate(self, gate, index):
        counts["single_site_gates"] += 1
        return orig_asg(self, gate, index)

    MPSCircuit.consecutive_swap = consecutive_swap
    MPSCircuit.apply_adjacent_double_gate = apply_adjacent_double_gate
    MPSCircuit.apply_nqubit_gate = apply_nqubit_gate
    MPSCircuit.apply_single_gate = apply_single_gate

    def restore():
        MPSCircuit.consecutive_swap = orig_cs
        MPSCircuit.apply_adjacent_double_gate = orig_adg
        MPSCircuit.apply_nqubit_gate = orig_nq
        MPSCircuit.apply_single_gate = orig_asg

    return counts, restore


@contextlib.contextmanager
def counting_swaps():
    """Context-manager form of tap_swaps. Yields the live counts dict."""
    counts, restore = tap_swaps()
    try:
        yield counts
    finally:
        restore()


def derived(counts):
    """The two figures worth quoting, added to a raw counts dict."""
    out = dict(counts)
    total = out["two_site_applications"]
    out["non_swap_two_site"] = total - out["swap_gates"]
    out["swap_fraction"] = (out["swap_gates"] / total) if total else float("nan")
    return out


def expected_swaps(ansatz, cartan_mode="fused"):
    """SWAP count predicted from the chain ordering alone, with no instrumentation.

    This is the cross-check that makes the measured number trustworthy. A
    two-qubit gate spanning distance d is walked in and walked back out, costing
    2*(d-1) SWAPs, and `ansatz.mps_claws` already holds the chain positions of
    every Cartan block after the ordering search. So the whole count is:

        nlayers * sum over claws of 2 * (|i - j| - 1)

    with no monkeypatching involved at all. If this and the instrumented count
    disagree, one of them is wrong and neither is reportable.

    cartan_mode="fused" applies each claw as a single qc.unitary, so one gate per
    claw. "separate" applies six single-qubit rotations (free -- single-site gates
    never swap) plus rxx, ryy and rzz, which is THREE two-qubit gates over the
    same pair, hence three times the swaps.

    Resets contribute nothing in either arm: trajectory writes its reset tensor in
    place at the orthogonality centre, purified uses a single-site ry plus
    Toffolis, and Toffolis go through apply_MPO, which never calls
    consecutive_swap.
    """
    per_pair = {"fused": 1, "separate": 3}[cartan_mode]
    per_layer = sum(2 * (abs(int(i) - int(j)) - 1) for i, j in ansatz.mps_claws)
    return ansatz.nlayers * per_layer * per_pair


def count_forward(ansatz, params, mode, status=None):
    """SWAPs issued by ONE forward pass, built eagerly (no jit).

    Eager is the right way to take this measurement: the count is a property of
    the traced program, and tracing it eagerly issues exactly the same gate
    sequence a jitted build would, without the compile.
    """
    params = np.asarray(params)
    with counting_swaps() as counts:
        if mode in ("traj", "none"):
            if status is None:
                status = np.zeros(ansatz.total_resets)
            ansatz._circuit_traj(params, np.asarray(status))
        elif mode == "enum":
            # One branch. Every branch issues the same gate sequence -- the branch
            # index only selects a 2x2 Kraus factor applied at the orthogonality
            # centre, which needs no SVD and no swap -- so one is representative.
            ansatz._circuit_branch(params, np.zeros(ansatz.total_resets, dtype=int))
        else:
            ansatz._circuit(params)
        return derived(counts)


def demo_jit(ansatz, params, mode, repeats=3):
    """Show that the count is fixed at trace time, not incurred per step.

    Calls the ansatz's REAL jitted value_and_grad -- the same callable
    optimize() drives -- `repeats` times, and reports the swap count accumulated
    during each call. The first call compiles and therefore traces; the rest
    replay the compiled program and issue no Python-level gate calls at all.
    """
    import jax
    import jax.numpy as jnp

    trials = ansatz.trials
    batch = jnp.asarray(np.tile(np.asarray(params), (trials, 1)))
    per_call = []
    for i in range(repeats):
        with counting_swaps() as counts:
            if mode in ("traj", "none"):
                key = jax.random.PRNGKey(i)
                status = jax.random.uniform(
                    key, (trials, ansatz.n_trajectories, ansatz.total_resets))
                _, g = ansatz._cost_vvag(batch, status, jnp.zeros(trials))
            else:
                _, g = ansatz._cost_vvag(batch)
            jax.block_until_ready(g)
            per_call.append(counts["swap_gates"])
    return per_call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lattice", default="3x3")
    parser.add_argument("--modes", default="traj,pur")
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--reset-layers", default="[1]")
    parser.add_argument("--bond-dim", type=int, default=64)
    parser.add_argument("--natural-ordering", action="store_true",
                        help="disable the qubit-ordering search, to show what it buys")
    parser.add_argument("--cartan-mode", default="fused", choices=("fused", "separate"),
                        help="'fused' applies each Cartan block as one qc.unitary; "
                             "'separate' applies rxx/ryy/rzz as three gates over the same "
                             "pair, so ~3x the SWAPs")
    parser.add_argument("--toffoli-mode", default="direct", choices=("direct", "decomposed"),
                        help="'direct' applies CCX as an MPO (no SWAPs); 'decomposed' "
                             "expands it into 2-qubit gates (which do SWAP)")
    parser.add_argument("--demo-jit", action="store_true",
                        help="call the real jitted value_and_grad 3x to show the count "
                             "is incurred at trace time, not per step")
    args = parser.parse_args()

    import ast
    reset_layers = ast.literal_eval(args.reset_layers)
    Lx, Ly = (int(v) for v in args.lattice.split("x"))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        parser.error(f"unknown mode(s) {unknown}; choose from {MODES}")

    print(f"{args.lattice}  nlayers={args.nlayers}  reset_layers={reset_layers}  "
          f"bond_dim={args.bond_dim}  cartan_mode={args.cartan_mode}  "
          f"toffoli_mode={args.toffoli_mode}  "
          f"ordering={'natural' if args.natural_ordering else 'optimised'}")
    print("\nSWAP gates issued by ONE forward pass (= one training step; see module docstring)\n")
    print(f"  {'arm':>5} {'chain':>6} {'SWAPs':>7} {'2-site':>8} {'non-swap':>9} "
          f"{'swap %':>7} {'MPO':>5} {'1-site':>7} {'max dist':>9} {'predicted':>10} {'':>7}")

    rows = {}
    agree = True
    for mode in modes:
        ansatz = build_ansatz(Lx, Ly, mode, nlayers=args.nlayers,
                              reset_layers=reset_layers, cap=args.bond_dim,
                              use_optimal_ordering=not args.natural_ordering)
        # toffoli_mode only reaches the purified builder; set it after
        # construction so build_ansatz keeps one signature for every arm.
        ansatz.toffoli_mode = args.toffoli_mode
        ansatz.cartan_mode = args.cartan_mode
        c = count_forward(ansatz, np.asarray(ansatz.initparams[0]), mode)
        rows[mode] = (ansatz, c)
        # Predicted from ansatz.mps_claws alone, with no instrumentation. A
        # mismatch means either the wrappers or the model of what the circuit
        # does is wrong, and the measured number must not be quoted.
        pred = expected_swaps(ansatz, args.cartan_mode)
        ok = "OK" if pred == c["swap_gates"] else "MISMATCH"
        agree = agree and (pred == c["swap_gates"])
        print(f"  {mode:>5} {ansatz.n_mps_qubits:>6} {c['swap_gates']:>7} "
              f"{c['two_site_applications']:>8} {c['non_swap_two_site']:>9} "
              f"{100 * c['swap_fraction']:>6.1f}% {c['nqubit_gates']:>5} "
              f"{c['single_site_gates']:>7} {c['max_swap_distance']:>9} "
              f"{pred:>10} {ok:>7}")

    print(f"\n  'predicted' comes from ansatz.mps_claws with no instrumentation: "
          f"nlayers * sum 2*(d-1)\n  over the Cartan pairs. It is the check that the "
          f"measured count means what it says.")
    if not agree:
        print("\n  *** MISMATCH: do not quote these counts until it is resolved. ***")

    if len(rows) == 2 and "traj" in rows and "pur" in rows:
        st, sp = rows["traj"][1]["swap_gates"], rows["pur"][1]["swap_gates"]
        if st:
            print(f"\n  purified issues {sp / st:.2f}x the SWAPs of single trajectory "
                  f"({sp} vs {st}) to apply the same "
                  f"{rows['traj'][1]['non_swap_two_site']} Cartan gates.")

    if args.demo_jit:
        print("\nSWAPs counted during 3 consecutive calls to the REAL jitted "
              "value_and_grad:\n")
        for mode in modes:
            ansatz, _ = rows[mode]
            per_call = demo_jit(ansatz, np.asarray(ansatz.initparams[0]), mode)
            print(f"  {mode:>5}: {per_call}   <- call 1 traces and compiles; "
                  f"calls 2-3 replay the compiled program")
        print("\n  The count is a property of the compiled program, not of the step.\n"
              "  Every optimisation step issues the same gate sequence, so 'SWAPs per\n"
              "  step' is the forward-pass number in the table above.")


if __name__ == "__main__":
    main()
