# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Equivalence tests for enumerated (ancilla-free, deterministic) resets.

The reference is again the *purified* reset path in production
(`construct_dyn_circuit_toriccodelattice_prob_resets`: ry(coin) + two Toffolis,
two ancillas per reset, ancillas traced out at expectation time). Where the
trajectory path samples one branch of the reset channel, the enumerated path
evaluates ALL 3^total_resets Kraus branches and recombines them, so it should
match the purified path *exactly* -- not merely in expectation.

The estimator under test is

    E = sum_j <phi_j|H|phi_j> / sum_j <phi_j|phi_j>

over unnormalised branch states |phi_j>. Two facts make this the channel energy:
||phi_j||^2 is the branch probability P_j, and <phi_j|H|phi_j> is P_j * E_j, so
the numerator is sum_j P_j E_j = Tr[H E(rho)] and the denominator is 1.

Test 3 is the acceptance criterion: enumerated energy == purified energy.
Test 5 is the one that motivates the whole mode -- the reset-theta gradient is
correct under *plain* autodiff, with no DiCE magic box, because theta now enters
each branch state as a literal cos(theta/2)/sin(theta/2) Kraus prefactor rather
than only through a sampling probability.

Run:
    source /home/s1931382/dpqc_venv/bin/activate
    python -m src.diagnostics.test_enumerated_resets
"""

import itertools
import sys

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.generate_ansatz import (
    _branch_reset,
    _branch_reset_matrix,
    _normalize_mps_if_requested,
)

K = tc.set_backend("jax")

# use_optimal_ordering=False keeps the system qubits in natural order on both
# sides, so the purified circuit merely appends its ancillas at the end of the
# chain and the two energies are directly comparable.
BOND_DIM = 16          # lossless at 2x2 (needs >= 16; see project memory)
FAIL = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)
    return ok


def make_pair(**kw):
    """Build (purified, enumerated) ansatze that share a parameter layout."""
    common = dict(trials=1, bond_dim=BOND_DIM, use_optimal_ordering=False,
                  seed=1234, use_prob_resets_ansatz=True, **kw)
    purified = ToricCodeAnsatz(use_trajectory_resets=False, **common)
    enum = ToricCodeAnsatz(use_trajectory_resets=False, use_enumerated_resets=True,
                           **common)
    assert purified.nparams == enum.nparams, "parameter layouts must match"
    return purified, enum


def branch_sums(enum, params):
    """(sum_j <phi_j|H|phi_j>, sum_j <phi_j|phi_j>) by an explicit Python loop.

    Deliberately does NOT use the vmapped production path, so that the two can
    be cross-checked against each other.
    """
    num, den = 0.0, 0.0
    for branch in itertools.product(range(3), repeat=enum.total_resets):
        qc = enum._circuit_branch(params, jnp.asarray(branch, dtype=jnp.int32))
        num += float(K.real(enum._energy_of_circuit(qc)))
        den += float(K.real(qc.get_norm()) ** 2)
    return num, den


# ---------------------------------------------------------------------------


def test_kraus_completeness():
    """sum_j K_j^dag K_j == I, for the matrices the builder actually applies."""
    print("\n1. Kraus completeness of _branch_reset_matrix")
    worst = 0.0
    for theta in [0.0, 0.3, np.pi / 4, np.pi / 2, 2.0, np.pi]:
        total = np.zeros((2, 2), dtype=complex)
        for j in range(3):
            onehot = jnp.asarray(np.eye(3)[j])
            m = np.asarray(_branch_reset_matrix(jnp.asarray(theta), onehot))
            total += m.conj().T @ m
        worst = max(worst, np.abs(total - np.eye(2)).max())
    check("sum_j K_j^dag K_j == I over a range of thetas", worst < 1e-14,
          f"max |deviation| = {worst:.3e}")

    # The three branches must be the operators the docstring claims.
    th = 0.7
    c, s = np.cos(th / 2), np.sin(th / 2)
    want = [c * np.eye(2),
            s * np.array([[1, 0], [0, 0]]),
            s * np.array([[0, 1], [0, 0]])]
    worst = max(
        np.abs(np.asarray(_branch_reset_matrix(jnp.asarray(th), jnp.asarray(np.eye(3)[j])))
               - want[j]).max()
        for j in range(3)
    )
    check("K_0 = cos(th/2) I, K_1 = sin(th/2)|0><0|, K_2 = sin(th/2)|0><1|",
          worst < 1e-15, f"max |deviation| = {worst:.3e}")


def test_weights_sum_to_one():
    """sum_j <phi_j|phi_j> == 1: the channel is trace preserving.

    The single best end-to-end sanity check -- it exercises the builder, the MPS
    canonical form and get_norm() at once, and it is exactly the quantity the
    estimator's denominator relies on.
    """
    print("\n2. Branch probabilities sum to one")
    _, enum = make_pair(Lx=2, Ly=2, nlayers=2, h=0.3, reset_layers=None)
    params = enum.initparams[0]
    num, den = branch_sums(enum, params)
    check(f"sum of {3 ** enum.total_resets} branch probabilities == 1",
          abs(den - 1.0) < 1e-12, f"sum = {den!r}")
    return enum, params, num, den


def test_matches_purified():
    """THE acceptance criterion: enumerated energy == purified energy."""
    print("\n3. Enumerated energy vs purified ancilla energy")
    for label, reset_layers in [("resets on last layer", [1]),
                                ("resets on every layer", None)]:
        purified, enum = make_pair(Lx=2, Ly=2, nlayers=2, h=0.3,
                                   reset_layers=reset_layers)
        params = enum.initparams[0]
        e_pur = float(purified.energy_from_params(params))
        e_enum = float(enum.energy_from_params(params))
        check(f"energies agree ({label}, {3 ** enum.total_resets} branches)",
              abs(e_pur - e_enum) < 1e-10,
              f"purified {e_pur!r} vs enumerated {e_enum!r} (diff {abs(e_pur-e_enum):.2e})")

        # The vmapped production path must agree with a plain Python loop.
        num, den = branch_sums(enum, params)
        check(f"vmapped path == explicit branch loop ({label})",
              abs(num / den - e_enum) < 1e-12,
              f"loop {num/den!r} vs vmap {e_enum!r}")

        check(f"chain is ancilla-free ({label})",
              enum.n_mps_qubits == enum.lattice.num_qubits and enum.nancillas == 0,
              f"{enum.n_mps_qubits} sites vs purified {purified.n_mps_qubits}")


def test_matches_trajectory_enumeration():
    """Cross-check against the independent trajectory implementation.

    The trajectory builder reaches the same branch states by a different route
    (Born probabilities computed from the site tensor, then an explicit
    normalisation), so agreement here catches an error that is somehow common to
    both the enumerated builder and the purified one.
    """
    print("\n4. Enumerated energy vs weighted trajectory enumeration")
    common = dict(Lx=2, Ly=2, nlayers=2, h=0.3, trials=1, bond_dim=BOND_DIM,
                  use_optimal_ordering=False, seed=1234,
                  use_prob_resets_ansatz=True, reset_layers=None)
    traj = ToricCodeAnsatz(use_trajectory_resets=True, **common)
    enum = ToricCodeAnsatz(use_trajectory_resets=False, use_enumerated_resets=True,
                           **common)
    params = enum.initparams[0]

    total_p, weighted = 0.0, 0.0
    for force in itertools.product(range(3), repeat=traj.total_resets):
        qc, log_w = traj._circuit_traj(params, jnp.zeros(traj.total_resets),
                                       force=list(force))
        _normalize_mps_if_requested(qc, traj.normalize_state)
        p = float(jnp.exp(log_w))
        total_p += p
        weighted += p * float(K.real(traj._energy_of_circuit(qc)))

    e_enum = float(enum.energy_from_params(params))
    check("trajectory branch probabilities sum to 1", abs(total_p - 1.0) < 1e-10,
          f"sum = {total_p!r}")
    check("sum_j P_j E_j (trajectory) == enumerated energy",
          abs(weighted - e_enum) < 1e-10,
          f"trajectory {weighted!r} vs enumerated {e_enum!r}")


def test_theta_gradient():
    """Plain autodiff gives the correct reset-theta gradient -- no DiCE needed."""
    print("\n5. Reset-theta gradient under plain autodiff")
    purified, enum = make_pair(Lx=2, Ly=2, nlayers=2, h=0.3, reset_layers=None)
    params = enum.initparams[0]

    g = np.asarray(jax.grad(enum.energy_from_params)(params))
    g_pur = np.asarray(jax.grad(purified.energy_from_params)(params))
    idx = np.asarray(enum.reset_param_indices)

    eps = 1e-6
    worst_rel = 0.0
    for i in idx:
        fd = (float(enum.energy_from_params(params.at[i].add(eps)))
              - float(enum.energy_from_params(params.at[i].add(-eps)))) / (2 * eps)
        worst_rel = max(worst_rel, abs(g[i] - fd) / max(abs(fd), 1e-12))
    check("autodiff theta gradient matches central finite differences",
          worst_rel < 1e-6, f"worst relative error = {worst_rel:.2e}")

    # Without this the mode would be silently useless, exactly as the naive
    # (box-free) trajectory estimator is.
    gnorm = float(np.linalg.norm(g[idx]))
    check("theta gradient is not identically zero", gnorm > 1e-8,
          f"|grad_theta| = {gnorm:.6f}")

    worst = float(np.abs(g - g_pur).max())
    check("full gradient matches the purified circuit's", worst < 1e-7,
          f"max |enumerated - purified| = {worst:.3e}")


def test_deterministic_and_chunked():
    """No sampling anywhere, and the memory-saving chunked path agrees."""
    print("\n6. Determinism and branch chunking")
    _, enum = make_pair(Lx=2, Ly=2, nlayers=2, h=0.3, reset_layers=None)
    params = enum.initparams[0]

    e1 = float(enum.energy_from_params(params))
    e2 = float(enum.energy_from_params(params))
    check("repeated evaluations are bit-identical", e1 == e2, f"{e1!r} vs {e2!r}")

    nbranch = 3 ** enum.total_resets
    chunk = 3
    enum.branch_chunk_size = chunk
    e_chunked = float(enum.energy_from_params(params))
    enum.branch_chunk_size = None
    check(f"branch_chunk_size={chunk} agrees with the wide vmap over {nbranch} branches",
          abs(e_chunked - e1) < 1e-13, f"{e_chunked!r} vs {e1!r}")

    try:
        enum.branch_chunk_size = 2      # does not divide 9
        enum.energy_from_params(params)
        ok = False
    except ValueError:
        ok = True
    finally:
        enum.branch_chunk_size = None
    check("a chunk size that does not divide the branch count raises", ok)


def test_no_bond_growth():
    """A Kraus branch is a single-site operator: bond dimensions are untouched."""
    print("\n7. Resets cause no bond growth")
    _, enum = make_pair(Lx=2, Ly=2, nlayers=2, h=0.3, reset_layers=[1])
    params = enum.initparams[0]
    qc = enum._circuit_branch(params, jnp.zeros(enum.total_resets, dtype=jnp.int32))
    before = list(np.asarray(qc.get_bond_dimensions()))
    for j in range(3):
        qc = enum._circuit_branch(params, jnp.zeros(enum.total_resets, dtype=jnp.int32))
        _branch_reset(qc, enum.mps_reset_qubits[0], params[enum.reset_param_indices[0]],
                      jnp.asarray(j, dtype=jnp.int32))
        after = list(np.asarray(qc.get_bond_dimensions()))
        check(f"bond dimensions unchanged by branch {j}", after == before,
              f"{before} -> {after}")


def test_zero_probability_branch_gradient():
    """A branch with probability exactly zero must not NaN the gradient.

    Regression test. The estimator's denominator was originally computed as
    qc.get_norm()**2, which is the right VALUE but the wrong gradient: get_norm
    is sqrt(<phi|phi>), and d(sqrt(x))/dx = 1/(2 sqrt(x)) is infinite at x = 0,
    so autodiff produced NaN on any branch of probability zero -- which is a
    routine occurrence (measuring 1 on a qubit already in |0>). The NaN then
    propagated through the sum and froze the whole trial. Summing |T|^2 off the
    orthogonality centre is the same number with a finite gradient.
    """
    print("\n8. Zero-probability branches keep the gradient finite")

    # The composition itself, isolated: this is the shape of the old bug.
    z = jnp.zeros(4)
    via_norm = jax.grad(lambda x: jnp.linalg.norm(x) ** 2)(z)
    direct = jax.grad(lambda x: jnp.sum(x * jnp.conj(x)).real)(z)
    check("sqrt-then-square gradient is NaN at zero (the bug)",
          bool(jnp.all(jnp.isnan(via_norm))), f"{via_norm}")
    check("direct sum-of-squares gradient is finite at zero (the fix)",
          bool(jnp.all(jnp.isfinite(direct))), f"{direct}")

    # End to end: every branch's gradient contribution must be finite, and a
    # dead branch must contribute exactly zero rather than a NaN.
    _, enum = make_pair(Lx=2, Ly=2, nlayers=2, h=0.3, reset_layers=None)
    params = enum.initparams[0]
    g = np.asarray(jax.grad(enum.energy_from_params)(params))
    check("full gradient is finite", bool(np.isfinite(g).all()),
          f"{int((~np.isfinite(g)).sum())} non-finite entries of {g.size}")

    # Force a state whose reset qubit is exactly |0>, so branch 2 (measure 1)
    # has identically zero weight, and differentiate through it.
    def dead_branch_weight(p):
        qc = enum._circuit_branch(p, jnp.asarray([2, 2], dtype=jnp.int32))
        centre = qc._mps.tensors[qc._mps.center_position]
        return K.real(K.sum(centre * K.conj(centre)))

    gd = np.asarray(jax.grad(dead_branch_weight)(params))
    check("gradient through a doubly-projected branch is finite",
          bool(np.isfinite(gd).all()),
          f"{int((~np.isfinite(gd)).sum())} non-finite entries")


def test_branch_cap_raises():
    """An accidentally huge 3^R must fail loudly at construction time."""
    print("\n8. max_reset_branches guard")
    try:
        ToricCodeAnsatz(Lx=3, Ly=3, nlayers=2, h=0.0, trials=1, bond_dim=BOND_DIM,
                        use_optimal_ordering=False, seed=1, use_prob_resets_ansatz=True,
                        reset_layers=None, use_trajectory_resets=False,
                        use_enumerated_resets=True)
        check("3^8 = 6561 branches is rejected", False, "no exception raised")
    except ValueError as exc:
        check("3^8 = 6561 branches is rejected", "max_reset_branches" in str(exc),
              str(exc).split(".")[0])

    try:
        ToricCodeAnsatz(Lx=2, Ly=2, nlayers=2, h=0.0, trials=1, bond_dim=BOND_DIM,
                        use_optimal_ordering=False, seed=1, use_prob_resets_ansatz=True,
                        reset_layers=[1], use_trajectory_resets=True,
                        use_enumerated_resets=True)
        check("enumerated + trajectory together is rejected", False, "no exception raised")
    except ValueError as exc:
        check("enumerated + trajectory together is rejected",
              "mutually exclusive" in str(exc))


def test_end_to_end():
    """A short real optimisation: energy descends and the thetas move."""
    print("\n9. End-to-end training")
    ansatz = ToricCodeAnsatz(Lx=2, Ly=2, nlayers=2, h=0.0, trials=2, maxiter=10,
                             bond_dim=BOND_DIM, use_optimal_ordering=False, seed=5,
                             use_prob_resets_ansatz=True, reset_layers=[1],
                             use_trajectory_resets=False, use_enumerated_resets=True)
    reset_idx = np.asarray(ansatz.reset_param_indices)
    theta0 = np.asarray(ansatz.initparams)[:, reset_idx]

    value, params, energies, *_ = ansatz.optimize()
    params = np.asarray(params)

    check("all final values finite", bool(np.isfinite(np.asarray(value)).all()),
          f"energies = {np.asarray(value)}")
    moved = np.abs(params[:, reset_idx] - theta0).max()
    check("reset thetas actually moved", moved > 1e-6,
          f"max |dtheta| over 10 steps = {moved:.3e}")
    e_first, e_last = energies[:, 0], np.asarray(value)
    check("energy decreased on average", e_last.mean() < e_first.mean(),
          f"{e_first.mean():.4f} -> {e_last.mean():.4f} (exact ground state is -5 at 2x2)")


def main():
    print("=" * 78)
    print("Enumerated-reset equivalence tests (reference: purified ancilla path)")
    print("=" * 78)
    test_kraus_completeness()
    test_weights_sum_to_one()
    test_matches_purified()
    test_matches_trajectory_enumeration()
    test_theta_gradient()
    test_deterministic_and_chunked()
    test_no_bond_growth()
    test_zero_probability_branch_gradient()
    test_branch_cap_raises()
    test_end_to_end()

    print("\n" + "=" * 78)
    if FAIL:
        print(f"{len(FAIL)} CHECK(S) FAILED:")
        for name in FAIL:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
