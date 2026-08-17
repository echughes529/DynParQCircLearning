# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Equivalence tests for ancilla-free trajectory resets.

The reference throughout is the *purified* reset path already in production
(`construct_dyn_circuit_toriccodelattice_prob_resets`: ry(coin) + two Toffolis,
two ancillas per reset, ancillas traced out at expectation time). The trajectory
path samples one branch of the same channel with no ancillas at all, so the two
must agree on

  * the branch decomposition   (probabilities and conditional states),
  * the energy                 (exactly, once branches are weighted),
  * the gradient               (exactly, once branches are weighted) -- including
                                the reset thetas, which is the whole point.

Correctness is established by **exact enumeration**, not Monte Carlo: the
`force=` hook on `_circuit_traj` takes a prescribed branch instead of sampling,
so all 3^n_resets trajectories can be visited and weighted by their exact
probabilities. Monte Carlo appears only in test 8, purely to confirm the sampler
draws from the distribution the enumeration already validated.

Test 5 is the load-bearing one: it asserts that WITHOUT the DiCE magic box the
reset-theta gradient is identically zero. That is the failure mode the whole
estimator design exists to avoid, and it is invisible to every other test.

Run:
    source /home/s1931382/dpqc_venv/bin/activate
    python -m src.diagnostics.test_trajectory_resets
"""

import itertools
import sys

import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.generate_ansatz import _normalize_mps_if_requested

K = tc.set_backend("jax")

# Both ansatze are built with use_optimal_ordering=False so the system qubits sit
# in natural order 0..nq-1 on both sides and the purified circuit appends its
# ancillas at the end of the chain. That makes the two wavefunctions directly
# comparable: projecting the purified state's trailing ancilla indices leaves the
# system qubits in the same order the trajectory state uses.
BOND_DIM = 16          # lossless at 2x2 (needs >= 16; see project memory)
FAIL = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)
    return ok


def make_pair(**kw):
    """Build (purified, trajectory) ansatze that share a parameter layout."""
    common = dict(trials=1, bond_dim=BOND_DIM, use_optimal_ordering=False,
                  seed=1234, **kw)
    purified = ToricCodeAnsatz(use_trajectory_resets=False, **common)
    traj = ToricCodeAnsatz(use_trajectory_resets=True, **common)
    assert purified.nparams == traj.nparams, "parameter layouts must match"
    return purified, traj


def traj_energy(traj, params, force):
    """Energy and log_w of one forced trajectory (no sampling)."""
    qc, log_w = traj._circuit_traj(params, jnp.zeros(traj.total_resets), force=force)
    _normalize_mps_if_requested(qc, traj.normalize_state)
    return traj._energy_of_circuit(qc), log_w, qc


def enumerate_forced(traj, params):
    """All 3^n_resets branches: list of (force, probability, energy)."""
    out = []
    for force in itertools.product(range(3), repeat=traj.total_resets):
        energy, log_w, _ = traj_energy(traj, params, list(force))
        out.append((list(force), float(jnp.exp(log_w)), float(energy)))
    return out


def dice_forced(traj, params, force, baseline):
    """The production estimator, with the branch forced instead of sampled."""
    qc, log_w = traj._circuit_traj(params, jnp.zeros(traj.total_resets), force=force)
    _normalize_mps_if_requested(qc, traj.normalize_state)
    energy = traj._energy_of_circuit(qc)
    box = K.exp(log_w - K.stop_gradient(log_w))
    return box * (energy - baseline) + baseline


def naive_forced(traj, params, force):
    """Same, but WITHOUT the magic box -- the broken estimator."""
    qc, _ = traj._circuit_traj(params, jnp.zeros(traj.total_resets), force=force)
    _normalize_mps_if_requested(qc, traj.normalize_state)
    return traj._energy_of_circuit(qc)


# ---------------------------------------------------------------------------
# Tests 1-5: one reset, 2x2. Everything exact, nothing sampled.
# ---------------------------------------------------------------------------
def test_single_reset():
    print("\n=== 2x2, nlayers=1, one reset: exact enumeration vs purified path ===")
    purified, traj = make_pair(Lx=2, Ly=2, nlayers=1, reset_layers=[0])
    params = purified.initparams[0]
    nq = purified.lattice.num_qubits

    # --- reference: purified branch decomposition -------------------------
    qc_ref = purified._circuit(params)
    _normalize_mps_if_requested(qc_ref, purified.normalize_state)
    psi = np.asarray(qc_ref.wavefunction()).reshape((2,) * purified.n_mps_qubits)
    e_ref = float(purified.energy_from_params(params))

    # ancilla layout: coin then sink, appended at the chain end
    coin, sink = purified.ancilla_mps_positions
    check("purified chain is nq + 2*total_resets",
          purified.n_mps_qubits == nq + 2 * purified.total_resets,
          f"{purified.n_mps_qubits} == {nq} + 2*{purified.total_resets}")
    check("trajectory chain is exactly nq",
          traj.n_mps_qubits == nq and traj.nancillas == 0,
          f"n_mps_qubits={traj.n_mps_qubits}, nancillas={traj.nancillas}")

    def branch(a_coin, a_sink):
        idx = [slice(None)] * purified.n_mps_qubits
        idx[coin], idx[sink] = a_coin, a_sink
        phi = np.array(psi[tuple(idx)]).reshape(-1)
        p = float(np.real(np.vdot(phi, phi)))
        return phi, p

    # Test 1: the fourth purified outcome is unreachable.
    _, p01 = branch(0, 1)
    check("purified outcome (coin=0, sink=1) has zero probability", p01 < 1e-14,
          f"p = {p01:.2e}  -> 3 reachable branches, not 4")

    # Test 2 + 3: probabilities and conditional states match, branch by branch.
    #   trajectory j=0 <-> coin 0 (no reset); j=1,2 <-> coin 1, sink = measured bit
    mapping = {0: (0, 0), 1: (1, 0), 2: (1, 1)}
    enum = enumerate_forced(traj, params)
    probs_by_j = {f[0]: p for f, p, _ in enum}
    for j, (a_c, a_s) in mapping.items():
        phi, p_ref = branch(a_c, a_s)
        p_traj = probs_by_j[j]
        check(f"branch j={j}: probability matches purified", abs(p_traj - p_ref) < 1e-9,
              f"traj {p_traj:.12f} vs purified {p_ref:.12f}  (diff {abs(p_traj-p_ref):.1e})")
        _, _, qc_j = traj_energy(traj, params, [j])
        psi_j = np.asarray(qc_j.wavefunction()).reshape(-1)
        overlap = abs(np.vdot(phi / np.sqrt(p_ref), psi_j))
        check(f"branch j={j}: state matches purified up to phase", abs(overlap - 1.0) < 1e-8,
              f"|<phi|psi_traj>| = {overlap:.12f}")

    total_p = sum(p for _, p, _ in enum)
    check("branch probabilities sum to 1", abs(total_p - 1.0) < 1e-10,
          f"sum = {total_p:.14f}")

    # Test 4: energy is exactly the probability-weighted branch energy.
    e_enum = sum(p * e for _, p, e in enum)
    check("enumerated energy == purified energy", abs(e_enum - e_ref) < 1e-9,
          f"{e_enum:.12f} vs {e_ref:.12f}  (diff {abs(e_enum - e_ref):.1e})")

    # Test 5: gradient, including the reset thetas.
    reset_idx = traj.reset_param_indices
    g_ref = np.asarray(jax.grad(purified.energy_from_params)(params))
    baseline = -0.5   # deliberately not the mean, to exercise the control variate
    g_dice = np.zeros_like(g_ref)
    for force, p, _ in enum:
        g_dice += p * np.asarray(jax.grad(dice_forced, argnums=1)(traj, params, force, baseline))
    check("expected DiCE gradient == purified gradient (all params)",
          np.max(np.abs(g_dice - g_ref)) < 1e-7,
          f"max abs err = {np.max(np.abs(g_dice - g_ref)):.2e}")
    theta_atol = 1e-8
    check("expected DiCE gradient == purified gradient (reset thetas)",
          np.allclose(g_dice[reset_idx], g_ref[reset_idx], atol=theta_atol),
          f"est {g_dice[reset_idx]} vs exact {g_ref[reset_idx]}")
    # Guards test_without_magic_box against passing vacuously: if the true
    # reset-theta gradient were itself ~0 at this parameter point, "no box gives
    # zero" would prove nothing. The threshold is tied to the tolerance the
    # agreement above was verified at, so the signal must sit orders of magnitude
    # clear of the numerical floor rather than clear of an arbitrary constant.
    theta_signal = np.abs(g_ref[reset_idx]).max()
    check("reference reset-theta gradient is non-negligible (test not vacuous)",
          theta_signal > 1000 * theta_atol,
          f"max|dE/dtheta| = {theta_signal:.4e} > 1000x the {theta_atol:.0e} "
          f"agreement tolerance")

    # baseline must not change the expectation
    g_dice_b0 = np.zeros_like(g_ref)
    for force, p, _ in enum:
        g_dice_b0 += p * np.asarray(jax.grad(dice_forced, argnums=1)(traj, params, force, 0.0))
    check("expected gradient is baseline-invariant",
          np.max(np.abs(g_dice - g_dice_b0)) < 1e-9,
          f"max diff between baseline=-0.5 and 0.0 = {np.max(np.abs(g_dice - g_dice_b0)):.2e}")

    return purified, traj, params, g_ref, enum


def test_without_magic_box(traj, params, g_ref, enum):
    """The definitive test: no magic box => reset thetas get NO gradient."""
    print("\n=== the magic box is load-bearing (this is the failure mode) ===")
    reset_idx = traj.reset_param_indices
    g_naive = np.zeros_like(g_ref)
    for force, p, _ in enum:
        g_naive += p * np.asarray(jax.grad(naive_forced, argnums=1)(traj, params, force))

    check("WITHOUT the box, reset-theta gradient is identically zero",
          np.abs(g_naive[reset_idx]).max() < 1e-14,
          f"max|dE/dtheta| = {np.abs(g_naive[reset_idx]).max():.2e}  "
          f"(exact value is {np.abs(g_ref[reset_idx]).max():.4e})")
    check("WITHOUT the box, upstream (Cartan) gradients are biased too",
          np.max(np.abs(g_naive - g_ref)) > 1e-5,
          f"max abs err = {np.max(np.abs(g_naive - g_ref)):.2e}  "
          f"(log q_b in log_w carries this)")


def test_composition():
    """Two sequential resets: does one global box handle both?"""
    print("\n=== 2x2, nlayers=2, resets in both layers: composition (9 branches) ===")
    purified, traj = make_pair(Lx=2, Ly=2, nlayers=2, reset_layers=None)
    params = purified.initparams[0]
    check("two reset events", traj.total_resets == 2, f"total_resets={traj.total_resets}")

    enum = enumerate_forced(traj, params)
    check("9 enumerated branches", len(enum) == 9, f"{len(enum)} branches")
    total_p = sum(p for _, p, _ in enum)
    check("branch probabilities sum to 1", abs(total_p - 1.0) < 1e-10, f"sum = {total_p:.14f}")

    e_ref = float(purified.energy_from_params(params))
    e_enum = sum(p * e for _, p, e in enum)
    check("enumerated energy == purified energy", abs(e_enum - e_ref) < 1e-9,
          f"{e_enum:.12f} vs {e_ref:.12f}")

    g_ref = np.asarray(jax.grad(purified.energy_from_params)(params))
    g_dice = np.zeros_like(g_ref)
    for force, p, _ in enum:
        g_dice += p * np.asarray(jax.grad(dice_forced, argnums=1)(traj, params, force, -0.5))
    check("expected gradient == purified gradient (all params)",
          np.max(np.abs(g_dice - g_ref)) < 1e-7,
          f"max abs err = {np.max(np.abs(g_dice - g_ref)):.2e}")
    reset_idx = traj.reset_param_indices
    check("both reset thetas correct", np.allclose(g_dice[reset_idx], g_ref[reset_idx], atol=1e-8),
          f"est {g_dice[reset_idx]} vs exact {g_ref[reset_idx]}")


def test_chain_lengths():
    print("\n=== chain length across lattice sizes ===")
    for (Lx, Ly, layers) in [(2, 2, [0]), (3, 2, [1]), (3, 3, [1]), (3, 3, None)]:
        purified, traj = make_pair(Lx=Lx, Ly=Ly, nlayers=2, reset_layers=layers)
        nq = purified.lattice.num_qubits
        ok = (traj.n_mps_qubits == nq
              and purified.n_mps_qubits == nq + 2 * purified.total_resets)
        check(f"{Lx}x{Ly} reset_layers={layers}", ok,
              f"purified {purified.n_mps_qubits} sites -> trajectory {traj.n_mps_qubits} "
              f"(nq={nq}, resets={traj.total_resets})")


def test_canonical_form():
    print("\n=== MPS invariants after a reset ===")
    _, traj = make_pair(Lx=3, Ly=2, nlayers=2, reset_layers=[1])
    params = traj.initparams[0]
    _, _, qc = traj_energy(traj, params, [1] * traj.total_resets)
    check("qc.is_valid()", bool(qc.is_valid()))
    norm = float(abs(qc.get_norm()))
    check("state is normalised", abs(norm - 1.0) < 1e-9, f"norm = {norm:.12f}")

    centre = qc._mps.center_position
    worst_left = worst_right = 0.0
    for site, T in enumerate(qc.get_tensors()):
        T = np.asarray(T)
        if site < centre:      # expect left-isometry
            m = T.reshape(-1, T.shape[2])
            worst_left = max(worst_left,
                             float(np.abs(m.conj().T @ m - np.eye(m.shape[1])).max()))
        elif site > centre:    # expect right-isometry
            m = T.reshape(T.shape[0], -1)
            worst_right = max(worst_right,
                              float(np.abs(m @ m.conj().T - np.eye(m.shape[0])).max()))
    check("sites left of centre are left-isometric", worst_left < 1e-9,
          f"worst deviation = {worst_left:.2e}")
    check("sites right of centre are right-isometric", worst_right < 1e-9,
          f"worst deviation = {worst_right:.2e}")


def test_sampler_monte_carlo():
    """Confirm the SAMPLER draws from the distribution enumeration validated."""
    print("\n=== sampled estimator under jit + vmap (sampler plumbing only) ===")
    purified, traj = make_pair(Lx=2, Ly=2, nlayers=1, reset_layers=[0])
    params = purified.initparams[0]
    e_ref = float(purified.energy_from_params(params))
    g_ref = np.asarray(jax.grad(purified.energy_from_params)(params))
    reset_idx = traj.reset_param_indices

    n = 20000
    status = jax.random.uniform(jax.random.PRNGKey(0), (n, traj.n_trajectories,
                                                       traj.total_resets))
    baseline = jnp.full((n,), e_ref)
    vals, grads = traj._cost_vvag(jnp.broadcast_to(params, (n, traj.nparams)),
                                 status, baseline)
    vals, grads = np.asarray(vals), np.asarray(grads)

    check("all sampled values finite", bool(np.isfinite(vals).all()))
    check("all sampled gradients finite", bool(np.isfinite(grads).all()))

    se = vals.std() / np.sqrt(n)
    z = abs(vals.mean() - e_ref) / se
    check("sampled mean energy is unbiased", z < 4,
          f"mean {vals.mean():.6f} +- {se:.6f} vs exact {e_ref:.6f}  (|z| = {z:.2f})")
    check("energy standard error is small enough to have teeth", se < 0.05, f"se = {se:.4f}")

    g_se = grads.std(0) / np.sqrt(n)
    z_all = np.abs(grads.mean(0) - g_ref) / np.maximum(g_se, 1e-30)
    check("sampled mean gradient is unbiased (all params)", z_all.max() < 4,
          f"worst |z| = {z_all.max():.2f} over {traj.nparams} params")
    z_theta = z_all[reset_idx].max()
    check("sampled mean reset-theta gradient is unbiased", z_theta < 4,
          f"|z| = {z_theta:.2f}, mean {grads[:, reset_idx].mean(0)} vs exact {g_ref[reset_idx]}")


def test_end_to_end():
    print("\n=== short end-to-end optimisation (3x3, 10 steps) ===")
    ansatz = ToricCodeAnsatz(
        Lx=3, Ly=3, nlayers=2, reset_layers=[1], trials=4, bond_dim=64,
        maxiter=10, howoften_tosave=5, use_trajectory_resets=True,
        use_optimal_ordering=True, seed=7, traj_seed=99,
    )
    reset_idx = ansatz.reset_param_indices
    theta0 = np.asarray(ansatz.initparams)[:, reset_idx].copy()
    value, params, energies, *_ = ansatz.optimize()
    params = np.asarray(params)

    check("all final values finite", bool(np.isfinite(np.asarray(value)).all()),
          f"energies = {np.asarray(value)}")
    check("no trials were frozen for non-finite gradients",
          getattr(ansatz, "n_nonfinite_steps", 0) == 0,
          f"n_nonfinite_steps = {getattr(ansatz, 'n_nonfinite_steps', 0)}")
    moved = np.abs(params[:, reset_idx] - theta0).max()
    check("reset thetas actually moved (training-level version of the box test)",
          moved > 1e-6, f"max |dtheta| over 10 steps = {moved:.3e}")
    e_first, e_last = energies[:, 0], np.asarray(value)
    check("energy decreased on average", e_last.mean() < e_first.mean(),
          f"{e_first.mean():.4f} -> {e_last.mean():.4f} (exact ground state is -13)")


def main():
    print("=" * 78)
    print("Trajectory-reset equivalence tests (reference: purified ancilla path)")
    print("=" * 78)
    purified, traj, params, g_ref, enum = test_single_reset()
    test_without_magic_box(traj, params, g_ref, enum)
    test_composition()
    test_chain_lengths()
    test_canonical_form()
    test_sampler_monte_carlo()
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
