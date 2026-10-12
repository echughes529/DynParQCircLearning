"""Correctness tests for the SVD-reduction gate-path feature flags.

Covers cartan_mode ("separate" vs "fused") on apply_cartanblock and
toffoli_mode ("decomposed" vs "direct") on the probabilistic-reset
construction. Every state/gradient comparison is run both at bond_dim=None
(untruncated, QR path for 2-qubit gates) and bond_dim=64 (the live
production setting in plot_training_curve_tc.py, SVD path for 2-qubit
gates) -- the two gate-application paths can have different autodiff
behaviour (see make_split_conf's docstring), so equivalence at only one of
them would not be enough to trust flipping the defaults.
"""

import sys
import types

import jax
import jax.numpy as jnp
import pytest
import tensorcircuit as tc

# TensorFlow is only needed by unrelated legacy constructors in
# generate_ansatz (see grep for `tf.` there); stub it unconditionally so this
# focused test doesn't depend on TF/Keras/sklearn being importable at all --
# in some environments the installed TF pulls in a sklearn build compiled
# against a different NumPy ABI than the installed NumPy, which segfaults
# pytest's own collection/traceback machinery rather than raising cleanly.
sys.modules["tensorflow"] = types.ModuleType("tensorflow")

from src.utilities.generate_ansatz import (
    apply_cartanblock,
    _decomposed_ccx,
    construct_dyn_circuit_toriccodelattice_prob_resets,
    make_split_conf,
)
from src.utilities.ansatz_classes import ToricCodeAnsatz

# Matches the live experiment config (src/examples/plot_training_curve_tc.py).
PRODUCTION_BOND_DIM = 64
BOND_DIMS = [None, PRODUCTION_BOND_DIM]


def _normalized_overlap(psi_a, psi_b):
    psi_a = psi_a / jnp.linalg.norm(psi_a)
    psi_b = psi_b / jnp.linalg.norm(psi_b)
    return jnp.abs(jnp.vdot(psi_a, psi_b)) ** 2


# ---------------------------------------------------------------------------
# Fused vs. separate Cartan block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bond_dim", BOND_DIMS)
@pytest.mark.parametrize("qubit0,qubit1,nq", [
    (0, 1, 2),  # adjacent
    (0, 3, 4),  # nonadjacent
    (3, 0, 4),  # nonadjacent, reversed order
])
def test_fused_cartan_matches_separate_state(qubit0, qubit1, nq, bond_dim):
    params = jnp.linspace(0.1, 0.9, 9)
    split_conf = make_split_conf(bond_dim)

    qc_sep = tc.MPSCircuit(nq, split=split_conf)
    apply_cartanblock(qc_sep, qubit0, qubit1, params, 0, cartan_mode="separate")

    qc_fused = tc.MPSCircuit(nq, split=split_conf)
    apply_cartanblock(qc_fused, qubit0, qubit1, params, 0, cartan_mode="fused")

    overlap = _normalized_overlap(qc_sep.wavefunction(), qc_fused.wavefunction())
    assert overlap == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("bond_dim", BOND_DIMS)
def test_fused_cartan_gradients_agree(bond_dim):
    split_conf = make_split_conf(bond_dim)

    def energy(params, cartan_mode):
        qc = tc.MPSCircuit(2, split=split_conf)
        apply_cartanblock(qc, 0, 1, params, 0, cartan_mode=cartan_mode)
        return jnp.real(qc.expectation((tc.gates.z(), [0])))

    params = jnp.linspace(0.1, 0.9, 9)
    grad_sep = jax.grad(lambda p: energy(p, "separate"))(params)
    grad_fused = jax.grad(lambda p: energy(p, "fused"))(params)
    assert jnp.allclose(grad_sep, grad_fused, atol=1e-8)


def test_cartan_block_consumes_nine_params_either_mode():
    params = jnp.zeros(20)
    for mode in ("separate", "fused"):
        qc = tc.MPSCircuit(2, split=make_split_conf())
        _, paramindex = apply_cartanblock(qc, 0, 1, params, 5, cartan_mode=mode)
        assert paramindex == 14


def test_cartan_mode_rejects_unknown_value():
    params = jnp.zeros(9)
    qc = tc.MPSCircuit(2, split=make_split_conf())
    with pytest.raises(ValueError):
        apply_cartanblock(qc, 0, 1, params, 0, cartan_mode="bogus")


# ---------------------------------------------------------------------------
# Direct vs. decomposed Toffoli
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bond_dim", BOND_DIMS)
@pytest.mark.parametrize("c0_val,c1_val", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_direct_toffoli_matches_decomposed_on_basis_states(c0_val, c1_val, bond_dim):
    split_conf = make_split_conf(bond_dim)

    qc_dec = tc.MPSCircuit(3, split=split_conf)
    if c0_val:
        qc_dec.x(0)
    if c1_val:
        qc_dec.x(1)
    _decomposed_ccx(qc_dec, 0, 1, 2)

    qc_dir = tc.MPSCircuit(3, split=split_conf)
    if c0_val:
        qc_dir.x(0)
    if c1_val:
        qc_dir.x(1)
    qc_dir.toffoli(0, 1, 2)

    overlap = _normalized_overlap(qc_dec.wavefunction(), qc_dir.wavefunction())
    assert overlap == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("bond_dim", BOND_DIMS)
def test_direct_toffoli_matches_decomposed_superposed_control(bond_dim):
    split_conf = make_split_conf(bond_dim)

    def build(toffoli_mode, theta):
        qc = tc.MPSCircuit(3, split=split_conf)
        qc.ry(0, theta=theta)  # superposed, differentiable control
        qc.x(1)
        if toffoli_mode == "direct":
            qc.toffoli(0, 1, 2)
        else:
            _decomposed_ccx(qc, 0, 1, 2)
        return qc

    theta = 0.37
    qc_dec = build("decomposed", theta)
    qc_dir = build("direct", theta)
    overlap = _normalized_overlap(qc_dec.wavefunction(), qc_dir.wavefunction())
    assert overlap == pytest.approx(1.0, abs=1e-8)

    def energy(theta, toffoli_mode):
        qc = build(toffoli_mode, theta)
        return jnp.real(qc.expectation((tc.gates.z(), [2])))

    grad_dec = jax.grad(lambda t: energy(t, "decomposed"))(theta)
    grad_dir = jax.grad(lambda t: energy(t, "direct"))(theta)
    # Loose-ish tolerance: decomposed (many chained 2-qubit SVDs) and direct
    # (QR-canonicalize + one apply_MPO SVD) accumulate rounding differently
    # through complex128 arithmetic even though both are mathematically exact
    # here (no truncation at this size) -- a few 1e-7-relative ULPs of drift
    # is expected, not a correctness bug.
    assert float(grad_dec) == pytest.approx(float(grad_dir), rel=1e-6, abs=1e-7)


def test_toffoli_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        construct_dyn_circuit_toriccodelattice_prob_resets(
            jnp.zeros(1), nlayers=1, nparams=1, n_mps_qubits=3,
            mps_claws=[], mps_reset_qubits=[], active_reset_layers=[],
            sys_mps_positions=[0], toffoli_mode="bogus",
        )


# ---------------------------------------------------------------------------
# ToricCodeAnsatz-level plumbing
# ---------------------------------------------------------------------------

def _tiny_ansatz(**overrides):
    kwargs = dict(Lx=2, Ly=2, nlayers=1, trials=1, use_optimal_ordering=False, seed=0)
    kwargs.update(overrides)
    return ToricCodeAnsatz(**kwargs)


def test_ansatz_mode_flags_participate_in_hash():
    base = _tiny_ansatz()
    fused = _tiny_ansatz(cartan_mode="fused")
    direct = _tiny_ansatz(toffoli_mode="direct")
    assert hash(base) != hash(fused)
    assert hash(base) != hash(direct)


def test_ansatz_rejects_unknown_mode_strings():
    with pytest.raises(ValueError):
        _tiny_ansatz(cartan_mode="bogus")
    with pytest.raises(ValueError):
        _tiny_ansatz(toffoli_mode="bogus")
