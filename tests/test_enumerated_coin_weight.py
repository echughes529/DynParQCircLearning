"""Enumerated resets with the coin weight p^(a) outside the amplitudes.

VariationalAnsatz.trajectory_table builds each branch with the parameter-free
Kraus operators (coin_in_amplitude=False) and multiplies by the classical coin
weight p^(a). These tests check that this is the same table as putting the
coin factors into the amplitudes (the original construction), and that it
stays finite where the original underflowed.

Run:
    python -m pytest tests/test_enumerated_coin_weight.py -q
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.generate_toric_code_hamiltonian import spread_rung_reset_qubits

K = tc.set_backend("jax")


def _enum(reset_qubits=None, Lx=2, Ly=2, shared=True):
    return ToricCodeAnsatz(
        Lx=Lx, Ly=Ly, nlayers=2, h=0.3, trials=1, maxiter=1, bond_dim=64,
        use_optimal_ordering=False, use_prob_resets_ansatz=True, unitary=True,
        sparse=False, use_mps=True, reset_layers=[1],
        which_qubits_for_prob_reset=reset_qubits, shared_reset_theta=shared,
        use_trajectory_resets=False, use_enumerated_resets=True, seed=7)


def _old_table(enum, params):
    """Per-branch (<phi|H|phi>, <phi|phi>) with the coin in the amplitudes."""
    nums, dens = [], []
    for branch in enum._reset_branches:
        qc = enum._circuit_branch(params, branch, coin_in_amplitude=True)
        centre = qc._mps.tensors[qc._mps.center_position]
        nums.append(float(enum._energy_of_circuit(qc)))
        dens.append(float(K.real(K.sum(centre * K.conj(centre)))))
    return np.array(nums), np.array(dens)


@pytest.mark.parametrize("shared", [True, False])
def test_table_matches_coin_in_amplitude(shared):
    enum = _enum(reset_qubits=spread_rung_reset_qubits(2, 2, 4), shared=shared)
    params = jnp.asarray(enum.initparams[0])
    nums, dens = enum.trajectory_table(params)
    old_nums, old_dens = _old_table(enum, params)
    np.testing.assert_allclose(np.asarray(dens), old_dens, atol=1e-12)
    np.testing.assert_allclose(np.asarray(nums), old_nums, atol=1e-12)
    assert abs(float(np.sum(dens)) - 1.0) < 1e-10      # trace preserving


def test_coin_weight_matches_formula():
    enum = _enum(reset_qubits=spread_rung_reset_qubits(2, 2, 4), shared=False)
    params = np.array(enum.initparams[0])
    th = np.asarray(enum._expand_params(jnp.asarray(params)))[enum._builder_reset_indices()]
    p = np.sin(th / 2) ** 2
    for branch in itertools.product(range(3), repeat=enum.total_resets):
        a = np.array(branch) != 0
        want = np.prod(np.where(a, p, 1 - p))
        got = float(enum._branch_coin_weight(jnp.asarray(params), jnp.asarray(branch)))
        assert abs(got - want) < 1e-14


@pytest.mark.parametrize("theta", [1e-10, 1e-40, 1e-160, 0.0])
def test_finite_at_tiny_reset_angle(theta):
    """The regime that NaN'd the Experiment 1 GPU runs, with the gradient."""
    enum = _enum(reset_qubits=spread_rung_reset_qubits(2, 2, 4))
    k = enum.reset_param_indices[0]
    params = jnp.asarray(enum.initparams[0]).at[k].set(theta)
    E, g = jax.value_and_grad(enum.energy_from_params)(params)
    assert np.isfinite(float(E)) and bool(jnp.all(jnp.isfinite(g)))
    # resets effectively off: the energy is the no-reset branch's energy
    nums, dens = enum.trajectory_table(params)
    no_reset = int(np.flatnonzero((np.asarray(enum._reset_branches) == 0).all(axis=1))[0])
    assert abs(float(E) - float(nums[no_reset] / dens[no_reset])) < 1e-10


def test_gradient_matches_finite_difference():
    """Plain autodiff through p^(a) gives the right reset-theta gradient."""
    enum = _enum(reset_qubits=spread_rung_reset_qubits(2, 2, 4))
    k = enum.reset_param_indices[0]
    params = jnp.asarray(enum.initparams[0]).at[k].set(0.9)
    g = jax.grad(enum.energy_from_params)(params)
    eps = 1e-5
    fd = (float(enum.energy_from_params(params.at[k].add(eps)))
          - float(enum.energy_from_params(params.at[k].add(-eps)))) / (2 * eps)
    assert abs(float(g[k]) - fd) < 1e-7 * max(1.0, abs(fd))
