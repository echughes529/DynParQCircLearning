"""Tests for the Experiment 1 tooling (src/experiments/, reset-qubit rule).

Run:
    source /home/s1931382/dpqc_venv/bin/activate
    python -m pytest tests/test_exp1_tools.py -q
"""

import numpy as np
import pytest

from src.experiments.exp1_common import steps_to_converge
from src.utilities.generate_toric_code_hamiltonian import ToricCode, spread_rung_reset_qubits


# ----------------------------------------------------------------------------
# Reset-qubit rule
# ----------------------------------------------------------------------------
def test_spread_rule_examples():
    assert spread_rung_reset_qubits(2, 2, 4) == [0, 1, 2, 3]          # all qubits
    assert spread_rung_reset_qubits(3, 2, 4) == [0, 4, 5, 6]          # 3 rungs + qubit 0
    assert spread_rung_reset_qubits(4, 2, 4) == [6, 7, 8, 9]          # all 4 rungs
    assert spread_rung_reset_qubits(5, 2, 4) == [8, 9, 11, 12]        # rungs x=0,1,3,4


@pytest.mark.parametrize("Lx", range(2, 21))
def test_spread_rule_valid(Lx):
    lat = ToricCode(Lx, 2)
    q = spread_rung_reset_qubits(Lx, 2, 4)
    assert len(q) == 4 and len(set(q)) == 4
    assert all(0 <= x < lat.num_qubits for x in q)
    if Lx >= 4:
        rungs = {lat.qubit_index(x, 0, 1) for x in range(Lx)}
        assert set(q) <= rungs
        # both ends of the ladder are always included
        assert lat.qubit_index(0, 0, 1) in q and lat.qubit_index(Lx - 1, 0, 1) in q


def test_spread_rule_rejects_non_ladder():
    with pytest.raises(ValueError):
        spread_rung_reset_qubits(3, 3, 4)


# ----------------------------------------------------------------------------
# Ladder ordering
# ----------------------------------------------------------------------------
def _ladder_ansatz(Lx, purified=False, **kw):
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    return ToricCodeAnsatz(
        Lx=Lx, Ly=2, nlayers=2, trials=1, maxiter=1, bond_dim=64, seed=3,
        use_prob_resets_ansatz=True, unitary=True, sparse=False, reset_layers=[1],
        which_qubits_for_prob_reset=spread_rung_reset_qubits(Lx, 2, 4),
        **{"use_trajectory_resets": not purified, **kw})


@pytest.mark.parametrize("Lx", [2, 3, 5, 12, 30])
def test_ladder_order_bounded_claws(Lx):
    for purified, bound in ((False, 2), (True, 6)):
        a = _ladder_ansatz(Lx, purified=purified, use_ladder_ordering=True)
        s = a.sys_to_mps
        assert sorted(s) == list(range(a.lattice.num_qubits))
        assert len(set(s.values())) == a.lattice.num_qubits
        # the claws the circuit applies, independent of Lx
        assert max(abs(s[x] - s[y]) for x, y in a.lattice.all_claws()) <= bound
        assert a.n_mps_qubits == a.lattice.num_qubits + (8 if purified else 0)


def test_ladder_order_rejects_non_ladder():
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    with pytest.raises(ValueError):
        ToricCodeAnsatz(Lx=3, Ly=3, nlayers=2, trials=1, maxiter=1, bond_dim=16,
                        use_prob_resets_ansatz=True, reset_layers=[1],
                        use_ladder_ordering=True)


def test_ladder_order_same_energy():
    for purified in (False, True):
        kw = {} if purified else dict(use_trajectory_resets=False, use_enumerated_resets=True)
        a = _ladder_ansatz(3, purified=purified, use_ladder_ordering=False, **kw)
        b = _ladder_ansatz(3, purified=purified, use_ladder_ordering=True, **kw)
        p = a.initparams[0]
        assert abs(float(a.energy_from_params(p)) - float(b.energy_from_params(p))) < 1e-10


# ----------------------------------------------------------------------------
# Convergence rule
# ----------------------------------------------------------------------------
def test_steps_to_converge():
    steps = np.arange(6) * 10
    assert steps_to_converge([1, 1, 1e-4, 1e-4, 1e-4, 1e-4], steps) == 20
    assert steps_to_converge([1, 1e-4, 1, 1e-4, 1e-4, 1e-4], steps) == 30   # dip doesn't count
    assert steps_to_converge([1e-4] * 6, steps) == 0
    assert steps_to_converge([1, 1, 1, 1, 1e-4, 1], steps) is None


# ----------------------------------------------------------------------------
# Enumerated twin: exact energy and variance
# ----------------------------------------------------------------------------
BOND_DIM = 64   # exact at 2x2 for every method (purified chain is 12 sites)


def _cfg(**kw):
    cfg = dict(Lx=2, Ly=2, nlayers=2, h=0.0, bond_dim=BOND_DIM, use_optimal_ordering=True,
               reset_layers=[1], reset_qubits=spread_rung_reset_qubits(2, 2, 4),
               shared_reset_theta=True, total_resets=4, seed=1234)
    cfg.update(kw)
    return cfg


def _ansatz(cfg, **flags):
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    return ToricCodeAnsatz(
        Lx=cfg["Lx"], Ly=cfg["Ly"], nlayers=cfg["nlayers"], h=cfg["h"], trials=1, maxiter=1,
        bond_dim=cfg["bond_dim"], use_optimal_ordering=cfg["use_optimal_ordering"],
        use_prob_resets_ansatz=True, unitary=True, sparse=False, use_mps=True,
        reset_layers=cfg["reset_layers"], which_qubits_for_prob_reset=cfg["reset_qubits"],
        shared_reset_theta=cfg["shared_reset_theta"], seed=cfg["seed"], **flags)


@pytest.fixture(scope="module")
def twin_and_params():
    from src.experiments.exp1_analyse_run import build_enumerated_twin
    cfg = _cfg()
    twin = build_enumerated_twin(cfg)
    assert twin.total_resets == 4
    params = np.array(twin.initparams[0])
    # p_reset = 0.5, so all 81 trajectories carry weight and the variance is large
    params[twin.reset_param_indices[0]] = np.pi / 2
    return cfg, twin, params


def test_twin_matches_purified(twin_and_params):
    from src.experiments.exp1_analyse_run import exact_energy_and_variance
    cfg, twin, params = twin_and_params
    pur = _ansatz(cfg, use_trajectory_resets=False)
    assert pur.nparams == twin.nparams
    E_twin, _ = exact_energy_and_variance(twin, params[None, :], batch=1)
    E_pur = float(pur.energy_from_params(params))
    assert abs(E_twin[0] - E_pur) < 1e-10, (E_twin[0], E_pur)


def test_twin_matches_sampled_trajectories(twin_and_params):
    import jax
    import jax.numpy as jnp
    from src.experiments.exp1_analyse_run import exact_energy_and_variance
    from src.utilities.generate_ansatz import _normalize_mps_if_requested

    cfg, twin, params = twin_and_params
    traj = _ansatz(cfg, use_trajectory_resets=True, n_trajectories=1)
    assert traj.nparams == twin.nparams
    E_exact, V_exact = exact_energy_and_variance(twin, params[None, :], batch=1)

    def one(st):
        qc, _ = traj._circuit_traj(jnp.asarray(params), st)
        _normalize_mps_if_requested(qc, traj.normalize_state)
        return traj._energy_of_circuit(qc)

    N = 2000
    status = jax.random.uniform(jax.random.PRNGKey(0), (N, traj.total_resets))
    samples = np.asarray(jax.jit(jax.vmap(one))(status))
    mean, var = samples.mean(), samples.var(ddof=1)
    se_mean = np.sqrt(V_exact[0] / N)
    assert V_exact[0] > 1e-3, "test needs a non-trivial trajectory spread"
    assert abs(mean - E_exact[0]) < 5 * se_mean, (mean, E_exact[0], se_mean)
    # relative SE of a sample variance is ~sqrt(2/(N-1)) for near-Gaussian data;
    # allow a generous margin for the discrete, skewed distribution here
    assert abs(var / V_exact[0] - 1) < 0.15, (var, V_exact[0])
