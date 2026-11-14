"""Tests for the Experiment 1.1 tooling (Lx x 3 lattices).

See experiments/experiment_1.1_scaling_system_size_nx3/PLAN.md.

Run:
    source /home/s1931382/dpqc_venv/bin/activate
    python -m pytest tests/test_exp1_1_tools.py -q
"""

import numpy as np
import pytest

from src.utilities.generate_toric_code_hamiltonian import ToricCode, spread_rung_reset_qubits


# ----------------------------------------------------------------------------
# Reset-qubit rule: alternating rows of rungs
# ----------------------------------------------------------------------------
def test_spread_rule_examples_ly3():
    # PLAN.md section 1, option A
    assert spread_rung_reset_qubits(3, 3, 4) == [6, 7, 10, 11]
    assert spread_rung_reset_qubits(4, 3, 4) == [9, 11, 14, 16]
    assert spread_rung_reset_qubits(5, 3, 4) == [12, 15, 18, 21]
    assert spread_rung_reset_qubits(6, 3, 4) == [15, 18, 23, 26]
    assert spread_rung_reset_qubits(10, 3, 4) == [27, 33, 40, 46]


@pytest.mark.parametrize("Lx", range(3, 49))
def test_spread_rule_valid_ly3(Lx):
    lat = ToricCode(Lx, 3)
    q = spread_rung_reset_qubits(Lx, 3, 4)
    assert len(q) == 4 and len(set(q)) == 4
    xs = np.round(np.linspace(0, Lx - 1, 4)).astype(int)
    # reset k is the rung at (x_k, row k mod 2): rows go 0, 1, 0, 1
    want = {lat.qubit_index(int(x), k % 2, 1) for k, x in enumerate(xs)}
    assert set(q) == want
    rungs_row0 = {lat.qubit_index(x, 0, 1) for x in range(Lx)}
    rungs_row1 = {lat.qubit_index(x, 1, 1) for x in range(Lx)}
    assert len(set(q) & rungs_row0) == 2 and len(set(q) & rungs_row1) == 2
    # both ends of the lattice are covered, as on the ladders
    assert lat.qubit_index(0, 0, 1) in q and lat.qubit_index(Lx - 1, 1, 1) in q


def test_spread_rule_rejects_2x3():
    # x = 0, 0, 1, 1 with rows 0, 1, 0, 1 is fine in principle, but 2x3 is not
    # in the experiment; make sure an impossible request raises, not duplicates.
    with pytest.raises(ValueError):
        spread_rung_reset_qubits(2, 3, 6)


# ----------------------------------------------------------------------------
# Column ordering on Lx x 3 (verticals first)
# ----------------------------------------------------------------------------
def _ansatz(Lx, purified=False, **kw):
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    flags = dict(use_trajectory_resets=not purified)
    flags.update(kw)
    return ToricCodeAnsatz(
        Lx=Lx, Ly=3, nlayers=2, trials=1, maxiter=1, bond_dim=64, seed=3,
        use_prob_resets_ansatz=True, unitary=True, sparse=False, reset_layers=[1],
        which_qubits_for_prob_reset=spread_rung_reset_qubits(Lx, 3, 4), **flags)


def test_column_order_pattern_ly3():
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    lat = ToricCode(4, 3)
    stub = ToricCodeAnsatz.__new__(ToricCodeAnsatz)
    stub.Lx, stub.Ly, stub.lattice = 4, 3, lat
    order = ToricCodeAnsatz._ladder_order(stub)
    # column x: v(x,0), v(x,1), h(x,0), h(x,1), h(x,2); last column verticals only
    want = []
    for x in range(4):
        want += [lat.qubit_index(x, 0, 1), lat.qubit_index(x, 1, 1)]
        if x < 3:
            want += [lat.qubit_index(x, y, 0) for y in range(3)]
    assert order == want


@pytest.mark.parametrize("Lx", [3, 4, 5, 12, 30, 48])
def test_column_order_bounded_claws_ly3(Lx):
    for purified, bound in ((False, 3), (True, 7 if Lx == 3 else 5)):
        a = _ansatz(Lx, purified=purified, use_ladder_ordering=True)
        s = a.sys_to_mps
        nq = a.lattice.num_qubits
        assert sorted(s) == list(range(nq))
        assert len(set(s.values())) == nq
        assert max(abs(s[x] - s[y]) for x, y in a.lattice.all_claws()) <= bound
        assert a.n_mps_qubits == nq + (8 if purified else 0)


def test_column_order_same_energy_as_search_3x3():
    """Reordering the chain is exact when nothing is truncated."""
    # Enumerated: 12-site chain, bond dimension 64 is the full Hilbert space.
    kw = dict(use_trajectory_resets=False, use_enumerated_resets=True)
    a = _ansatz(3, use_ladder_ordering=False, **kw)
    b = _ansatz(3, use_ladder_ordering=True, **kw)
    p = a.initparams[0]
    assert abs(float(a.energy_from_params(p)) - float(b.energy_from_params(p))) < 1e-10


def test_column_order_same_energy_as_search_3x3_purified():
    # Purified: 20-site chain (exact cap 1024). At bond dimension 256 the
    # random-init need at 3x3 is met with room to spare (PLAN.md section 3:
    # 192 already suffices at 4x3-20x3), so neither ordering truncates
    # measurably and the energies must agree; two differently-truncated states
    # would not agree to 1e-10. normalize_state is on in both and is a no-op
    # without truncation.
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    from src.utilities.generate_ansatz import make_split_conf
    a = _ansatz(3, purified=True, use_ladder_ordering=False)
    b = _ansatz(3, purified=True, use_ladder_ordering=True)
    for x in (a, b):
        x.bond_dim, x.split_conf = 256, make_split_conf(256)
    p = a.initparams[0]
    assert abs(float(a.energy_from_params(p)) - float(b.energy_from_params(p))) < 1e-10


# ----------------------------------------------------------------------------
# Calibration runs reuse trials 0..k-1 of the 10-trial runs
# ----------------------------------------------------------------------------
def test_initparams_prefix_independent_of_trial_count():
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    common = dict(Lx=4, Ly=3, nlayers=2, maxiter=1, bond_dim=16, seed=1234,
                  use_prob_resets_ansatz=True, unitary=True, sparse=False, reset_layers=[1],
                  which_qubits_for_prob_reset=spread_rung_reset_qubits(4, 3, 4),
                  use_ladder_ordering=True, use_trajectory_resets=True)
    full = np.asarray(ToricCodeAnsatz(trials=10, **common).initparams)
    for k in (1, 2, 3, 5):
        part = np.asarray(ToricCodeAnsatz(trials=k, **common).initparams)
        np.testing.assert_array_equal(part, full[:k])
