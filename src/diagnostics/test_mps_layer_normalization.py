"""Regression tests for the optional per-layer MPS normalization."""

import importlib.util
import sys
import types

import jax.numpy as jnp
import tensorcircuit as tc

# TensorFlow is only needed by unrelated legacy constructors in
# generate_ansatz; keep this focused test runnable with the MPS dependencies.
if importlib.util.find_spec("tensorflow") is None:
    sys.modules["tensorflow"] = types.ModuleType("tensorflow")

from src.utilities.generate_ansatz import construct_unitary_circuit_brickwork


def test_brickwork_normalization_toggle_covers_every_layer(monkeypatch):
    scaffold = {
        "nq": 2,
        "n_ancillas": 0,
        "nlayers": 2,
        "howoften": 0,
        "claws": [(0, 1)],
        "ancillas": [],
        "nparams": 24,
        "backend": "tc",
    }
    params = jnp.linspace(0.01, 0.24, 24)

    normalize_calls = []
    original_normalize = tc.MPSCircuit.normalize

    def counted_normalize(circuit):
        normalize_calls.append(circuit.get_center_position())
        return original_normalize(circuit)

    monkeypatch.setattr(tc.MPSCircuit, "normalize", counted_normalize)

    construct_unitary_circuit_brickwork(
        params, scaffold, normalize_state=False
    )
    assert normalize_calls == []

    construct_unitary_circuit_brickwork(
        params, scaffold, normalize_state=True
    )
    assert len(normalize_calls) == 3  # two repeated layers plus the final layer
