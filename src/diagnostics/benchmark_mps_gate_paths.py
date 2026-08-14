"""Benchmark real SVD-invocation counts and timing across the SVD-reduction
gate-path feature-flag combinations (cartan_mode x toffoli_mode).

Run: python -m src.diagnostics.benchmark_mps_gate_paths

Reports, per mode combination, at a fixed seed and the live production
bond_dim (see plot_training_curve_tc.py):
  - first forward / first value_and_grad wall-clock time
  - warm forward-pass time
  - warm value_and_grad time
  - actual TensorNetwork SVD invocation count (svd_call_counting.tap_backends)
  - qc.get_norm()
  - max bond dimension
  - energy and gradient norm
  - NaN/Inf presence

Plus a small-lattice, untruncated overlap check of every mode combination's
forward state against the legacy (separate/decomposed) state.

Read the printed SVD-count numbers as a measurement, not a confirmation of
any specific pre-derived count: apply_MPO's n-qubit MPO path (used by the
direct Toffoli) costs (k-1) SVD calls per k-site gate span, not "1 SVD per
gate", so the true totals depend on the actual qubit ordering and are not
known ahead of running this.
"""

import sys
import time
import types

import jax
import jax.numpy as jnp
import numpy as np

# TensorFlow is only needed by unrelated legacy constructors in
# generate_ansatz (see grep for `tf.` there); stub it unconditionally so this
# benchmark doesn't depend on TF/Keras/sklearn being importable at all -- in
# some environments the installed TF pulls in a sklearn build compiled
# against a different NumPy ABI than the installed NumPy, which crashes hard
# rather than raising cleanly.
sys.modules["tensorflow"] = types.ModuleType("tensorflow")

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.diagnostics.svd_call_counting import tap_backends

# Matches the live experiment config (src/examples/plot_training_curve_tc.py)
# and the NaN repro seed used in src/examples/test.py's LIVE_SEED.
LX, LY = 3, 3
NLAYERS = 2
HOWOFTEN_TORESET = 7
RESET_LAYERS = [1]
BOND_DIM = 64
SEED = 22405

MODE_COMBOS = [
    ("separate", "decomposed"),
    ("fused", "decomposed"),
    ("separate", "direct"),
    ("fused", "direct"),
]


def _build_ansatz(cartan_mode, toffoli_mode, **overrides):
    kwargs = dict(
        Lx=LX, Ly=LY, nlayers=NLAYERS, howoften_toreset=HOWOFTEN_TORESET,
        reset_layers=RESET_LAYERS, bond_dim=BOND_DIM, use_optimal_ordering=False,
        trials=1, seed=SEED, cartan_mode=cartan_mode, toffoli_mode=toffoli_mode,
    )
    kwargs.update(overrides)
    return ToricCodeAnsatz(**kwargs)


def _time_call(fn, *args):
    start = time.perf_counter()
    result = jax.block_until_ready(fn(*args))
    return result, time.perf_counter() - start


def benchmark_one(cartan_mode, toffoli_mode):
    print(f"\n{'=' * 70}\ncartan_mode={cartan_mode!r} toffoli_mode={toffoli_mode!r}\n{'=' * 70}")

    ansatz = _build_ansatz(cartan_mode, toffoli_mode)
    params = ansatz.initparams[0]

    def forward(p):
        return ansatz._circuit(p).wavefunction()

    def energy_and_grad(p):
        return jax.value_and_grad(ansatz.energy_from_params)(p)

    _, called_tn, restore = tap_backends()
    try:
        _, first_forward_time = _time_call(forward, params)
        _, first_vg_time = _time_call(energy_and_grad, params)
        first_svd_calls = dict(called_tn)
    finally:
        restore()

    _, warm_forward_time = _time_call(forward, params)
    (energy, grad), warm_vg_time = _time_call(energy_and_grad, params)

    qc = ansatz._circuit(params)
    norm = float(jax.block_until_ready(qc.get_norm()))
    bond_dims = np.asarray(qc.get_bond_dimensions())
    grad_norm = float(jnp.linalg.norm(grad))
    has_nan = bool(jnp.isnan(energy)) or bool(jnp.any(jnp.isnan(grad)))
    has_inf = bool(jnp.isinf(energy)) or bool(jnp.any(jnp.isinf(grad)))

    print(f"  first forward time: {first_forward_time:.3f}s")
    print(f"  first value_and_grad time: {first_vg_time:.3f}s")
    print(f"  warm forward time: {warm_forward_time:.4f}s")
    print(f"  warm value_and_grad time: {warm_vg_time:.4f}s")
    print(f"  real SVD calls (first forward+grad, tensornetwork JaxBackend): {first_svd_calls}")
    print(f"  qc.get_norm(): {norm}")
    print(f"  max bond dimension: {int(np.max(bond_dims))}")
    print(f"  energy: {float(energy)}  grad_norm: {grad_norm}")
    print(f"  NaN present: {has_nan}  Inf present: {has_inf}")

    return {
        "cartan_mode": cartan_mode,
        "toffoli_mode": toffoli_mode,
        "svd_calls": first_svd_calls["svd"],
        "warm_forward_time": warm_forward_time,
        "warm_vg_time": warm_vg_time,
        "norm": norm,
        "max_bond_dim": int(np.max(bond_dims)),
        "energy": float(energy),
        "grad_norm": grad_norm,
        "has_nan": has_nan,
        "has_inf": has_inf,
    }


def small_circuit_overlap_check():
    """Sanity check every mode combo's forward state against the legacy
    (separate/decomposed) state on a small, untruncated lattice, where
    running all four configurations is cheap."""
    print(f"\n{'=' * 70}\nsmall-circuit overlap check (Lx=Ly=2, bond_dim=None)\n{'=' * 70}")
    legacy = _build_ansatz("separate", "decomposed", Lx=2, Ly=2, nlayers=1,
                            reset_layers=[0], bond_dim=None, seed=0)
    legacy_psi = legacy._circuit(legacy.initparams[0]).wavefunction()
    legacy_psi = legacy_psi / jnp.linalg.norm(legacy_psi)

    for cartan_mode, toffoli_mode in MODE_COMBOS:
        ansatz = _build_ansatz(cartan_mode, toffoli_mode, Lx=2, Ly=2, nlayers=1,
                                reset_layers=[0], bond_dim=None, seed=0)
        psi = ansatz._circuit(ansatz.initparams[0]).wavefunction()
        psi = psi / jnp.linalg.norm(psi)
        overlap = float(jnp.abs(jnp.vdot(legacy_psi, psi)) ** 2)
        print(f"  cartan_mode={cartan_mode:<8} toffoli_mode={toffoli_mode:<10} overlap vs legacy: {overlap:.12f}")


def main():
    small_circuit_overlap_check()
    results = [benchmark_one(c, t) for c, t in MODE_COMBOS]

    print(f"\n{'=' * 70}\nSummary (Lx={LX}, Ly={LY}, bond_dim={BOND_DIM})\n{'=' * 70}")
    header = (f"{'cartan':<10}{'toffoli':<12}{'svd_calls':<12}"
              f"{'warm_fwd(s)':<14}{'warm_vg(s)':<14}{'energy':<14}{'nan':<6}")
    print(header)
    for r in results:
        print(f"{r['cartan_mode']:<10}{r['toffoli_mode']:<12}{r['svd_calls']:<12}"
              f"{r['warm_forward_time']:<14.4f}{r['warm_vg_time']:<14.4f}"
              f"{r['energy']:<14.6f}{str(r['has_nan']):<6}")


if __name__ == "__main__":
    main()
