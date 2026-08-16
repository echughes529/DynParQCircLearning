"""Shared machinery for the periodic-MPS-normalization study.

Three normalization regimes are compared throughout:

  "off"    - no normalization anywhere. The energy handed to the optimizer is
             <psi|H|psi> with an UNNORMALIZED psi, i.e. ||psi||^2 * E_true.
             This is what `normalize_state=False` does today.
  "end"    - normalize once, immediately before the Hamiltonian expectations.
             Mathematically the correct objective (a true Rayleigh quotient),
             but the MPS tensors carry a shrinking scale all the way through
             the circuit and through the whole backward pass.
  "layer"  - normalize at every layer boundary AND at the end. This is the
             current `normalize_state=True` default.

"end" and "layer" compute the *same mathematical function* of the parameters
(truncation by `max_singular_values` keeps the top-k singular values, which is
scale invariant, so rescaling the state part-way through cannot change which
values survive). Any difference between them is therefore purely a
floating-point conditioning effect -- which is exactly what makes the
three-way split diagnostic: "off" vs "end" isolates the *objective* bias,
"end" vs "layer" isolates the *numerical* benefit of periodic rescaling.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import tensorcircuit as tc

from src.find_gs import K
from src.utilities.ansatz_classes import ToricCodeAnsatz

NORM_MODES = ("off", "end", "layer")

# tensorcircuit/backends/jax_ops.py::_safe_reciprocal's Lorentzian broadening.
# It is an ABSOLUTE floor, applied to (s_i^2 - s_j^2) and to s itself, so its
# effect depends on the overall scale of the MPS tensors being split.
SAFE_RECIPROCAL_EPS = 1e-15
SAFE_RECIPROCAL_KNEE = np.sqrt(SAFE_RECIPROCAL_EPS)  # ~3.16e-8


@dataclass
class NormModeToricCodeAnsatz(ToricCodeAnsatz):
    """ToricCodeAnsatz with the normalization regime selected by ``norm_mode``.

    ``normalize_state`` is derived from ``norm_mode`` rather than set by hand,
    so the per-layer normalization inside the circuit builders and the final
    normalization before the energy can never drift out of sync.
    """

    norm_mode: str = "layer"

    def __post_init__(self):
        if self.norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {self.norm_mode!r}")
        self.normalize_state = self.norm_mode == "layer"
        super().__post_init__()

    def energy_from_params(self, params, seed=None):
        qc = self._circuit(params, seed)
        if self.norm_mode != "off":
            qc.normalize()
        energy = 0.0
        for ops, coeff in self._hamiltonian_terms():
            energy += coeff * qc.expectation(*ops)
        return K.real(energy)

    def norm_sq_from_params(self, params, seed=None):
        """<psi|psi> of the (unnormalized) circuit output, as a differentiable scalar."""
        qc = self._circuit(params, seed)
        return K.real(qc.get_norm() ** 2)

    def rayleigh_from_params(self, params, seed=None):
        """The true energy <psi|H|psi>/<psi|psi>, built without any in-circuit
        normalization, so it can be differentiated as one quotient."""
        qc = self._circuit(params, seed)
        norm_sq = qc.get_norm() ** 2
        energy = 0.0
        for ops, coeff in self._hamiltonian_terms():
            energy += coeff * qc.expectation(*ops)
        return K.real(energy / norm_sq)


def make_ansatz(norm_mode: str, **kwargs) -> NormModeToricCodeAnsatz:
    """Build a NormModeToricCodeAnsatz, silencing the noisy __post_init__ dump."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return NormModeToricCodeAnsatz(norm_mode=norm_mode, **kwargs)


# ---------------------------------------------------------------------------
# Forward-pass instrumentation
# ---------------------------------------------------------------------------


class SplitRecorder:
    """Records every truncating SVD performed while building an MPS circuit.

    The hook point is ``tensornetwork.backends.jax.jax_backend.JaxBackend.svd``,
    which tensorcircuit replaces at import time with its own ``_svd_jax``
    (backends/jax_backend.py:197). That single function is reached by *both*
    MPS split paths:

      * two-qubit gates -> ``apply_adjacent_double_gate`` ->
        ``FiniteMPS.apply_two_site_gate`` -> ``self.backend.svd``
      * three-qubit gates (``toffoli_mode="direct"``) -> ``apply_nqubit_gate``
        -> ``apply_MPO`` -> ``split_tensor`` -> ``backend.svd``

    Patching the gate methods instead would silently miss the MPO path, which
    is exactly where the direct-Toffoli truncation happens.

    Because each split acts on the orthogonality-center block of a canonical
    MPS, its singular values are the Schmidt coefficients of the *whole* state
    at that cut, so ``sum(s_kept^2) / sum(s_all^2)`` is exactly the fraction of
    ``||psi||^2`` surviving that split. Multiplying those ratios reconstructs
    the norm trajectory without ever touching the circuit object.
    """

    def __init__(self, capture_spectra: bool = True):
        self.capture_spectra = capture_spectra
        self.weight_kept: List[float] = []   # sum(s_kept^2) / sum(s_all^2)
        self.total_weight: List[float] = []  # sum(s_all^2), i.e. ||psi||^2 before
        self.spectra: List[np.ndarray] = []  # full pre-truncation spectrum
        self.n_kept: List[int] = []
        self.matrix_shapes: List[tuple] = []

    @contextmanager
    def patch(self):
        import tensornetwork
        from tensornetwork.backends.jax import jax_backend as _tn_jax

        original = _tn_jax.JaxBackend.svd
        recorder = self

        def wrapped(self_backend, tensor, *a, **kw):
            u, s, vh, s_rest = original(self_backend, tensor, *a, **kw)
            s_np = np.abs(np.asarray(s)).astype(float)
            rest_np = np.abs(np.asarray(s_rest)).astype(float)
            kept = float((s_np ** 2).sum())
            total = kept + float((rest_np ** 2).sum())
            recorder.weight_kept.append(kept / total if total > 0 else np.nan)
            recorder.total_weight.append(total)
            recorder.n_kept.append(int(s_np.size))
            recorder.matrix_shapes.append(tuple(np.asarray(tensor).shape))
            if recorder.capture_spectra:
                recorder.spectra.append(np.concatenate([s_np, rest_np]))
            return u, s, vh, s_rest

        _tn_jax.JaxBackend.svd = wrapped
        try:
            yield self
        finally:
            _tn_jax.JaxBackend.svd = original

    def norm_sq_trajectory(self) -> np.ndarray:
        """||psi||^2 after each split, reconstructed from the kept weights."""
        return np.cumprod(np.asarray(self.weight_kept, dtype=float))

    def summary(self) -> Dict[str, Any]:
        kept = np.asarray(self.weight_kept, dtype=float)
        lost = 1.0 - kept
        traj = self.norm_sq_trajectory()
        return {
            "n_splits": int(kept.size),
            "final_norm_sq": float(traj[-1]) if traj.size else 1.0,
            "final_norm": float(np.sqrt(traj[-1])) if traj.size else 1.0,
            "n_truncating_splits": int((lost > 1e-14).sum()),
            "total_weight_lost": float(np.nansum(lost)),
            "max_split_weight_lost": float(np.nanmax(lost)) if lost.size else 0.0,
            "max_bond_kept": int(np.max(self.n_kept)) if self.n_kept else 0,
        }


def spectrum_gap_stats(spectra: List[np.ndarray], scale: float = 1.0) -> Dict[str, Any]:
    """How close the SVD backward pass runs to its Lorentzian-broadening floor.

    ``jaxsvd_bwd`` forms ``F_ij = 1/(s_i^2 - s_j^2)`` via
    ``_safe_reciprocal(x) = x / (x^2 + 1e-15)``. That is an *absolute* floor:
    for ``|x| >> 3.16e-8`` it is the true reciprocal, for ``|x| << 3.16e-8`` it
    collapses towards zero. Rescaling the state by ``scale`` scales every
    ``s`` by ``scale`` and every ``s_i^2 - s_j^2`` by ``scale^2``, so a shrinking
    norm silently pushes more of ``F`` into the broadened (wrong) regime.
    """
    below_knee = 0
    total = 0
    min_gap = np.inf
    min_sv = np.inf
    max_damping = 0.0

    for s in spectra:
        s = np.asarray(s, dtype=float) * scale
        s = s[s > 0]
        if s.size < 2:
            continue
        sq = s ** 2
        gaps = np.abs(sq[:, None] - sq[None, :])
        iu = np.triu_indices(s.size, k=1)
        g = gaps[iu]
        total += g.size
        below_knee += int((g < SAFE_RECIPROCAL_KNEE).sum())
        if g.size:
            min_gap = min(min_gap, float(g.min()))
        min_sv = min(min_sv, float(s.min()))
        # ratio of the broadened reciprocal to the true one: x^2/(x^2+eps)
        damping = 1.0 - (g ** 2) / (g ** 2 + SAFE_RECIPROCAL_EPS)
        if damping.size:
            max_damping = max(max_damping, float(damping.max()))

    return {
        "n_gap_pairs": total,
        "frac_gaps_below_knee": (below_knee / total) if total else np.nan,
        "min_sq_gap": float(min_gap) if np.isfinite(min_gap) else np.nan,
        "min_singular_value": float(min_sv) if np.isfinite(min_sv) else np.nan,
        "max_backward_damping": max_damping,
    }
