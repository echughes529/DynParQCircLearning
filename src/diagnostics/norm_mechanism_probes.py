"""Experiment 5: small, self-contained probes of the failure mechanism.

These do not touch the ansatz at all -- they isolate the three numerical
ingredients the full-circuit experiments implicate, so the mechanism can be
stated without hand-waving:

  P1  tensorcircuit's SVD backward pass is scale dependent. Its Lorentzian
      broadening eps = 1e-15 is an ABSOLUTE floor on (s_i^2 - s_j^2), so
      rescaling a state by lambda moves the entire F matrix by 1/lambda^2 and
      pushes near-degenerate pairs into the damped (silently wrong) regime.
  P2  What the true and broadened reciprocals actually do to a real gradient,
      as a function of the overall scale.
  P3  Adam's response to one non-finite gradient: the optimizer state is
      poisoned permanently, which is why a single bad step ends the run
      rather than causing a transient spike.

Usage:
    python -m src.diagnostics.norm_mechanism_probes
"""
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import optax

from src.diagnostics.normalization_study import (
    SAFE_RECIPROCAL_EPS,
    SAFE_RECIPROCAL_KNEE,
)


def safe_reciprocal(x):
    return x / (x * x + SAFE_RECIPROCAL_EPS)


def p1_scale_dependence():
    print("P1: the SVD backward pass is scale dependent")
    print(f"    _safe_reciprocal(x) = x/(x^2+{SAFE_RECIPROCAL_EPS:g});  "
          f"peaks at |x|={SAFE_RECIPROCAL_KNEE:.3e} with value {1/(2*SAFE_RECIPROCAL_KNEE):.3e}")
    print(f"    {'lambda':>10} {'|s^2 gap|':>12} {'true 1/gap':>14} {'F used':>14} {'rel err':>10}")
    gap_at_unit_norm = 1e-4          # a fairly ordinary squared-singular-value gap
    for lam_sq in [1.0, 0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        gap = gap_at_unit_norm * lam_sq
        true = 1.0 / gap
        used = float(safe_reciprocal(gap))
        print(f"    {np.sqrt(lam_sq):>10.4f} {gap:>12.3e} {true:>14.3e} {used:>14.3e} "
              f"{abs(used-true)/true:>10.2%}")
    print()


def p2_gradient_through_svd():
    """Does rescaling the state change the gradient tensorcircuit computes?

    U and V are scale invariant, so for a loss that reads them out through a
    fixed external contraction, L(lambda*A) == L(A) exactly, and therefore

        dL/dA |_(lambda*A)  ==  (1/lambda) * dL/dA |_A

    holds exactly in real arithmetic, at every lambda. Any departure from that
    scaling law is the eps=1e-15 broadening corrupting the gradient -- and it
    is precisely what letting ||psi|| decay through the circuit does.
    """
    print("P2: does rescaling the state corrupt the gradient? (exact answer: it must not)")
    from tensorcircuit.backends.jax_ops import adaware_svd

    rng = np.random.default_rng(0)
    n = 8
    fixed = jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))

    def make_base(relative_degeneracy):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        u0, s0, vh0 = np.linalg.svd(a)
        s0 = s0 / np.linalg.norm(s0)
        s0[3] = s0[2] * (1 - relative_degeneracy)  # a near-tied Schmidt pair
        return jnp.asarray((u0 * s0) @ vh0), s0

    def loss(A):
        u, s, vh = adaware_svd(A)
        return jnp.real(jnp.sum(jnp.conj(u) * fixed) + jnp.sum(jnp.conj(vh) * fixed))

    grad = jax.jit(jax.grad(loss, holomorphic=False))

    for relative_degeneracy in (1e-2, 1e-4, 1e-6):
        base, s0 = make_base(relative_degeneracy)
        gap_at_unit_norm = float(s0[2] ** 2 - s0[3] ** 2)
        g1 = np.asarray(grad(base))
        print(f"    nearest Schmidt pair split by {relative_degeneracy:.0e} "
              f"(|s_i^2-s_j^2| = {gap_at_unit_norm:.2e} at ||psi||=1)")
        print(f"      {'||psi||':>10} {'|s^2 gap|':>12} {'rel. error vs exact scaling':>30}")
        for lam in [1.0, 1e-1, 1e-2, 1e-3, 1e-4]:
            g = np.asarray(grad(lam * base))
            expected = g1 / lam
            rel = np.linalg.norm(g - expected) / np.linalg.norm(expected)
            print(f"      {lam:>10.0e} {gap_at_unit_norm*lam**2:>12.2e} {rel:>29.2%}")
    print("    A decaying norm therefore does not merely rescale the gradient -- past\n"
          "    the point where lambda^2 * (s_i^2 - s_j^2) reaches ~1e-8 it changes it.")
    print()


def p3_adam_poisoning():
    """One non-finite gradient permanently destroys Adam's state."""
    print("P3: Adam's response to a single bad gradient")
    for bad in [np.inf, np.nan, 1e300, 1e30]:
        params = jnp.ones(4)
        opt = optax.adam(1e-2)
        state = opt.init(params)
        g_ok = jnp.full(4, 0.1)
        for step in range(6):
            g = g_ok.at[0].set(bad) if step == 2 else g_ok
            updates, state = opt.update(g, state)
            params = optax.apply_updates(params, updates)
        recovered = bool(np.isfinite(np.asarray(params)).all())
        print(f"    injected grad component {bad:>10.3g} at step 2 -> params after 3 "
              f"more clean steps: {np.asarray(params)}  finite={recovered}")
    print("    An Inf or NaN gradient is unrecoverable: Adam's first/second moment\n"
          "    buffers become non-finite and every later update is NaN, even if the\n"
          "    circuit itself would have produced perfectly good gradients again.\n"
          "    A merely huge (1e30/1e300) gradient is survivable -- Adam's\n"
          "    normalisation absorbs it -- so the failure needs an actual overflow,\n"
          "    not just a large number.")
    print()


if __name__ == "__main__":
    p1_scale_dependence()
    p2_gradient_through_svd()
    p3_adam_poisoning()
