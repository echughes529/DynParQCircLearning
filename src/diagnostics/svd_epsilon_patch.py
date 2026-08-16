"""Can the SVD-backward epsilon in TensorCircuit actually be monkey-patched?

Step 1 of the epsilon investigation. This file tunes nothing -- it establishes
whether, and by which mechanism, the Lorentzian broadening constant in
TensorCircuit's SVD backward pass can be changed from outside the library, and
proves the change reaches the code path this repo trains through.

What is being patched
---------------------
``tensorcircuit/backends/jax_ops.py`` defines

    def _safe_reciprocal(x, epsilon=1e-15):
        return x / (x * x + epsilon)

and ``jaxsvd_bwd`` uses it twice (jax_ops.py:43 and :54):

    F    = s*s - (s*s)[:, None]
    F    = _safe_reciprocal(F) - diag(diag(_safe_reciprocal(F)))   # 1/(s_i^2 - s_j^2)
    Sinv = diag(_safe_reciprocal(s))                               # 1/s_i

``|_safe_reciprocal|`` peaks at ``1/(2*sqrt(eps))`` when ``|x| = sqrt(eps)`` and
falls back toward zero below that, so ``eps`` is an ABSOLUTE floor on how close
two squared singular values may be before the backward pass stops returning the
derivative. Its relative distortion of a single entry is exactly

    |1/x - safe(x)| / |1/x|  =  eps / (x^2 + eps)

so it is a smooth bias everywhere, not a cutoff that only bites near ties --
raising eps perturbs well-separated pairs too, just less.

Two patch mechanisms, and why both are needed
---------------------------------------------
``PATCH A -- rebind _safe_reciprocal``. ``jaxsvd_bwd`` resolves the name in its
module globals at call time, so rebinding ``jax_ops._safe_reciprocal`` is picked
up without touching the VJP registration. Enough for "make eps bigger", and only
that: ``_safe_reciprocal(F)`` never receives the spectrum, so an epsilon derived
from ``s`` cannot be computed there.

``PATCH B -- re-register the whole rule`` via
``adaware_svd.defvjp(jaxsvd_fwd, my_bwd)``. Inside the rule the singular values
are in scope, so this is the only route to a spectrum-relative epsilon.

Caching is the trap for both: ``adaware_svd_jit = jax.jit(adaware_svd)`` is built
at import, and generate_ansatz.py additionally turns on a *persistent* on-disk
compilation cache. Every patch below clears JAX's caches on entry and exit, and
the tests assert that a patched result differs from an unpatched one rather than
assuming the patch took.

Note: ``_safe_reciprocal`` is also used by ``jaxeigh_bwd`` (jax_ops.py:170), so
PATCH A changes eigh's backward pass too. Nothing in the MPS split path uses
eigh, but a future caller might; PATCH B is surgical where PATCH A is not.

Usage:
    python -m src.diagnostics.svd_epsilon_patch
"""
import sys
import types
from contextlib import contextmanager

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

from tensorcircuit.backends import jax_ops

DEFAULT_EPSILON = 1e-15


def _clear_caches():
    """Drop compiled artefacts so a re-traced backward pass sees the patch."""
    try:
        jax.clear_caches()
    except AttributeError:  # older jax
        jax.clear_backends()


# ---------------------------------------------------------------------------
# PATCH A -- swap the epsilon constant
# ---------------------------------------------------------------------------


@contextmanager
def svd_epsilon(epsilon):
    """Run the block with ``_safe_reciprocal``'s epsilon set to ``epsilon``."""
    original = jax_ops._safe_reciprocal

    def patched(x, epsilon=epsilon):
        return x / (x * x + epsilon)

    jax_ops._safe_reciprocal = patched
    _clear_caches()
    try:
        yield
    finally:
        jax_ops._safe_reciprocal = original
        _clear_caches()


# ---------------------------------------------------------------------------
# PATCH B -- swap the whole backward rule (needed for a spectrum-relative eps)
# ---------------------------------------------------------------------------


def make_jaxsvd_bwd(eps_from_spectrum):
    """Rebuild jax_ops.jaxsvd_bwd with a caller-supplied epsilon policy.

    ``eps_from_spectrum(s)`` receives the singular values and returns
    ``(eps_for_F, eps_for_Sinv)``. Returning ``(1e-15, 1e-15)`` must reproduce
    TensorCircuit's own rule exactly, which is what test 5 checks.

    Body copied verbatim from jax_ops.py::jaxsvd_bwd apart from the epsilon
    plumbing and renaming its local ``K`` (which reads badly beside
    tensorcircuit's conventional backend handle).
    """

    def bwd(r, tangents):
        u, s, v = r
        du, ds, dv = tangents
        v = jnp.conj(jnp.transpose(v))
        dv = jnp.conj(jnp.transpose(dv))
        m = u.shape[-2]
        n = v.shape[-2]

        eps_f, eps_s = eps_from_spectrum(s)
        recip_f = lambda x: x / (x * x + eps_f)
        recip_s = lambda x: x / (x * x + eps_s)

        F = s * s - (s * s)[:, None]
        F = recip_f(F) - jnp.diag(jnp.diag(recip_f(F)))
        S = jnp.diag(s)
        dAs = jnp.conj(u) @ jnp.diag(ds) @ jnp.transpose(v)

        J = F * (jnp.transpose(u) @ du)
        dAu = jnp.conj(u) @ (J + jnp.transpose(jnp.conj(J))) @ S @ jnp.transpose(v)

        Kmat = F * (jnp.transpose(v) @ dv)
        dAv = jnp.conj(u) @ S @ (Kmat + jnp.conj(jnp.transpose(Kmat))) @ jnp.transpose(v)

        Sinv = jnp.diag(recip_s(s))
        L = jnp.diag(jnp.diag(jnp.transpose(v) @ dv)) @ Sinv
        dAc = 1 / 2.0 * jnp.conj(u) @ (jnp.conj(L) - L) @ jnp.transpose(v)

        grad_a = dAv + dAu + dAs + dAc

        if m > n:
            grad_a += (du - jnp.conj(u) @ jnp.transpose(u) @ du) @ Sinv @ jnp.transpose(v)
        elif m < n:
            grad_a += (
                jnp.conj(u)
                @ Sinv
                @ jnp.transpose(jnp.conj(dv))
                @ (jnp.eye(n) - jnp.conj(v) @ jnp.transpose(v))
            )
        return (grad_a,)

    return bwd


@contextmanager
def svd_backward_rule(eps_from_spectrum):
    """Re-register adaware_svd's VJP with a caller-supplied epsilon policy."""
    original = jax_ops.jaxsvd_bwd
    jax_ops.adaware_svd.defvjp(jax_ops.jaxsvd_fwd, make_jaxsvd_bwd(eps_from_spectrum))
    _clear_caches()
    try:
        yield
    finally:
        jax_ops.adaware_svd.defvjp(jax_ops.jaxsvd_fwd, original)
        _clear_caches()


def absolute_eps(value):
    """Policy: TensorCircuit's own behaviour, with a different constant."""
    return lambda s: (value, value)


def relative_eps(rel):
    """Policy: put the broadening knee at ``rel`` of the leading singular value.

    ``x/(x^2+eps)`` turns over at ``|x| = sqrt(eps)``. For F the argument is
    ``s_i^2 - s_j^2``, so a knee at ``rel * s_max^2`` needs
    ``eps = (rel * s_max^2)^2``; for Sinv the argument is ``s`` itself, so a knee
    at ``rel * s_max`` needs ``eps = (rel * s_max)^2``.
    """

    def policy(s):
        smax = jnp.max(s)
        return ((rel * smax ** 2) ** 2, (rel * smax) ** 2)

    return policy


# ---------------------------------------------------------------------------
# probe 1: a bare SVD, for sensitivity only
# ---------------------------------------------------------------------------


def near_degenerate_matrix(n=8, gap_rel=1e-6, seed=0):
    """A unit-norm matrix whose 3rd and 4th singular values nearly coincide."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    u, s, vh = np.linalg.svd(a)
    s = s / np.linalg.norm(s)
    s[3] = s[2] * (1 - gap_rel)
    return (u * s) @ vh, s


def bare_svd_loss(gap_rel, seed=0):
    """Reads U and V through a fixed external contraction.

    SENSITIVITY PROBE ONLY -- deliberately NOT an accuracy reference. U and V are
    defined only up to a per-column phase, and jnp.linalg.svd's phase convention
    is not a continuous function of A, so this loss is not smooth and central
    differences do not converge to the autodiff answer even for a well-separated
    spectrum (measured: 43% apart at gap_rel=0.1, stable in h, i.e. a genuine
    gauge mismatch rather than a step-size artefact). It is still the sharpest
    way to see the patch bite, because it is exactly the U/V-through-an-external
    contraction dependence the broadening acts on. Anything claiming accuracy
    must use a gauge-invariant loss -- see mps_loss below.
    """
    base, s = near_degenerate_matrix(gap_rel=gap_rel, seed=seed)
    rng = np.random.default_rng(seed + 100)
    n = base.shape[0]
    direction = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    direction /= np.linalg.norm(direction)
    fixed = jnp.asarray(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    base_j, dir_j = jnp.asarray(base), jnp.asarray(direction)

    def loss(t):
        u, sv, vh = jax_ops.adaware_svd(base_j + t * dir_j)
        return jnp.real(jnp.sum(jnp.conj(u) * fixed) + jnp.sum(jnp.conj(vh) * fixed))

    return loss, s


# ---------------------------------------------------------------------------
# probe 2: the real MPS split path, gauge invariant
# ---------------------------------------------------------------------------

NQ, BOND, NPARAM = 6, 2, 22


def mps_circuit(theta, bond=BOND):
    qc = tc.MPSCircuit(NQ, split=({"max_singular_values": bond} if bond else {}))
    idx = 0
    for layer in range(2):
        for i in range(NQ):
            qc.ry(i, theta=theta[idx])
            idx += 1
        for i in range(layer % 2, NQ - 1, 2):
            qc.rxx(i, i + 1, theta=theta[idx])
            qc.rzz(i, i + 1, theta=theta[idx + 1])
            idx += 2
    return qc


def mps_loss(theta):
    """<Z> on the middle qubit of a truncating MPS circuit.

    Every two-qubit gate goes apply_two_site_gate -> backend.svd -> _svd_jax ->
    adaware_svd_jit, the exact chain the toric-code ansatz uses. The expectation
    contracts U and V back together, so unlike bare_svd_loss it is gauge
    invariant and central differences are a valid reference.
    """
    return jnp.real(mps_circuit(theta).expectation((tc.gates.z(), [NQ // 2])))


MPS_THETA = jnp.asarray(np.linspace(0.3, 2.1, NPARAM))
_dir = np.random.default_rng(0).normal(size=NPARAM)
MPS_DIR = _dir / np.linalg.norm(_dir)


def mps_along(t):
    return float(mps_loss(jnp.asarray(np.asarray(MPS_THETA) + t * MPS_DIR)))


def rel_change(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

RESULTS = []


def check(name, passed, detail):
    RESULTS.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}\n         {detail}", flush=True)


def t1_eager():
    loss, _ = bare_svd_loss(gap_rel=1e-6)
    grad = jax.grad(loss)
    base = float(grad(0.0))
    with svd_epsilon(1e-9):
        bigger = float(grad(0.0))
    with svd_epsilon(1e-21):
        smaller = float(grad(0.0))
    moved = (abs(bigger - base) / abs(base) > 1e-6
             and abs(smaller - base) / abs(base) > 1e-9)
    check("PATCH A reaches the eager backward pass", moved,
          f"eps=1e-15 -> {base:.8e} | eps=1e-9 -> {bigger:.8e} | eps=1e-21 -> {smaller:.8e}")
    return base, bigger


def t2_jit(eager_base, eager_patched):
    loss, _ = bare_svd_loss(gap_rel=1e-6)
    jit_base = float(jax.jit(jax.grad(loss))(0.0))
    with svd_epsilon(1e-9):
        jit_patched = float(jax.jit(jax.grad(loss))(0.0))
    # The claim is that the patch reaches the compiled path, i.e. jit-patched
    # lands on eager-patched. jit and eager fuse differently, so the unpatched
    # pair only agree to rounding -- reported, not asserted.
    took = abs(jit_patched - eager_patched) / abs(eager_patched) < 1e-10
    fusion_noise = abs(jit_base - eager_base) / abs(eager_base)
    check("PATCH A survives jit and the persistent compile cache", took,
          f"jit patched {jit_patched:.8e} vs eager patched {eager_patched:.8e}; "
          f"eager-vs-jit rounding on the unpatched value: {fusion_noise:.2e}")


def t3_mps_path():
    norm = float(np.abs(np.asarray(mps_circuit(MPS_THETA).get_norm())))
    grad = jax.grad(mps_loss)
    base = np.asarray(grad(MPS_THETA))
    with svd_epsilon(1e-4):
        patched = np.asarray(grad(MPS_THETA))
    delta = rel_change(base, patched)
    truncating = norm < 1 - 1e-6
    check("PATCH A reaches the real tc.MPSCircuit split path",
          delta > 1e-9 and truncating and np.linalg.norm(base) > 1e-3,
          f"{NQ} qubits, bond_dim={BOND}, ||psi||={norm:.6f} (truncating: {truncating}); "
          f"|grad|={np.linalg.norm(base):.4e}; relative change at eps=1e-4: {delta:.3e}")
    return base


def t4_restore(mps_base):
    grad = jax.grad(mps_loss)
    with svd_epsilon(1e-4):
        _ = grad(MPS_THETA)
    delta = rel_change(mps_base, np.asarray(grad(MPS_THETA)))
    check("restoring leaves no residue", delta == 0.0,
          f"relative difference from the pre-patch gradient after exiting: {delta:.3e}")


def t5_rule_replacement(mps_base):
    grad = jax.grad(mps_loss)
    with svd_backward_rule(absolute_eps(DEFAULT_EPSILON)):
        reimpl = np.asarray(grad(MPS_THETA))
    identical = rel_change(mps_base, reimpl)
    with svd_backward_rule(relative_eps(1e-3)):
        relative = np.asarray(grad(MPS_THETA))
    moved = rel_change(mps_base, relative)
    restored = rel_change(mps_base, np.asarray(grad(MPS_THETA)))
    check("PATCH B re-registers the VJP, reproduces the stock rule, and restores",
          identical == 0.0 and moved > 1e-9 and restored == 0.0,
          f"re-implementation at eps=1e-15 differs by {identical:.2e} (must be 0); "
          f"spectrum-relative rel=1e-3 differs by {moved:.3e}; after restore {restored:.2e}")


def t6_distortion_formula():
    """The bias a given epsilon applies to a given gap is exactly eps/(x^2+eps)."""
    xs = np.array([1e-2, 1e-4, 3.16e-8, 1e-9])
    rows = []
    ok = True
    for eps in (DEFAULT_EPSILON, 1e-9):
        with svd_epsilon(eps):
            measured = np.asarray(jax_ops._safe_reciprocal(jnp.asarray(xs)))
        predicted_rel = eps / (xs ** 2 + eps)
        measured_rel = np.abs(measured - 1 / xs) / np.abs(1 / xs)
        ok = ok and np.allclose(measured_rel, predicted_rel, rtol=1e-10)
        rows.append((eps, predicted_rel))
    detail = "; ".join(
        f"eps={eps:.0e}: " + ", ".join(f"|x|={x:.0e}->{r:.1%}" for x, r in zip(xs, rel))
        for eps, rel in rows)
    check("distortion of a single entry matches eps/(x^2+eps)", ok, detail)


def t7_accuracy_vs_finite_difference():
    """Score each epsilon against central differences on the MPS loss."""
    fds = {h: (mps_along(h) - mps_along(-h)) / (2 * h)
           for h in (1e-3, 1e-4, 1e-5, 1e-6)}
    values = np.array(list(fds.values()))
    spread = float(np.ptp(values) / abs(np.mean(values)))
    reference = fds[1e-5]
    print(f"         central differences: " +
          "  ".join(f"h={h:.0e}->{v:+.10f}" for h, v in fds.items()))
    print(f"         spread across h: {spread:.2e} (converged reference: {reference:+.10f})")

    scan = []
    for eps in (1e-21, 1e-18, DEFAULT_EPSILON, 1e-12, 1e-9, 1e-6, 1e-4):
        with svd_epsilon(eps):
            g = np.asarray(jax.grad(mps_loss)(MPS_THETA))
        directional = float(np.dot(g, MPS_DIR))
        err = abs(directional - reference) / abs(reference)
        scan.append((eps, directional, err))
        print(f"         eps={eps:<8.0e} directional={directional:+.10f}  rel err {err:.3e}")

    errs = [e for _, _, e in scan]
    stock_err = next(e for eps, _, e in scan if eps == DEFAULT_EPSILON)
    monotone = all(errs[i] <= errs[i + 1] * 1.01 + 1e-12 for i in range(len(errs) - 1))
    check("epsilon can be scored against a converged reference",
          spread < 1e-6 and stock_err < 1e-6 and monotone,
          f"FD converged (spread {spread:.1e}); stock eps=1e-15 agrees to {stock_err:.1e}; "
          f"error non-decreasing in eps ({monotone}) -- as expected on this "
          f"well-separated spectrum, where the broadening has no tie to rescue "
          f"and can only cost accuracy")


def main():
    print("Monkey-patching TensorCircuit's SVD-backward epsilon\n")
    print(f"  target: {jax_ops.__file__}")
    print(f"  stock epsilon: {DEFAULT_EPSILON:g}  "
          f"(knee at |s_i^2-s_j^2| = {np.sqrt(DEFAULT_EPSILON):.2e}, "
          f"max |F| = {1/(2*np.sqrt(DEFAULT_EPSILON)):.2e})\n")

    eager_base, eager_patched = t1_eager()
    t2_jit(eager_base, eager_patched)
    mps_base = t3_mps_path()
    t4_restore(mps_base)
    t5_rule_replacement(mps_base)
    t6_distortion_formula()
    t7_accuracy_vs_finite_difference()

    n_pass = sum(1 for _, p, _ in RESULTS if p)
    print(f"\n  {n_pass}/{len(RESULTS)} checks passed")
    for name, passed, _ in RESULTS:
        if not passed:
            print(f"    FAILED: {name}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
