"""T22: (a) the OTHER degeneracy -- theta near 0/pi, where the kept s1 collides with
   the discarded s2 = 0;  (b) do the candidate fixes actually fix it?

   Circuit and reference identical to T17/T19.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import os
import numpy as np
import tensorcircuit as tc
import jax
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
import jax.numpy as jnp

SPLIT2 = {"max_singular_values": 2, "fixed_choice": 1}   # what the repo uses
SPLIT4 = {"max_singular_values": 4, "fixed_choice": 1}   # no truncation, SVD still there


def circ(t, split, gate):
    c = tc.Circuit(3, split=split)
    c.ry(0, theta=0.61); c.ry(1, theta=1.24); c.ry(2, theta=0.33)
    c.cnot(1, 2)
    getattr(c, gate)(0, 1, theta=t)
    c.rx(1, theta=0.77)
    return K.real(c.expectation((tc.gates.z(), [0]), (tc.gates.z(), [1])))


def scan(center, label, gate="rzz", dtname="complex64", split=SPLIT2, eps=None):
    tc.set_dtype(dtname)
    if eps is not None:
        from tensorcircuit.backends import jax_ops
        jax_ops._safe_reciprocal = lambda x, epsilon=eps: x/(x*x + eps)
    g_s = jax.jit(jax.grad(lambda t: circ(t, split, gate)))
    g_r = jax.jit(jax.grad(lambda t: circ(t, None, gate)))
    out = []
    for d in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 0.0]:
        t = np.float64(center + d)
        a, b = float(g_s(t)), float(g_r(t))
        out.append((d, a, b, abs(a-b)))
    print(f"\n--- {label} | gate={gate} dtype={dtname} "
          f"max_sv={split['max_singular_values'] if split else 'NO SPLIT'} eps={eps} ---")
    print(f"{'theta-c':>10} {'g_split':>16} {'g_reference':>15} {'abs error':>13}")
    for d, a, b, e in out:
        print(f"{d:10.1e} {a:16.6e} {b:15.6e} {e:13.4e}"
              + ("   <<<" if e > 1e-3 else ""))
    return max(o[3] for o in out)


print("=" * 78)
print("(a) the two degeneracies, repo configuration (complex64, max_singular_values=2)")
print("=" * 78)
w1 = scan(np.pi/2, "theta -> pi/2   (s0 == s1, both KEPT)", "rzz")
w2 = scan(0.0, "theta -> 0      (kept s1 -> 0 = discarded s2)", "rzz")
w3 = scan(np.pi, "theta -> pi     (kept s1 -> 0 = discarded s2)", "rzz")
w4 = scan(np.pi/2, "theta -> pi/2   (s0 == s1, both KEPT)", "rxx")
print(f"\nworst absolute gradient errors: pi/2(rzz)={w1:.2e}  0(rzz)={w2:.2e}  "
      f"pi(rzz)={w3:.2e}  pi/2(rxx)={w4:.2e}")

print("\n" + "=" * 78)
print("(b) candidate fixes")
print("=" * 78)
f1 = scan(np.pi/2, "FIX 1: tc.set_dtype('complex128')", "rzz", "complex128", SPLIT2)
f2 = scan(np.pi/2, "FIX 2: no split at all", "rzz", "complex64", None)
f3 = scan(np.pi/2, "FIX 3: max_singular_values=4 (SVD kept, no truncation)",
          "rzz", "complex64", SPLIT4)
print(f"\nworst error: fix1(complex128) {f1:.2e} | fix2(no split) {f2:.2e} | "
      f"fix3(max_sv=4, still complex64) {f3:.2e}")
