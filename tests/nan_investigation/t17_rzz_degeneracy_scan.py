"""T17: THE key measurement.  Gradient error vs distance from the rzz degeneracy at
   theta = pi/2, where the two kept operator-Schmidt values collide (s0 = s1 = sqrt2).

   circuit: 3 qubits, a cnot, then rzz(theta), then rx, observable Z0 Z1.
   The reference is the same circuit with the splitter switched off; the two are
   mathematically identical because rzz has operator-Schmidt rank 2, so the rank-2
   truncation is exact.  The true gradient happens to vanish linearly as theta -> pi/2,
   which makes the error easy to read off.
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
DT = os.environ.get("DT", "complex64")
tc.set_dtype(DT)
EPS = os.environ.get("EPS")          # optionally override tc's _safe_reciprocal epsilon
if EPS:
    from tensorcircuit.backends import jax_ops
    import jax.numpy as jnp
    e = float(EPS)
    jax_ops._safe_reciprocal = lambda x, epsilon=e: x / (x*x + e)
    import importlib
    importlib.reload(jax_ops)          # no-op guard; keep the patched module object
SPLIT = {"max_singular_values": 2, "fixed_choice": 1}
print(f"### tc dtype = {tc.dtypestr}   safe_reciprocal eps override = {EPS}")


def mk(split):
    def e_(t):
        c = tc.Circuit(3, split=split)
        c.ry(0, theta=0.61); c.ry(1, theta=1.24); c.ry(2, theta=0.33)
        c.cnot(1, 2)
        c.rzz(0, 1, theta=t)
        c.rx(1, theta=0.77)
        return K.real(c.expectation((tc.gates.z(), [0]), (tc.gates.z(), [1])))
    return jax.jit(jax.value_and_grad(e_))


fs, fn = mk(SPLIT), mk(None)
print(f"\n{'theta - pi/2':>15} {'s0-s1 (f32)':>13} {'s0^2-s1^2':>12} {'safe_recip':>12} "
      f"{'g_split':>15} {'g_nosplit':>12} {'ERROR':>13}")


def schmidt(th):
    g = tc.gates.rzz_gate(theta=np.float32(th) if DT == "complex64" else np.float64(th))
    t = np.asarray(g.tensor).reshape(2, 2, 2, 2)
    return np.linalg.svd(t.transpose(0, 2, 1, 3).reshape(4, 4), compute_uv=False)


def sr(x, eps=1e-15):
    return x/(x*x+eps)


ds = [0.0] + [s*10.0**(-k) for k in range(0, 13) for s in (1, -1)]
for d in sorted(ds, key=lambda z: (-abs(z), z)):
    th = np.pi/2 + d
    v, g = fs(np.float64(th)); _, gr = fn(np.float64(th))
    s = schmidt(th)
    gap = float(s[0]**2 - s[1]**2)
    print(f"{d:15.1e} {s[0]-s[1]:13.3e} {gap:12.3e} {sr(gap):12.3e} "
          f"{float(g):15.6e} {float(gr):12.3e} {abs(float(g)-float(gr)):13.4e}"
          + ("   <<<" if abs(float(g)-float(gr)) > 1e-3 else ""))
