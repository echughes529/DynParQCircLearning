"""T2: what floating point precision does the repo actually simulate in?"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import jax
jax.config.update("jax_enable_x64", True)
import tensorcircuit as tc
import numpy as np
K = tc.set_backend("jax")

print("after `config.update('jax_enable_x64', True)` + `tc.set_backend('jax')` "
      "(exactly what src/find_gs.py does):")
print("  tc.dtypestr  =", tc.dtypestr)
print("  tc.rdtypestr =", tc.rdtypestr)
g = tc.gates.rxx_gate(theta=0.3)
print("  gate tensor dtype :", g.tensor.dtype)
c = tc.Circuit(2)
c.rxx(0, 1, theta=0.3)
print("  statevector dtype :", c.state().dtype)
x = jax.numpy.array([1.0])
print("  a plain jnp float :", x.dtype, "  <- x64 IS on, tc just doesn't use it")

print()
print("machine epsilon:")
print("  float32:", np.finfo(np.float32).eps)
print("  float64:", np.finfo(np.float64).eps)

print()
print("tensorcircuit._safe_reciprocal(x) = x/(x*x+1e-15); peak location/height:")
eps = 1e-15
xpk = np.sqrt(eps)
print(f"  argmax x = sqrt(1e-15) = {xpk:.3e}   value = {xpk/(xpk*xpk+eps):.3e}")
for x in [1e-16, 1e-15, 1e-14, 3.16e-8, 1e-7, 1e-6, 1e-4, 1e-2, 1.0]:
    print(f"  x={x:9.2e} -> {x/(x*x+eps):12.4e}   (1/x = {1/x:.3e})")
