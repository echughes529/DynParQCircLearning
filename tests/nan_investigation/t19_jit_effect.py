"""T19: does jit change the answer at the degeneracy?  (unmodified tensorcircuit)"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np, tensorcircuit as tc, jax
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
import os
tc.set_dtype(os.environ.get("DT","complex64"))
SPLIT={"max_singular_values":2,"fixed_choice":1}
def e_(t, split):
    c=tc.Circuit(3,split=split)
    c.ry(0,theta=0.61); c.ry(1,theta=1.24); c.ry(2,theta=0.33)
    c.cnot(1,2); c.rzz(0,1,theta=t); c.rx(1,theta=0.77)
    return K.real(c.expectation((tc.gates.z(),[0]),(tc.gates.z(),[1])))
print(f"dtype={tc.dtypestr}")
print(f"{'theta-pi/2':>12} {'grad NO jit':>16} {'grad WITH jit':>16} {'truth':>10}")
gj = jax.jit(jax.grad(lambda t: e_(t,SPLIT)))
gp = jax.grad(lambda t: e_(t,SPLIT))
for d in [1e-2,1e-4,1e-5,1e-6,1e-7,1e-8,0.0]:
    t=np.float64(np.pi/2+d)
    print(f"{d:12.1e} {float(gp(t)):16.6e} {float(gj(t)):16.6e} {0.0:10.1f}")
