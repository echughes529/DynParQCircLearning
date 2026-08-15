"""T23:
   (a) is _safe_reciprocal's epsilon=1e-15 the knob?  sweep it in complex64.
   (b) what does the split actually buy?  compile time + runtime, split vs no split.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import os, sys, time
import numpy as np
import tensorcircuit as tc
import jax
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
import jax.numpy as jnp
from tensorcircuit.backends import jax_ops

SPLIT = {"max_singular_values": 2, "fixed_choice": 1}
ORIG = jax_ops._safe_reciprocal


def circ(t, split):
    c = tc.Circuit(3, split=split)
    c.ry(0, theta=0.61); c.ry(1, theta=1.24); c.ry(2, theta=0.33)
    c.cnot(1, 2); c.rzz(0, 1, theta=t); c.rx(1, theta=0.77)
    return K.real(c.expectation((tc.gates.z(), [0]), (tc.gates.z(), [1])))


print("=" * 74)
print("(a) sweep tensorcircuit's _safe_reciprocal epsilon, complex64, gate=rzz")
print("    peak height of x/(x^2+eps) is 1/(2 sqrt(eps)), reached at x = sqrt(eps)")
print("=" * 74)
tc.set_dtype("complex64")
ds = [1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
print(f"{'epsilon':>10} {'peak':>10} " + " ".join(f"{d:>11.0e}" for d in ds))
for eps in [1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-6, 1e-5]:
    jax_ops._safe_reciprocal = (lambda e: (lambda x, epsilon=e: x/(x*x + e)))(eps)
    gs = jax.jit(jax.grad(lambda t: circ(t, SPLIT)))
    gr = jax.jit(jax.grad(lambda t: circ(t, None)))
    errs = []
    for d in ds:
        t = np.float64(np.pi/2 + d)
        errs.append(abs(float(gs(t)) - float(gr(t))))
    print(f"{eps:10.0e} {1/(2*np.sqrt(eps)):10.2e} " + " ".join(f"{e:11.3e}" for e in errs))
jax_ops._safe_reciprocal = ORIG

print()
print("=" * 74)
print("(b) what does split={'max_singular_values':2} actually buy?")
print("=" * 74)
import tensorcircuit.quantum as qu
from tensorcircuit.templates.measurements import sparse_expectation
from src.utilities.generate_toric_code_hamiltonian import ToricCode
import fastansatz as fa
tc.set_dtype("complex64")
Lx, Ly, NL, HOW, h = 3, 2, 2, 3, 0.1
lat = ToricCode(Lx, Ly); nq = lat.num_qubits; nplaq = (Lx-1)*(Ly-1)
nanc = nplaq + nplaq*(NL//HOW); npar = nplaq*4*9*NL + 3*nq
st, w = lat.hamiltonian_tc(1-h, nanc); ps, pw = lat.hamiltonian_tc_perturbation(h, nanc)
st.extend(ps); w = np.concatenate((w, pw)); H = qu.PauliStringSum2COO(st, w)
p = np.random.default_rng(0).uniform(0, np.pi, npar)
print(f"toric code {Lx}x{Ly}, {nq+nanc} qubits, {npar} params")
for lbl, sp in [("split (repo)", SPLIT), ("no split", None)]:
    c = fa.dyn_toric(np.float32(p), Lx, Ly, NL, HOW, sp)
    f = K.jit(K.value_and_grad(
        lambda pp: K.real(sparse_expectation(fa.dyn_toric(pp, Lx, Ly, NL, HOW, sp), H))))
    t0 = time.time(); v, _ = f(p); tc_ = time.time()-t0
    t0 = time.time()
    for _ in range(20):
        f(p)
    tr = (time.time()-t0)/20
    print(f"  {lbl:14s}: TN nodes {len(c._nodes):4d}  jit compile {tc_:7.1f}s  "
          f"per grad eval {tr*1000:7.2f} ms   E={float(v):.8f}")
