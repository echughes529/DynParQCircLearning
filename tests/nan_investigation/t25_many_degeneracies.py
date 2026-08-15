"""T25: in the real toric-code ansatz, how big does the gradient get when many
   two-qubit angles sit at the Clifford value pi/2 simultaneously (which is where the
   toric-code optimum drives them)?  Does the damage compound?"""
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
tc.set_dtype(os.environ.get("DT", "complex64"))
import tensorcircuit.quantum as qu
from tensorcircuit.templates.measurements import sparse_expectation
from src.utilities.generate_toric_code_hamiltonian import ToricCode
import fastansatz as fa

SPLIT = {"max_singular_values": 2, "fixed_choice": 1}
Lx, Ly, NL, HOW, h = 3, 2, 2, 3, 0.1
lat = ToricCode(Lx, Ly); nq = lat.num_qubits; nplaq = (Lx-1)*(Ly-1)
nanc = nplaq + nplaq*(NL//HOW); npar = nplaq*4*9*NL + 3*nq
nblocks = nplaq*4*NL
tq = np.concatenate([9*b + np.array([6, 7, 8]) for b in range(nblocks)])
st, w = lat.hamiltonian_tc(1-h, nanc); ps, pw = lat.hamiltonian_tc_perturbation(h, nanc)
st.extend(ps); w = np.concatenate((w, pw)); H = qu.PauliStringSum2COO(st, w)
print(f"dtype={tc.dtypestr} qubits={nq+nanc} npar={npar} two-qubit angles={len(tq)}",
      flush=True)


def mk(split):
    return K.jit(K.value_and_grad(
        lambda p: K.real(sparse_expectation(fa.dyn_toric(p, Lx, Ly, NL, HOW, split), H))))


t0 = time.time(); fs = mk(SPLIT); fr = mk(None)
rng = np.random.default_rng(0)
base = rng.uniform(0, np.pi, npar)
fs(base); fr(base)
print(f"compiled in {time.time()-t0:.0f}s\n", flush=True)

print(f"{'#angles at pi/2':>16} {'E_split':>12} {'E_ref':>12} {'max|g_split|':>14} "
      f"{'max|g_ref|':>12} {'max abs err':>13} {'#g comps >1e3':>14}")
for k in [0, 1, 2, 4, 8, 16, 24, 32, len(tq)]:
    p = base.copy()
    p[tq[:k]] = np.pi/2
    v, g = fs(p); v2, g2 = fr(p)
    g = np.asarray(g, np.float64); g2 = np.asarray(g2, np.float64)
    print(f"{k:16d} {float(v):12.6f} {float(v2):12.6f} {np.max(np.abs(g)):14.4e} "
          f"{np.max(np.abs(g2)):12.4e} {np.max(np.abs(g-g2)):13.4e} "
          f"{int(np.sum(np.abs(g) > 1e3)):14d}")

print("\nsame, but all two-qubit angles at 0 (the other Clifford value):")
p = base.copy(); p[tq] = 0.0
v, g = fs(p); g = np.asarray(g, np.float64)
print(f"   E={float(v):.6f}  max|g|={np.max(np.abs(g)):.4e}  "
      f"non-finite grad components: {int(np.sum(~np.isfinite(g)))}/{g.size}")

print("\nsame, but all two-qubit angles at pi:")
p = base.copy(); p[tq] = np.pi
v, g = fs(p); v2, g2 = fr(p)
g = np.asarray(g, np.float64); g2 = np.asarray(g2, np.float64)
print(f"   E={float(v):.6f}  max|g|={np.max(np.abs(g)):.4e}  "
      f"max abs err vs ref={np.max(np.abs(g-g2)):.4e}  "
      f"non-finite: {int(np.sum(~np.isfinite(g)))}/{g.size}")
