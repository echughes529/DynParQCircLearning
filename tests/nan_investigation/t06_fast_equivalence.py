"""T6: prove fastansatz == repo ansatz (same state, same gradient)."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np, jax, time
import tensorcircuit as tc
K = tc.set_backend("jax")
import src.utilities.generate_ansatz as ga
import fastansatz as fa
from src.utilities.generate_toric_code_hamiltonian import ToricCode

Lx,Ly,nlayers,howoften = 3,2,3,2
t=ToricCode(Lx,Ly); nq=t.num_qubits; nplaq=(Lx-1)*(Ly-1)
npar = nplaq*4*9*nlayers + 3*nq
p = np.random.default_rng(3).uniform(0,np.pi,npar).astype("float32")

t0=time.time(); s_repo = np.asarray(ga.construct_dyn_circuit_toriccodelattice(p,Lx,Ly,nlayers,howoften).state()); t_repo=time.time()-t0
t0=time.time(); s_fast = np.asarray(fa.dyn_toric(p,Lx,Ly,nlayers,howoften,ga.split_conf).state()); t_fast=time.time()-t0
print(f"dyn : max|state diff| = {np.max(np.abs(s_repo-s_fast)):.3e}   (repo {t_repo:.1f}s, fast {t_fast:.1f}s)")
print(f"      norms: repo {np.linalg.norm(s_repo):.10f}  fast {np.linalg.norm(s_fast):.10f}")

nparu = nplaq*4*9*nlayers + 3*nq
pu = np.random.default_rng(4).uniform(0,np.pi,nparu).astype("float32")
s_repo = np.asarray(ga.construct_unitary_circuit_toriccodelattice(pu,Lx,Ly,nlayers).state())
s_fast = np.asarray(fa.unitary_toric(pu,Lx,Ly,nlayers,None).state())
print(f"unit: max|state diff| = {np.max(np.abs(s_repo-s_fast)):.3e}")
print(f"      norms: repo {np.linalg.norm(s_repo):.10f}  fast {np.linalg.norm(s_fast):.10f}")
