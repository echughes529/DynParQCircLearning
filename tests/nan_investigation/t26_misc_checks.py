"""T26: independent checks of the toric-code setup + two other repo issues."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np, scipy.sparse.linalg as sla
from src.utilities.generate_toric_code_hamiltonian import ToricCode
from qiskit.quantum_info import SparsePauliOp

print("=== (1) exact spectrum of the 3x3 lattice Hamiltonian (reference for training) ===")
for (Lx,Ly) in [(3,3),(3,2),(2,2)]:
    t=ToricCode(Lx,Ly); n=t.num_qubits
    for h in (0.0,0.1):
        H = -(1-h)*(SparsePauliOp(t.all_stars(tc=0))+SparsePauliOp(t.all_plaquettes(tc=0)))
        H = H - SparsePauliOp.from_sparse_list([("Z",[i],h) for i in range(n)],num_qubits=n)
        M=H.to_matrix(sparse=True)
        k=min(8,2**n-2)
        ev=np.sort(sla.eigsh(M,k=k,which="SA",return_eigenvectors=False))
        print(f"  {Lx}x{Ly} n={n} stars={len(t.all_stars())} plaqs={len(t.all_plaquettes())} "
              f"h={h}: E0={ev[0]:.6f}  gap={ev[1]-ev[0]:.6f}  "
              f"degeneracy={int(np.sum(np.abs(ev-ev[0])<1e-9))}")

print("\n=== (2) is `split` lossy for any gate the repo could use? ===")
import tensorcircuit as tc, jax
K=tc.set_backend("jax")
SP={"max_singular_values":2,"fixed_choice":1}
def U(gate_fn, split):
    c=tc.Circuit(2,split=split); gate_fn(c); return np.asarray(c.matrix())
tests={
 "rxx(1.1)":  lambda c: c.rxx(0,1,theta=1.1),
 "ryy(1.1)":  lambda c: c.ryy(0,1,theta=1.1),
 "rzz(1.1)":  lambda c: c.rzz(0,1,theta=1.1),
 "cnot":      lambda c: c.cnot(0,1),
 "cz":        lambda c: c.cz(0,1),
 "swap":      lambda c: c.swap(0,1),
 "iswap":     lambda c: c.iswap(0,1),
 "exp1(ZZ)":  lambda c: c.exp1(0,1,unitary=np.kron(np.diag([1,-1]),np.diag([1,-1])),theta=0.7),
 "cry(0.7)":  lambda c: c.cry(0,1,theta=0.7),
}
for name,fn in tests.items():
    a=U(fn,SP); b=U(fn,None)
    print(f"  {name:10s} max|U_split - U_exact| = {np.max(np.abs(a-b)):.3e}"
          + ("   <<< LOSSY" if np.max(np.abs(a-b))>1e-4 else ""))

print("\n=== (3) ansatz object used as a JIT static argument (find_gs.purity_vec) ===")
from src.utilities.ansatz_classes import ToricCodeAnsatz
try:
    a=ToricCodeAnsatz(Lx=2,Ly=2,nlayers=1,howoften_toreset=1,trials=2,maxiter=2,unitary=True,sparse=False)
    b=ToricCodeAnsatz(Lx=2,Ly=2,nlayers=1,howoften_toreset=1,trials=2,maxiter=2,unitary=False,sparse=False)
    print("  hash(unitary ansatz) == hash(dynamic ansatz)?", hash(a)==hash(b))
    try:
        print("  a == b ->", a==b)
    except Exception as ex:
        print(f"  a == b raises {type(ex).__name__}: {str(ex)[:90]}")
except Exception as ex:
    print("  construction failed:", type(ex).__name__, str(ex)[:200])
