"""T5: does the repo's cartanblock -> qc.append(...) route actually go through the splitter?
   And is the split lossless for the gates used?"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np, jax
import tensorcircuit as tc
K = tc.set_backend("jax")
import src.utilities.generate_ansatz as ga

sc = ga.split_conf
print("repo split_conf =", sc)

qc = tc.Circuit(4, split=sc)
print("circuit_param   =", qc.circuit_param)

blk, _ = ga.cartanblock(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]), 0)
qc.append(blk, indices=[1,2])
shapes = [tuple(n.tensor.shape) for n in qc._nodes]
print("node shapes after append (4 leading (2,) = initial qubits):")
print("  ", shapes)
n4 = [s for s in shapes if len(s)==4]
n3 = [s for s in shapes if len(s)==3]
print(f"  rank-4 (unsplit 2q) nodes: {len(n4)}   rank-3 (SPLIT 2q) nodes: {len(n3)}")
print("  -> split IS applied through append" if len(n3)>0 else "  -> split NOT applied")

# losslessness: unitary of a split circuit vs unsplit
def uni(split):
    c = tc.Circuit(2, split=split)
    c2, _ = ga.cartanblock(np.arange(1,10)*0.37, 0)
    c.append(c2, indices=[0,1])
    return np.asarray(c.matrix())
a, b = uni(sc), uni(None)
print(f"\n|U_split - U_nosplit|_max = {np.max(np.abs(a-b)):.3e}")
print(f"|U_split^H U_split - I|_max = {np.max(np.abs(a.conj().T@a - np.eye(4))):.3e}")

# and for a genuinely rank-4 two qubit gate (e.g. a Haar random one)
rng = np.random.default_rng(1)
M = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4))
q,_ = np.linalg.qr(M)
def uni2(split):
    c = tc.Circuit(2, split=split); c.unitary(0,1,unitary=q); return np.asarray(c.matrix())
a2, b2 = uni2(sc), uni2(None)
print(f"\nHaar-random 2q gate: |U_split - U_exact|_max = {np.max(np.abs(a2-q)):.3e}  "
      f"(unsplit err {np.max(np.abs(b2-q)):.3e})")
print("  -> for a rank-4 gate the max_singular_values=2 split is LOSSY")
