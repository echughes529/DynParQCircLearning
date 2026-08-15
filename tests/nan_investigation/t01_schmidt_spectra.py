"""T1: operator-Schmidt spectra of every 2-qubit gate used in the ansatze,
    in exactly the reshaping convention tensorcircuit's _split_two_qubit_gate uses."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import tensorcircuit as tc
K = tc.set_backend("jax")

def schmidt_svals(g4):
    """g4: tc Gate tensor of shape (2,2,2,2) with legs (out0,out1,in0,in1).
    tn.split_node uses left_edges=[n[0], n[2]] (out0,in0), right=[n[1],n[3]] (out1,in1)."""
    t = np.asarray(g4).reshape(2, 2, 2, 2)
    m = t.transpose(0, 2, 1, 3).reshape(4, 4)
    return np.linalg.svd(m, compute_uv=False)

print("=== constant 2q gates ===")
for name in ["cnot", "cz", "swap", "iswap"]:
    g = getattr(tc.gates, name)()
    print(f"{name:6s}", np.round(schmidt_svals(g.tensor), 12))

print()
print("=== parameterised 2q rotations: singular values vs theta ===")
for name in ["rxx", "ryy", "rzz"]:
    print(f"-- {name} --")
    for th in [0.0, 1e-8, 1e-4, 0.5, np.pi/2 - 1e-8, np.pi/2, np.pi/2 + 1e-4,
               2.0, np.pi - 1e-4, np.pi]:
        g = getattr(tc.gates, name + "_gate")(theta=th)
        s = schmidt_svals(g.tensor)
        print(f"   theta={th: .12f}  s={np.round(s,10)}  "
              f"s0^2-s1^2={s[0]**2-s[1]**2: .3e}  s1^2-s2^2={s[1]**2-s[2]**2: .3e}")
    print()

print("=== analytic check: rxx svals should be 2|cos(t/2)|, 2|sin(t/2)|, 0, 0 ===")
for th in [0.3, 1.1, np.pi/2, 2.7]:
    g = tc.gates.rxx_gate(theta=th)
    s = schmidt_svals(g.tensor)
    pred = np.sort([2*abs(np.cos(th/2)), 2*abs(np.sin(th/2)), 0.0, 0.0])[::-1]
    print(f"   theta={th:.4f} numeric={np.round(s,10)} analytic={np.round(pred,10)} "
          f"maxdiff={np.max(np.abs(s-pred)):.2e}")
