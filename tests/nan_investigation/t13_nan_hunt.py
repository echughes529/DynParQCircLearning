"""T13: exhaustive NaN hunt near the degeneracies, in the EXACT repo configuration
   (jax x64 ON for the parameters, tensorcircuit dtype complex64 for the tensors --
   which is what src/find_gs.py + src/utilities/generate_ansatz.py produce)."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np
import tensorcircuit as tc          # sets x64 OFF
import jax
jax.config.update("jax_enable_x64", True)   # ...and the repo turns it back ON
K = tc.set_backend("jax")
print(f"tc.dtypestr={tc.dtypestr}  jax x64={jax.config.jax_enable_x64}  "
      f"param dtype={jax.numpy.array([1.0]).dtype}")

SPLIT = {"max_singular_values": 2, "fixed_choice": 1}


def mk(gate, split):
    def e(t):
        c = tc.Circuit(3, split=split)
        c.ry(0, theta=0.61); c.ry(1, theta=1.24); c.ry(2, theta=0.33)
        c.cnot(1, 2)
        getattr(c, gate)(0, 1, theta=t)
        c.rx(1, theta=0.77)
        return K.real(c.expectation((tc.gates.z(), [0]), (tc.gates.z(), [1])))
    return jax.jit(jax.value_and_grad(e))


for gate in ["rxx", "ryy", "rzz"]:
    f = mk(gate, SPLIT)
    fn = mk(gate, None)
    bad, worst = [], 0.0
    # dense grid + exact special points + values approached from both sides
    pts = set()
    for c in [0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi, -np.pi/2, -np.pi]:
        pts.add(c)
        for e_ in range(1, 18):
            for s in (+1, -1):
                pts.add(c + s*10.0**(-e_))
        for k in range(1, 60):      # walk in float64 ulps of the special point
            pts.add(np.nextafter(c, np.inf) if k == 1 else
                    float(np.nextafter(np.float64(c), np.inf*np.sign(1)) + k*1e-16))
    pts |= set(np.linspace(-0.05, 6.35, 4001).tolist())
    pts = sorted(pts)
    for t in pts:
        v, g = f(np.float64(t))
        if not (np.isfinite(float(v)) and np.isfinite(float(g))):
            bad.append(t)
        else:
            _, gr = fn(np.float64(t))
            worst = max(worst, abs(float(g) - float(gr)))
    print(f"{gate}: scanned {len(pts)} thetas -> {len(bad)} non-finite; "
          f"worst |g_split - g_nosplit| among finite = {worst:.3e}")
    if bad:
        print(f"      non-finite at theta = {bad[:12]}{' ...' if len(bad)>12 else ''}")

print("\n--- the same for the constant entangler used in the reset gadget (cnot) ---")


def e_cnot(t, split):
    c = tc.Circuit(3, split=split)
    c.ry(0, theta=t); c.ry(1, theta=1.24)
    c.cnot(0, 1); c.cnot(1, 0)
    return K.real(c.expectation((tc.gates.z(), [0]),))


gc = jax.jit(jax.value_and_grad(lambda t: e_cnot(t, SPLIT)))
gn = jax.jit(jax.value_and_grad(lambda t: e_cnot(t, None)))
w = 0.0; nb = 0
for t in np.linspace(-0.01, 6.3, 2001).tolist() + [0.0, np.pi/2, np.pi, 2*np.pi]:
    v, g = gc(np.float64(t)); _, g2 = gn(np.float64(t))
    if not np.isfinite(float(g)):
        nb += 1
    else:
        w = max(w, abs(float(g)-float(g2)))
print(f"cnot/cnot reset gadget: {nb} non-finite, worst |dg| = {w:.3e}")
