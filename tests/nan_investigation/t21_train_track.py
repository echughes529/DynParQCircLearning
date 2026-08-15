"""T21: real training run, instrumented.

Every step we compute the gradient BOTH ways -- through the repo's splitter and with
the splitter disabled (mathematically identical, since every 2-qubit gate in this
ansatz has operator-Schmidt rank <= 2) -- and report the discrepancy, together with
how close the two-qubit angles are to the degeneracies at multiples of pi/2.

MODE=c64 (repo) | c128 (fix)
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import os, sys, time
import numpy as np
import jax
MODE = os.environ.get("MODE", "c64")
if MODE == "c128":
    jax.config.update("jax_enable_x64", True)
import tensorcircuit as tc
K = tc.set_backend("jax")
tc.set_dtype("complex128" if MODE == "c128" else "complex64")
jax.config.update("jax_enable_x64", True)      # params stay float64 either way, as in the repo
import jax.numpy as jnp, optax
import tensorcircuit.quantum as qu
from tensorcircuit.templates.measurements import sparse_expectation
from src.utilities.generate_toric_code_hamiltonian import ToricCode
import fastansatz as fa

SPLIT = {"max_singular_values": 2, "fixed_choice": 1}
Lx = int(os.environ.get("LX", 3)); Ly = int(os.environ.get("LY", 2))
NL = int(os.environ.get("NL", 2)); HOW = int(os.environ.get("HOW", 2))
TR = int(os.environ.get("TRIALS", 12)); STEPS = int(os.environ.get("STEPS", 800))
LR = float(os.environ.get("LR", 1e-2)); SEED = int(os.environ.get("SEED", 0))
h = 0.1
lat = ToricCode(Lx, Ly); nq = lat.num_qubits; nplaq = (Lx-1)*(Ly-1)
nanc = nplaq + nplaq*(NL//HOW); npar = nplaq*4*9*NL + 3*nq
nblocks = nplaq*4*NL
tq = np.concatenate([9*b + np.array([6, 7, 8]) for b in range(nblocks)])
print(f"MODE={MODE} tc.dtype={tc.dtypestr} L={Lx}x{Ly} qubits={nq+nanc} npar={npar} "
      f"2q-angles/trial={len(tq)} trials={TR} steps={STEPS}", flush=True)

st, w = lat.hamiltonian_tc(1-h, nanc)
ps, pw = lat.hamiltonian_tc_perturbation(h, nanc)
st.extend(ps); w = np.concatenate((w, pw))
H = qu.PauliStringSum2COO(st, w)


def mk(split):
    return K.jit(K.value_and_grad(
        lambda p: K.real(sparse_expectation(fa.dyn_toric(p, Lx, Ly, NL, HOW, split), H))))


t0 = time.time(); f_split = mk(SPLIT); f_ref = mk(None)
p0 = np.asarray(jax.random.uniform(jax.random.PRNGKey(SEED), shape=[TR, npar],
                                   minval=0, maxval=jnp.pi), np.float64)
f_split(p0[0]); print(f"  split jit compiled {time.time()-t0:.0f}s", flush=True)
t0 = time.time(); f_ref(p0[0]); print(f"  ref   jit compiled {time.time()-t0:.0f}s", flush=True)

params = p0.copy()
opt = optax.adam(learning_rate=LR); state = opt.init(jnp.asarray(params))
worst = 0.0; nbad = 0; nonfinite_step = None
t0 = time.time()
for i in range(STEPS):
    vs = np.zeros(TR); gs = np.zeros((TR, npar)); gr = np.zeros((TR, npar))
    for t in range(TR):
        v, g = f_split(params[t]); _, g2 = f_ref(params[t])
        vs[t] = float(v); gs[t] = np.asarray(g, np.float64); gr[t] = np.asarray(g2, np.float64)
    err = np.abs(gs - gr)
    scale = max(np.max(np.abs(gr)), 1e-12)
    rel = np.max(err)/scale
    th = params[:, tq]
    dmin = np.min(np.abs(((th + np.pi/4) % (np.pi/2)) - np.pi/4))   # dist to nearest k*pi/2
    if rel > 1e-2:
        nbad += 1
    worst = max(worst, rel if np.isfinite(rel) else np.inf)
    bad = (not np.all(np.isfinite(gs))) or (not np.all(np.isfinite(vs)))
    if bad and nonfinite_step is None:
        nonfinite_step = i
    if i % 20 == 0 or rel > 1e-2 or bad:
        print(f"step {i:4d} Emin={vs.min():+.6f} max|g|={np.max(np.abs(gs)):.3e} "
              f"| split-vs-ref: abs {np.max(err):.3e} rel {rel:.3e} "
              f"| closest 2q angle to k*pi/2: {dmin:.2e}"
              + ("  <<< CORRUPTED GRADIENT" if rel > 1e-2 else "")
              + ("  <<< NON-FINITE" if bad else "")
              + f"  [{time.time()-t0:.0f}s]", flush=True)
    if bad:
        np.save(f"bad_params_{MODE}_{SEED}.npy", params); break
    u, state = opt.update(jnp.asarray(gs), state)
    params = np.asarray(optax.apply_updates(jnp.asarray(params), u), np.float64)

print(f"\nDONE mode={MODE} seed={SEED}: steps with rel-error>1e-2: {nbad}/{i+1}; "
      f"worst rel error {worst:.3e}; first non-finite step {nonfinite_step}; "
      f"final Emin {vs.min():+.6f}", flush=True)
