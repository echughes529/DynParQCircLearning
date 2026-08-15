"""T28: consistency checks on the parts of the pipeline the SVD report did not cover.

  (a) sparse=True vs sparse=False must give the same energy (they build the Hamiltonian
      by two completely different routes).
  (b) the ancilla count baked into the Hamiltonian must match the circuit for every
      ToricCodeAnsatz branch, otherwise the Pauli strings are padded to the wrong width.
  (c) the claws[i::4][j] re-ordering must be a permutation.
  (d) the noisy path: does the per-step seed actually change the sampled noise under
      jit+vmap, and how large is the resulting shot noise on the gradient?
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)

import io, contextlib
import numpy as np
import tensorcircuit as tc
import tensorcircuit.noisemodel  # tc 1.9.1 does not auto-import this submodule
import jax
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")

from src.utilities.ansatz_classes import ToricCodeAnsatz
from src.utilities.generate_toric_code_hamiltonian import ToricCode
import src.utilities.generate_ansatz as ga


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


print("=" * 74)
print("(a) sparse=True vs sparse=False energy, same parameters")
print("=" * 74)
for kw in [dict(unitary=True), dict(unitary=False),
           dict(use_small_angle_initialization=True)]:
    lbl = list(kw)[0]
    try:
        a_sp = quiet(ToricCodeAnsatz, Lx=2, Ly=2, nlayers=2, howoften_toreset=1,
                     h=0.1, trials=2, maxiter=2, sparse=True, **kw)
        a_de = quiet(ToricCodeAnsatz, Lx=2, Ly=2, nlayers=2, howoften_toreset=1,
                     h=0.1, trials=2, maxiter=2, sparse=False, **kw)
        p = np.asarray(a_sp.initparams[0], np.float64)
        e_sp = float(a_sp.energy_from_params(p))
        e_de = float(a_de.energy_from_params(p))
        print(f"  {lbl:34s} sparse {e_sp:+.8f}   term-by-term {e_de:+.8f}   "
              f"diff {abs(e_sp-e_de):.2e}"
              + ("   <<< MISMATCH" if abs(e_sp-e_de) > 1e-4 else ""))
    except Exception as ex:
        print(f"  {lbl:34s} RAISED {type(ex).__name__}: {str(ex)[:90]}")

print()
print("=" * 74)
print("(b) does self.nancillas (used to pad the Pauli strings) match the circuit?")
print("=" * 74)
for kw in [dict(unitary=True), dict(unitary=False),
           dict(use_prob_resets=True), dict(use_small_angle_initialization=True)]:
    lbl = list(kw)[0]
    try:
        a = quiet(ToricCodeAnsatz, Lx=3, Ly=3, nlayers=2, howoften_toreset=1,
                  h=0.1, trials=1, maxiter=2, sparse=False, **kw)
        qc = a._circuit(np.asarray(a.initparams[0], np.float64))
        want = a.lattice.num_qubits + a.nancillas
        print(f"  {lbl:34s} nancillas={a.nancillas:3d} -> expects {want:3d} qubits, "
              f"circuit has {qc._nqubits:3d}"
              + ("   <<< MISMATCH" if want != qc._nqubits else "   ok"))
    except Exception as ex:
        print(f"  {lbl:34s} RAISED {type(ex).__name__}: {str(ex)[:90]}")

print()
print("=" * 74)
print("(c) claws[i::4][j] re-ordering: permutation, or silently dropping gates?")
print("=" * 74)
for (Lx, Ly) in [(2, 2), (3, 2), (3, 3), (4, 3), (4, 4)]:
    t = ToricCode(Lx, Ly)
    nplaq = (Lx-1)*(Ly-1)
    for name, claws, stride in [("measurements", t.all_claws_measurements(), 4),
                                ("unitaries", t.all_claws_unitaries(), 4),
                                ("plain (prob_resets)", t.all_claws(), 3)]:
        try:
            re = [claws[i::stride][j] for i in range(stride) for j in range(nplaq)]
            ok = (len(re) == len(claws)) and (sorted(map(tuple, re)) == sorted(map(tuple, claws)))
            print(f"  {Lx}x{Ly} {name:20s} len={len(claws):3d} expected={stride*nplaq:3d} "
                  + ("permutation ok" if ok else "<<< NOT A PERMUTATION"))
        except Exception as ex:
            print(f"  {Lx}x{Ly} {name:20s} len={len(claws):3d} expected={stride*nplaq:3d} "
                  f"<<< {type(ex).__name__}")

print()
print("=" * 74)
print("(d) noisy path: does the seed change anything?  how big is the shot noise?")
print("=" * 74)
try:
    a = quiet(ToricCodeAnsatz, Lx=2, Ly=2, nlayers=1, howoften_toreset=1, h=0.1,
              trials=1, maxiter=2, unitary=True, sparse=False,
              perform_noisy_simulations=True, noise_rate=5e-2, number_of_shots=20)
    p = np.asarray(a.initparams[0], np.float64)
    f = K.jit(K.value_and_grad(lambda pp, s: a.energy_from_params(pp, seed=s), argnums=0))
    vals, grads = [], []
    for seed in [42, 43, 44, 100, 1000]:
        v, g = f(p, seed)
        vals.append(float(v)); grads.append(np.asarray(g, np.float64))
    vals = np.array(vals); grads = np.array(grads)
    print(f"  energies over 5 different seeds : {np.array2string(vals, precision=8)}")
    print(f"  spread (max-min)                : {vals.max()-vals.min():.3e}")
    print(f"  max spread of any grad component: {np.max(grads.max(0)-grads.min(0)):.3e}")
    if vals.max()-vals.min() == 0:
        print("  -> the seed has NO effect: every step samples the SAME noise realisation")
    else:
        print("  -> the seed does change the sampled noise")
    # noiseless reference for the same params
    a0 = quiet(ToricCodeAnsatz, Lx=2, Ly=2, nlayers=1, howoften_toreset=1, h=0.1,
               trials=1, maxiter=2, unitary=True, sparse=False)
    e0 = float(a0.energy_from_params(p))
    print(f"  noiseless energy                : {e0:+.8f}   (noisy mean {vals.mean():+.8f})")
except Exception as ex:
    import traceback
    print("  RAISED", type(ex).__name__, str(ex)[:200])
    traceback.print_exc(limit=3)
