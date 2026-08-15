"""T29: what does ONE corrupted gradient do to Adam, when nothing is NaN?

Two things happen in sequence, and both are measured below.

  Phase 1 (~ln(G/g)/ln(1/0.9) steps): m is dominated by the spike, so the parameter
          marches at roughly the full learning rate in whatever direction the corrupted
          gradient pointed.  Adam is scale invariant -- m and v are inflated together --
          so the step size is NOT reduced here.
  Phase 2 (then ~ln(1e-3 G^2/g^2)/ln(1/0.999) steps): m has decayed back to the honest
          gradient but v has not, so the effective step collapses by a factor of
          ~g/(0.0316 G) and the parameter is frozen.

Neither phase needs a NaN.  Both are fatal to a 501-step run.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax

LR = 1e-2
G_TYP = 0.1          # a typical honest gradient component in these runs
SPIKE_AT = 50
N = 60000


def run(spike, sign=-1, n=N):
    opt = optax.adam(LR)
    p = jnp.array(0.0); st = opt.init(p)
    ps, us = [], []
    for i in range(n):
        g = jnp.array(sign*spike if i == SPIKE_AT else G_TYP)
        u, st = opt.update(g, st)
        p = optax.apply_updates(p, u)
        ps.append(float(p)); us.append(float(u))
    return np.array(ps), np.array(us)


p0, u0 = run(G_TYP, sign=+1)
print(f"control, no spike: steady step {abs(u0[-1]):.3e} (= lr); "
      f"parameter travels {abs(p0[-1]):.1f} rad in {N} steps\n")

print("One corrupted step of size G, of the wrong sign (the realistic case: the split")
print("gradient at a degeneracy bears no relation to the true one).\n")
hdr = (f"{'G':>10} {'|step| t=51':>12} {'|step| t=150':>13} {'|step| t=400':>13} "
       f"{'freeze factor':>14} {'steps to recover':>17} {'lost progress':>14}")
print(hdr); print("-"*len(hdr))
for spike in [1e2, 1e4, 1e6, 2.3e7, 1e8]:
    ps, us = run(spike)
    a = np.abs(us)
    # recovery: once phase 1 is over (m no longer carries the spike), how long until
    # the step climbs back to half the learning rate?
    start = SPIKE_AT + 300
    idx = np.where(a[start:] > 0.5*LR)[0]
    rec = f"{int(idx[0]) + 300:,}" if len(idx) else f">{N-start:,}"
    print(f"{spike:10.1e} {a[51]:12.3e} {a[150]:13.3e} {a[400]:13.3e} "
          f"{a[400]/LR:14.2e} {rec:>17} {abs(p0[-1])-abs(ps[-1]):14.1f}")

print()
print("Phase 1 length, ln(G/g_typ)/ln(1/0.9):")
for G in [1e4, 2.3e7, 1e8]:
    print(f"   G={G:8.1e} -> {np.log(G/G_TYP)/np.log(1/0.9):5.0f} steps marching the wrong way "
          f"at full learning rate")
print("Phase 2 length, ln(1e-3 G^2/g_typ^2)/ln(1/0.999):")
for G in [1e4, 2.3e7, 1e8]:
    print(f"   G={G:8.1e} -> {np.log(1e-3*G**2/G_TYP**2)/np.log(1/0.999):8,.0f} steps frozen")
print()
print("The repo default is maxiter=501.  A single spike anywhere in the first half of")
print("the run therefore ends that parameter's optimisation permanently -- and t21")
print("measured 58-126 such steps per 800, arriving in bursts once the angles reach")
print("the Clifford points.  This is what 'poor training behaviour with no NaN' looks like.")
