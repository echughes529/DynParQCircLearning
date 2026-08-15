"""T24: epsilon sweep done properly (jax caches must be cleared, and
   adaware_svd_jit rebuilt, or the patch never reaches the compiled code);
   plus an instrumented backward pass running UNDER jit."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np
import tensorcircuit as tc
import jax
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
import jax.numpy as jnp
from tensorcircuit.backends import jax_ops

SPLIT = {"max_singular_values": 2, "fixed_choice": 1}
tc.set_dtype("complex64")


def circ(t, split):
    c = tc.Circuit(3, split=split)
    c.ry(0, theta=0.61); c.ry(1, theta=1.24); c.ry(2, theta=0.33)
    c.cnot(1, 2); c.rzz(0, 1, theta=t); c.rx(1, theta=0.77)
    return K.real(c.expectation((tc.gates.z(), [0]), (tc.gates.z(), [1])))


def make_bwd(eps, verbose=False):
    def bwd(r, tangents):
        u, s, v = r
        du, ds, dv = tangents
        sr = lambda x: x/(x*x + eps)
        v = jnp.conj(jnp.transpose(v)); dv = jnp.conj(jnp.transpose(dv))
        F0 = s*s - (s*s)[:, None]
        SR = sr(F0)
        F = SR - jnp.diag(jnp.diag(SR))
        S = jnp.diag(s)
        dAs = jnp.conj(u) @ jnp.diag(ds) @ jnp.transpose(v)
        J = F * (jnp.transpose(u) @ du)
        dAu = jnp.conj(u) @ (J + jnp.transpose(jnp.conj(J))) @ S @ jnp.transpose(v)
        Kk = F * (jnp.transpose(v) @ dv)
        dAv = jnp.conj(u) @ S @ (Kk + jnp.conj(jnp.transpose(Kk))) @ jnp.transpose(v)
        Sinv = jnp.diag(sr(s))
        L = jnp.diag(jnp.diag(jnp.transpose(v) @ dv)) @ Sinv
        dAc = 0.5 * jnp.conj(u) @ (jnp.conj(L) - L) @ jnp.transpose(v)
        ga = dAv + dAu + dAs + dAc
        if verbose:
            jax.debug.print(
                "   [bwd] s={s}\n         F0={f0}\n         max|F|={mf} max|Sinv|={ms} "
                "max|du|={du} max|dv|={dv}\n         |dAs|={a} |dAu|={b} |dAv|={c} "
                "|dAc|={d} -> |gA|={e}",
                s=s, f0=F0.ravel()[:8], mf=jnp.max(jnp.abs(F)), ms=jnp.max(jnp.abs(Sinv)),
                du=jnp.max(jnp.abs(du)), dv=jnp.max(jnp.abs(dv)),
                a=jnp.max(jnp.abs(dAs)), b=jnp.max(jnp.abs(dAu)), c=jnp.max(jnp.abs(dAv)),
                d=jnp.max(jnp.abs(dAc)), e=jnp.max(jnp.abs(ga)))
        return (ga,)
    return bwd


def install(eps, verbose=False):
    jax_ops.adaware_svd.defvjp(jax_ops.jaxsvd_fwd, make_bwd(eps, verbose))
    jax.clear_caches()
    jax_ops.adaware_svd_jit = jax.jit(jax_ops.adaware_svd)


ds = [1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
print("(a) epsilon sweep, complex64, gate=rzz, scanning theta = pi/2 + d")
print(f"{'epsilon':>10} {'peak 1/(2sqrt(e))':>18} " + " ".join(f"{d:>11.0e}" for d in ds))
for eps in [1e-15, 1e-12, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4]:
    install(eps)
    gs = jax.jit(jax.grad(lambda t: circ(t, SPLIT)))
    gr = jax.jit(jax.grad(lambda t: circ(t, None)))
    errs = [abs(float(gs(np.float64(np.pi/2+d))) - float(gr(np.float64(np.pi/2+d))))
            for d in ds]
    print(f"{eps:10.0e} {1/(2*np.sqrt(eps)):18.2e} " + " ".join(f"{e:11.3e}" for e in errs))

print("\n(b) instrumented backward, UNDER jit, eps=1e-15 (tensorcircuit's value)")
install(1e-15, verbose=True)
gs = jax.jit(jax.grad(lambda t: circ(t, SPLIT)))
for d in [1e-2, 1e-7, 0.0]:
    print(f"  --- theta = pi/2 + {d:.0e} ---")
    val = float(gs(np.float64(np.pi/2 + d)))
    print(f"  => dE/dtheta = {val:.6e}\n")
