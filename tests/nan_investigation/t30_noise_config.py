"""T30: is perform_noisy_simulations=True actually simulating any noise?

find_gs.energy_from_params builds

    noise_conf.add_noise("depolarizing", [self.noise_rate*0.1],
                         ["x","y","z","h","s","t","rx","ry","rz"])

but tensorcircuit's signature is

    NoiseConf.add_noise(gate_name, kraus, qubit=None)

so "depolarizing" goes in as the *gate name*, the rate as the *Kraus channel*, and the
list of gates as the *qubit list*.  Three checks: what the config actually contains, what
energy it produces, and what a correctly configured channel at the same rate would give.
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)

import warnings
import numpy as np
import tensorcircuit as tc
import tensorcircuit.noisemodel
import jax
jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")

RATE = 5e-2            # noise_rate used in src/examples/find_gs_tc_example.py
NMC = 200
p = np.linspace(0.3, 2.1, 8)
obs = ((tc.gates.z(), [0]), (tc.gates.z(), [1]))


def build(C):
    c = C(4)
    for i in range(4):
        c.ry(i, theta=p[i])
    for i in range(3):
        c.cnot(i, i + 1)
    for i in range(4):
        c.rx(i, theta=p[4 + i])
    for i in range(3):
        c.cnot(i, i + 1)
    return c


print("=" * 74)
print("(1) what does each noise_conf actually contain?")
print("=" * 74)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    repo_nc = tc.noisemodel.NoiseConf()
    repo_nc.add_noise("depolarizing", [RATE * 0.1],
                      ["x", "y", "z", "h", "s", "t", "rx", "ry", "rz"])
    repo_nc.add_noise("depolarizing", [RATE],
                      ["cnot", "cz", "swap", "iswap", "rxx", "ryy", "rzz"])

    good_nc = tc.noisemodel.NoiseConf()
    one = tc.channels.depolarizingchannel(RATE * 0.1 / 3, RATE * 0.1 / 3, RATE * 0.1 / 3)
    for g in ["rx", "ry", "rz", "x", "y", "z", "h"]:
        good_nc.add_noise(g, one)

def descriptions(nc):
    # NoiseConf.nc is a list of (description, condition, kraus)
    return [d for d, _c, _k in nc.nc]


print(f"  repo's config  -> {len(repo_nc.nc)} rule(s): {descriptions(repo_nc)}")
print(f"  correct config -> {len(good_nc.nc)} rule(s): {descriptions(good_nc)}")
print()
print("  Two things go wrong at once in add_noise(gate_name, kraus, qubit):")
print("   * gate_name='depolarizing' is not a gate -- tc warns and the match condition")
print("     `gatef.n == 'depolarizing'` is false for every gate in the circuit;")
print("   * kraus=[rate] has ONE element while qubit=[9 gate names] has nine, and")
print("     add_noise does `zip(qubit, kraus)`, so eight of the nine are silently")
print("     dropped and the survivor is filed against 'qubit' ('x',) with a float")
print("     where a Kraus list belongs.")

print()
print("=" * 74)
print("(2) energies: noiseless vs the repo's 'noisy' path")
print("=" * 74)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    e_clean = float(K.real(build(tc.Circuit).expectation(*obs)))
    status = K.convert_to_tensor(np.random.default_rng(0).uniform(size=[NMC]))
    e_repo = float(K.real(tc.noisemodel.expectation_noisfy(
        build(tc.Circuit), *obs, noise_conf=repo_nc, nmc=NMC, status=status)))

print(f"  noiseless <Z0 Z1>            : {e_clean:+.10f}")
print(f"  repo's noisy path (nmc={NMC}) : {e_repo:+.10f}")
print(f"  |difference|                 : {abs(e_repo - e_clean):.3e}")

print()
print("=" * 74)
print("(3) what a correctly configured depolarizing channel does at the same rate")
print("=" * 74)
dmn = tc.DMCircuit(4)
s1 = RATE * 0.1 / 3
s2 = RATE / 3
for i in range(4):
    dmn.ry(i, theta=p[i]); dmn.depolarizing(i, px=s1, py=s1, pz=s1)
for i in range(3):
    dmn.cnot(i, i + 1)
    for q in (i, i + 1):
        dmn.depolarizing(q, px=s2, py=s2, pz=s2)
for i in range(4):
    dmn.rx(i, theta=p[4 + i]); dmn.depolarizing(i, px=s1, py=s1, pz=s1)
for i in range(3):
    dmn.cnot(i, i + 1)
    for q in (i, i + 1):
        dmn.depolarizing(q, px=s2, py=s2, pz=s2)
e_true = float(K.real(dmn.expectation(*obs)))
print(f"  density-matrix sim with real depolarizing noise : {e_true:+.10f}")
print(f"  |difference from noiseless|                     : {abs(e_true - e_clean):.3e}")

print()
print("=" * 74)
if abs(e_repo - e_clean) < 1e-5 < abs(e_true - e_clean):
    print("VERDICT: the repo's configuration applies NO NOISE.  The Monte-Carlo loop runs")
    print(f"         nmc={NMC} trajectories that are all the noiseless circuit -- so the run")
    print("         costs nmc times more and returns the noiseless answer.  Real noise at")
    print(f"         the same rate would have shifted the energy by {abs(e_true - e_clean):.2e}.")
else:
    print("VERDICT: inconclusive -- inspect the numbers above.")
print("=" * 74)
