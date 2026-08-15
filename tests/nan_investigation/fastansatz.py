"""A drop-in, O(M) equivalent of the repo's circuit builders (no qc.append rebuild).
   Verified against the repo version in t6_equiv.py."""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT); _sys.path.insert(0, _HERE)
import numpy as np
import tensorcircuit as tc
from src.utilities.generate_toric_code_hamiltonian import ToricCode


def universalsingle(qc, index, params, pi):
    qc.ry(index, theta=params[pi]); qc.rz(index, theta=params[pi+1]); qc.ry(index, theta=params[pi+2])
    return qc, pi + 3


def cartan_inplace(qc, q0, q1, params, pi):
    qc, pi = universalsingle(qc, q0, params, pi)
    qc, pi = universalsingle(qc, q1, params, pi)
    qc.rxx(q0, q1, theta=params[pi])
    qc.ryy(q0, q1, theta=params[pi+1])
    qc.rzz(q0, q1, theta=params[pi+2])
    return qc, pi + 3


def onesetofunitaries(qc, claws, params, pi):
    for cl in claws:
        qc, pi = cartan_inplace(qc, cl[0], cl[1], params, pi)
    return qc, pi


def dyn_toric(params, Lx, Ly, nlayers, howoften, split):
    t = ToricCode(Lx, Ly)
    nplaq = (Lx-1)*(Ly-1); nq = 2*Lx*Ly - Lx - Ly
    qc = tc.Circuit(nq + nplaq + nplaq*(nlayers//howoften), split=split)
    claws = t.all_claws_measurements()
    claws = [claws[i::4][j] for i in range(4) for j in range(nplaq)]
    plaqs = [t.qubit_index(x, y, 2) for x in range(Lx-1) for y in range(Ly-1)]
    pi = 0; mi = 0
    for l in range(nlayers):
        qc, pi = onesetofunitaries(qc, claws, params, pi)
        if l % howoften == howoften-1:
            for p in plaqs:
                qc.cx(p, nq+nplaq+mi); qc.cx(nq+nplaq+mi, p); mi += 1
    for i in range(nq):
        qc, pi = universalsingle(qc, i, params, pi)
    return qc


def unitary_toric(params, Lx, Ly, nlayers, split):
    t = ToricCode(Lx, Ly)
    nplaq = (Lx-1)*(Ly-1); nq = 2*Lx*Ly - Lx - Ly
    qc = tc.Circuit(nq, split=split)
    claws = t.all_claws_unitaries()
    claws = [claws[i::4][j] for i in range(4) for j in range(nplaq)]
    pi = 0
    for l in range(nlayers):
        qc, pi = onesetofunitaries(qc, claws, params, pi)
    for i in range(nq):
        qc, pi = universalsingle(qc, i, params, pi)
    return qc
