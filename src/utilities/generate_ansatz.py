# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Dynamic Parameterized Quantum Circuit Ansatz Generation."""

import csv
import os

import tensorcircuit as tc
import jax
import jax.numpy as jnp
import tensorflow as tf
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

K = tc.set_backend("jax")
tc.set_dtype("complex128")

# Optional override of the Lorentzian-broadening epsilon in TensorCircuit's
# SVD backward pass (tensorcircuit/backends/jax_ops.py::_safe_reciprocal,
# stock value 1e-15). jaxsvd_bwd amplifies cotangents by up to 1/(2*sqrt(eps))
# whenever a kept singular value or a pair gap |s_i^2 - s_j^2| sits near
# sqrt(eps), so raising eps caps the blowup that near-degenerate spectra feed
# into the gradient (see src/diagnostics/svd_eps_blowup_training.py and
# nan_capture_bd96.py). jaxsvd_bwd looks the name up in its module globals on
# every call, so rebinding it here -- before anything is traced -- changes the
# compiled rule; eps is a constant in the traced HLO, so the persistent
# compilation cache above keys patched and unpatched programs separately.
# jaxeigh_bwd shares this helper and is patched along with it; nothing in the
# MPS split path calls eigh.
_svd_bwd_eps = os.environ.get("DPQC_SVD_BWD_EPS")
if _svd_bwd_eps:
    from tensorcircuit.backends import jax_ops as _tc_jax_ops

    _svd_bwd_eps_val = float(_svd_bwd_eps)

    def _patched_safe_reciprocal(x, epsilon=_svd_bwd_eps_val):
        return x / (x * x + epsilon)

    _tc_jax_ops._safe_reciprocal = _patched_safe_reciprocal
    print(f"[generate_ansatz] SVD-backward epsilon patched: 1e-15 -> {_svd_bwd_eps_val:g} "
          f"(DPQC_SVD_BWD_EPS)")

# Optional override of the SVD forward kernel. Under vmap every MPS split's
# jnp.linalg.svd becomes a batched cuSOLVER call, and the batched Jacobi kernel
# deterministically NaNs single batch elements on the rank-deficient matrices
# this ansatz produces (captured live at 3x3/bd=70 step 6 and 3x3/bd=96; see
# src/diagnostics/nan_capture_bd96.py + fwd_svd_algorithm_test.py). Forcing the
# QR kernel fixed the NaN, matched unbatched values to 13 digits, cut the worst
# batch gradient norm 38.7 -> 2.5, and ran 3.3x FASTER (17.7 vs 58.4 s/step at
# 3x3/bd=70/20 trials on an A40). _svd_jax re-imports `adaware_svd_jit` from
# jax_ops on every call, so rebinding the module attribute redirects every
# split; the custom VJP backward (jaxsvd_bwd) is reused unchanged, so
# DPQC_SVD_BWD_EPS composes with this hook.
_svd_fwd_alg = os.environ.get("DPQC_SVD_FWD_ALG")
if _svd_fwd_alg:
    if _svd_fwd_alg.lower() != "qr":
        raise ValueError(f"DPQC_SVD_FWD_ALG={_svd_fwd_alg!r}: only 'qr' is supported")
    from jax.lax.linalg import svd as _lax_svd, SvdAlgorithm as _SvdAlgorithm
    from tensorcircuit.backends import jax_ops as _tc_jax_ops

    @jax.custom_vjp
    def _qr_adaware_svd(A):
        u, s, vh = _lax_svd(A, full_matrices=False, algorithm=_SvdAlgorithm.QR)
        return (u, s, vh)

    def _qr_adaware_svd_fwd(A):
        u, s, v = _qr_adaware_svd(A)
        return (u, s, v), (u, s, v)

    _qr_adaware_svd.defvjp(_qr_adaware_svd_fwd, _tc_jax_ops.jaxsvd_bwd)
    _tc_jax_ops.adaware_svd_jit = jax.jit(_qr_adaware_svd)
    print("[generate_ansatz] SVD forward kernel patched: batched-Jacobi -> QR "
          "(DPQC_SVD_FWD_ALG)")

from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister, ClassicalRegister
from src.utilities.generate_toric_code_hamiltonian import *

def precompute(c):
    """Not MPS-compatible (converts to DMCircuit for Kraus channel application)."""
    if hasattr(c, 'wavefunction'):
        s = c.wavefunction()
    else:
        s = c.state()
    return tc.DMCircuit(c._nqubits, dminputs=s)

def cartanblock(params=None, paramindex = 0):
    qubit0, qubit1 = 0, 1
    blk = tc.Circuit(2)
    blk, paramindex = universalsingle(blk, qubit0, params, paramindex)
    blk, paramindex = universalsingle(blk, qubit1, params, paramindex)
    blk.rxx(qubit0, qubit1, theta=params[paramindex])
    blk.ryy(qubit0, qubit1, theta=params[paramindex+1])
    blk.rzz(qubit0, qubit1, theta=params[paramindex+2])
    paramindex = paramindex+3
    return blk, paramindex

def apply_cartanblock(qc, qubit0, qubit1, params, paramindex, cartan_mode="separate"):
    """Apply a 9-parameter Cartan block (2x Ry-Rz-Ry + Rxx/Ryy/Rzz) to qubit0, qubit1.

    cartan_mode="separate" applies the six single-qubit rotations and three
    two-qubit rotations as individual gate calls (the original behaviour).
    cartan_mode="fused" builds the same 9-parameter block as a precomputed
    4x4 unitary and applies it as a single two-qubit gate via qc.unitary(),
    so a non-adjacent MPS pair only pays for one swap-in/swap-out instead of
    up to three (see _cartan_gate).
    """
    if cartan_mode == "fused":
        gate, paramindex = _cartan_gate(params, paramindex)
        qc.unitary(qubit0, qubit1, unitary=gate, name="cartan")
        return qc, paramindex
    elif cartan_mode != "separate":
        raise ValueError(f"Unknown cartan_mode: {cartan_mode!r} (expected 'separate' or 'fused')")

    qc, paramindex = universalsingle(qc, qubit0, params, paramindex)
    qc, paramindex = universalsingle(qc, qubit1, params, paramindex)
    qc.rxx(qubit0, qubit1, theta=params[paramindex])
    qc.ryy(qubit0, qubit1, theta=params[paramindex+1])
    qc.rzz(qubit0, qubit1, theta=params[paramindex+2])
    paramindex = paramindex + 3
    return qc, paramindex

def _cartan_gate(params, paramindex):
    """Build the 9-parameter Cartan block as a precomputed 4x4 unitary matrix.

    Same gate sequence/parameter order as the "separate" branch of
    apply_cartanblock, but composed on a standalone 2-qubit circuit and
    extracted via .matrix() so it can be applied as one qc.unitary() call.
    """
    block = tc.Circuit(2)
    block, paramindex = universalsingle(block, 0, params, paramindex)
    block, paramindex = universalsingle(block, 1, params, paramindex)
    block.rxx(0, 1, theta=params[paramindex])
    block.ryy(0, 1, theta=params[paramindex + 1])
    block.rzz(0, 1, theta=params[paramindex + 2])
    paramindex += 3
    return block.matrix(), paramindex

def universalsingle(circuit, index, params, paramindex):
    circuit.ry(index, theta=params[paramindex])
    circuit.rz(index, theta=params[paramindex+1])
    circuit.ry(index, theta=params[paramindex + 2])
    return circuit, paramindex + 3

def onesetofunitaries(qc, claws, params, paramindex, measindx=None,measure=False,seed=None,cartan_mode="separate"):
    # Note: measure=True path is not MPS-compatible (uses precompute/DMCircuit/Kraus channels).
    for cl in claws:
        qc, paramindex = apply_cartanblock(qc, cl[0], cl[1], params, paramindex, cartan_mode=cartan_mode)
        if measure:
            if measindx is None:
                raise ValueError("Please supply an argument for `measindx`")
            locs = cl[0]
            kraus_ops = list(genreset_kraus(params[paramindex],params[paramindex+1],params[paramindex+2]))
            paramindex += 3
            qc = precompute(qc)
            qc.general_kraus(kraus_ops,[(locs,)]) # If using DMCircuit
            measindx += 1
            qc = precompute(qc)
    return qc, paramindex

def onelayerofsingleunitaries(qc, params, paramindex, n=None, qubit_indices=None):
    if qubit_indices is not None:
        for i in qubit_indices:
            qc, paramindex = universalsingle(qc, i, params, paramindex)
    else:
        if n is None:
            n = qc._nqubits
        for i in range(n):
            qc, paramindex = universalsingle(qc, i, params, paramindex)
    return qc, paramindex

def measurements(qc,plaquettes,params, paramindex, measindex):
    for p in plaquettes:
        qc.measure(p,measindex)
        with qc.if_test((measindex, 1)):
            qc.x(p)
        measindex += 1
    return qc, paramindex, measindex

def genreset_kraus(p, theta, phi):
    knothing = tc.gates.Gate(K.cos(p)*np.array([[1, 0], [0, 1]]))
    state = K.convert_to_tensor([[K.cos(theta)], [K.sin(theta)*K.exp(-1j*phi)]])
    zero = np.array([[1], [0]])
    one = np.array([[0], [1]])
    k0 = tc.gates.Gate(K.sin(p)*(state @ zero.conj().transpose()))
    k1 = tc.gates.Gate(K.sin(p)*(state @ one.conj().transpose()))
    return knothing, k0, k1

def construct_circuit(n):
    qc = QuantumCircuit(n)
    params = ParameterVector('θ', n)
    for j in range(n):
        qc.ry(params[j], j)
    for k in range(n-1):
        qc.cx(k, k+1)
    return params, qc

def _normalize_mps_if_requested(qc, normalize_state):
    """Normalize an MPS at a layer boundary when the A/B toggle is enabled."""
    if normalize_state:
        qc.normalize()


def construct_dyn_circuit_brickwork(params,sc,split_conf=None,normalize_state=True):
    if split_conf is None:
        split_conf = make_split_conf()
    nq = sc["nq"]; n_ancillas = sc["n_ancillas"]; nlayers = sc["nlayers"]; howoften = sc["howoften"]
    claws = sc["claws"]; ancillas = sc["ancillas"]; nparams = sc["nparams"]
    backend = sc["backend"]
    paramindex = 0
    measindex = 0
    if backend=="tc":
        qc = tc.MPSCircuit(nq+n_ancillas,split=split_conf)
        for l in range(nlayers):
            qc, paramindex = onesetofunitaries(qc,claws,params,paramindex)
            if howoften and (l % howoften == howoften - 1) and ancillas:
                for a in ancillas:
                    qc.mid_measurement(a, keep=0)
                    measindex += 1
            _normalize_mps_if_requested(qc, normalize_state)

        qc, paramindex = onelayerofsingleunitaries(qc, params, paramindex,nq)
        _normalize_mps_if_requested(qc, normalize_state)

    # elif backend=="qiskit":
    #     parameters = ParameterVector('θ', nparams)
    #     qr = QuantumRegister(nq+n_ancillas)
    #     qc = QuantumCircuit(qr)
    #     for l in range(nlayers):
    #         qc, paramindex = onesetofunitaries(qc,claws,parameters,paramindex,backend="qiskit")
    #         if howoften and (l % howoften == howoften - 1) and ancillas:
    #             for a in ancillas:
    #                 qc.reset(a)
    #                 measindex += 1

    #     qc, paramindex = onelayerofsingleunitaries(qc, parameters, paramindex,nq,backend="qiskit")

    return qc

def construct_unitary_circuit_brickwork(params,sc,split_conf=None,normalize_state=True):
    if split_conf is None:
        split_conf = make_split_conf()
    nq = sc["nq"]; n_ancillas = sc["n_ancillas"]; nlayers = sc["nlayers"]; howoften = sc["howoften"]
    claws = sc["claws"]; ancillas = sc["ancillas"]; nparams = sc["nparams"]
    backend = sc["backend"]
    paramindex = 0
    measindex = 0
    if backend=="tc":
        qc = tc.MPSCircuit(nq+n_ancillas,split=split_conf)

        for l in range(nlayers):
            qc, paramindex = onesetofunitaries(qc,claws,params,paramindex)
            _normalize_mps_if_requested(qc, normalize_state)

        qc, paramindex = onelayerofsingleunitaries(qc, params, paramindex,nq)
        _normalize_mps_if_requested(qc, normalize_state)

    return qc


def construct_dyn_circuit_toriccodelattice(params,Lx,Ly,nlayers = None,howoften=3,
                                           split_conf=None,normalize_state=True,cartan_mode="separate"):
    if split_conf is None:
        split_conf = make_split_conf()
    toriccode = ToricCode(Lx,Ly)
    nplaquettes = (Lx-1)*(Ly-1)
    nq = 2*Lx*Ly - Lx - Ly
    if nlayers is None:
        # nlayers = max(Lx,Ly)
        nlayers = 2
    qc = tc.MPSCircuit(nq+nplaquettes + nplaquettes * (nlayers//howoften), split=split_conf)
    
    nmeasurements = nplaquettes * (nlayers//howoften)
    nparams = nplaquettes * 4 * 9 *nlayers + 0*nmeasurements + 3*nq
    if len(params) != nparams:
        raise ValueError(f"Parameter vector has wrong size: got {len(params)}, expected {nparams}.")
    
    paramindex = 0

    claws = toriccode.all_claws_measurements()
    claws = [claws[i::4][j] for i in range(4) for j in range((Lx-1)*(Ly-1))] # Rearranges them so as to parallelise
    plaquettes = [toriccode.qubit_index(x,y,2) for x in range(Lx-1) for y in range(Ly-1)]
    measindex = 0
    for l in range(nlayers):
        qc, paramindex = onesetofunitaries(qc,claws,params,paramindex,cartan_mode=cartan_mode)
        if l % howoften == howoften-1:
            for p in plaquettes:
                qc.cx(p,nq + nplaquettes + measindex)
                qc.cx(nq+nplaquettes+measindex, p)
                measindex += 1
        _normalize_mps_if_requested(qc, normalize_state)

    qc, paramindex = onelayerofsingleunitaries(qc, params, paramindex,nq)
    _normalize_mps_if_requested(qc, normalize_state)
    return qc

def get_nresets_per_layer_toriccode(Lx, Ly, reset_direction=1):
    """
    Calculate the number of resets per layer for toric code lattice.
    
    Args:
        Lx, Ly: Lattice dimensions
        reset_direction: Direction for reset qubits (0=horizontal, 1=vertical, 2=plaquette corners)
    
    Returns:
        int: Number of valid reset qubits per layer
    """
    toriccode = ToricCode(Lx, Ly)
    reset_qubits = [toriccode.qubit_index(x, y, reset_direction) for x in range(Lx - 1) for y in range(Ly - 1)]
    reset_qubits = [q for q in reset_qubits if q is not None]
    return len(reset_qubits)

def construct_unitary_circuit_toriccodelattice(params,Lx,Ly,nlayers = None,
                                               split_conf=None,normalize_state=True,cartan_mode="separate"):
    if split_conf is None:
        split_conf = make_split_conf()
    toriccode = ToricCode(Lx,Ly)
    nplaquettes = (Lx-1)*(Ly-1)
    nq = 2*Lx*Ly - Lx - Ly
    if nlayers is None:
        # nlayers = max(Lx,Ly)
        nlayers = 2
    qc = tc.MPSCircuit(nq, split=split_conf)

    nparams = nplaquettes * 4 * 9 *nlayers + 3*nq
    if params.shape[0] != nparams:
        raise ValueError(f"Parameter vector has wrong size: got {params.shape[0]}, expected {nparams}.")
    
    paramindex = 0

    claws = toriccode.all_claws_unitaries()
    claws = [claws[i::4][j] for i in range(4) for j in range((Lx-1)*(Ly-1))] # Rearranges them so as to parallelise; still in steps of 4 because the last cartan block between qubit and ancilla is replaced by one between system qubits.
    # plaquettes = [toriccode.qubit_index(x,y,2) for x in range(Lx-1) for y in range(Ly-1)]
    measindex = 0
    for l in range(nlayers):
        qc, paramindex = onesetofunitaries(qc,claws,params,paramindex,cartan_mode=cartan_mode)
        _normalize_mps_if_requested(qc, normalize_state)

    qc, paramindex = onelayerofsingleunitaries(qc, params, paramindex,nq)
    _normalize_mps_if_requested(qc, normalize_state)
    return qc


def _decomposed_ccx(qc, c0, c1, target):
    """Toffoli (CCX) decomposed into 1- and 2-qubit gates.

    Legacy/reproduction path (toffoli_mode="decomposed"). Despite the
    original docstring, tc.MPSCircuit *does* support a direct .ccx()/
    .toffoli() call (used elsewhere in this file, e.g.
    construct_dissipative_ansatz_genresets) -- this decomposition was not
    required for MPSCircuit compatibility. It's kept as the default/legacy
    path pending correctness+gradient validation of toffoli_mode="direct"
    (see src/diagnostics/test_mps_gate_paths.py); it may be a workaround for
    a real numerical issue hit during the original MPS port (git history:
    commit 294afef, "works for 2x2, strange results for 3x2"), not merely an
    API-compatibility shim.
    """
    qc.h(target)
    qc.cnot(c1, target)
    qc.td(target)
    qc.cnot(c0, target)
    qc.t(target)
    qc.cnot(c1, target)
    qc.td(target)
    qc.cnot(c0, target)
    qc.t(c1)
    qc.t(target)
    qc.h(target)
    qc.cnot(c0, c1)
    qc.t(c0)
    qc.td(c1)
    qc.cnot(c0, c1)

# Probability floor for the trajectory-reset sampler. Used as an ADDITIVE floor
# on log-probabilities rather than via jnp.clip: clip has zero gradient outside
# its range, which would silently kill the reset-theta score term exactly at
# theta -> 0 or pi where a reset becomes (almost) certain or impossible.
_TRAJ_PROB_FLOOR = 1e-12


def _sampled_reset(qc, k, theta, u, force=None):
    """Ancilla-free trajectory reset of MPS chain site ``k``. Mutates ``qc``.

    The purified reset in construct_dyn_circuit_toriccodelattice_prob_resets
    spends two ancillas per event to represent the reset channel
    ``rho -> (1-p) rho + p (reset-to-|0>)(rho)``, with ``p = sin^2(theta/2)``,
    coherently. Here we instead sample ONE branch of that channel (a quantum
    trajectory), so no ancillas are needed and the chain stays at nq sites.

    Two facts make the ancillas removable. First, the coin ancilla's marginal is
    ``cos^2(theta/2)`` no matter what the system state is, so it is a classical
    random variable and needs no qubit. Second, once the coin is classical the
    remaining operation is just "measure site k in Z, keep the sampled outcome,
    relabel to |0>". That leaves a 3-way categorical over

        j=0  no reset,          probability 1 - p
        j=1  reset, measured 0, probability p * q0
        j=2  reset, measured 1, probability p * q1

    (the fourth purified outcome, coin=0 with an excited sink, has identically
    zero amplitude). Returns log P(sampled outcome), which the caller sums into
    ``log_w`` and feeds to the DiCE factor in
    VariationalAnsatz.dice_energy_from_params -- WITHOUT that factor the
    gradient with respect to theta is identically zero, since theta only enters
    through the branch probabilities and not through any branch's state.

    Args:
        qc: tc.MPSCircuit being built.
        k: chain position of the system qubit to reset.
        theta: reset angle; p = sin^2(theta/2).
        u: scalar uniform in [0, 1) supplying the randomness. Passed in rather
           than drawn here so the whole forward pass stays a pure function of
           its arguments under jit/vmap, and so tests can hold it fixed.
        force: debug hook. When set to 0, 1 or 2, takes that branch instead of
           sampling (log_w still accumulates the forced branch's log-prob), so
           tests can enumerate all branches exactly instead of relying on Monte
           Carlo. See src/diagnostics/test_trajectory_resets.py.
    """
    # Moving the orthogonality center to k makes every other tensor an isometry,
    # so site k alone carries the Born probabilities of qubit k.
    qc.position(k)
    T = qc._mps.tensors[k]                                    # (Dl, 2, Dr)

    weights = K.real(K.einsum("iaj,iaj->a", T, K.conj(T)))
    # Dividing by the sum is what makes these true probabilities even when
    # truncation has drifted the state norm off 1. Skipping it would leak
    # d log||psi||^2 into log_w, i.e. straight into the gradient.
    q = weights / K.sum(weights)

    p = K.sin(theta / 2.0) ** 2
    probs = K.stack([1.0 - p, p * q[0], p * q[1]])

    if force is None:
        # probability_sample is cumsum + searchsorted only: no Python branching
        # on the outcome, so it survives jit and vmap.
        j = K.probability_sample(1, probs, status=K.reshape(1.0 - u, [1]))[0]
        onehot3 = K.cast(K.onehot(K.cast(j, "int32"), 3), tc.rdtypestr)
    else:
        onehot3 = K.convert_to_tensor(np.eye(3, dtype=np.float64)[int(force)])
        onehot3 = K.cast(onehot3, tc.rdtypestr)

    log_w = K.log(K.sum(onehot3 * probs) + _TRAJ_PROB_FLOOR)

    did_reset = onehot3[1] + onehot3[2]              # exactly 0.0 or 1.0
    onehot_b = K.stack([onehot3[1], onehot3[2]])     # one-hot over the measured bit
    projected = K.einsum("iaj,a->ij", T, K.cast(onehot_b, tc.dtypestr))
    q_b = K.sum(onehot_b * q)
    # Both lanes are always evaluated under vmap, so in the j=0 lane q_b is
    # exactly 0 and an unguarded divide gives 0/0 = nan -- which survives
    # multiplication by did_reset=0, since nan * 0 is nan.
    q_safe = jnp.where(q_b > _TRAJ_PROB_FLOOR, q_b, 1.0)
    projected = projected / K.cast(K.sqrt(q_safe), tc.dtypestr)
    # Write the projected block into physical slot 0 and zero slot 1: the qubit
    # is now |0>, while its correlations with the rest of the chain are intact.
    reset_tensor = jnp.stack([projected, jnp.zeros_like(projected)], axis=1)

    # A one-hot blend, not lax.cond: under vmap JAX turns lax.cond into "run
    # both branches then select", so branching buys nothing. did_reset is
    # exactly 0.0 or 1.0, so one lane survives bit-for-bit. Same idiom as
    # tc's AbstractCircuit.select_gate.
    c = K.cast(did_reset, tc.dtypestr)
    qc._mps.tensors[k] = (1.0 - c) * T + c * reset_tensor
    return log_w


def _branch_reset_matrix(theta, onehot3):
    """2x2 matrix of ONE unnormalised Kraus branch of the reset channel.

    The reset channel ``rho -> (1-p) rho + p (reset-to-|0>)(rho)`` with
    ``p = sin^2(theta/2)`` has exactly three Kraus operators:

        K_0 = cos(theta/2) * I          "no reset"
        K_1 = sin(theta/2) * |0><0|     "reset, qubit was measured 0"
        K_2 = sin(theta/2) * |0><1|     "reset, qubit was measured 1"

    They are complete -- sum_j K_j^dag K_j = (1-p)I + p(|0><0| + |1><1|) = I --
    which is what makes the channel trace-preserving. ``onehot3`` selects one,
    so this returns

                    [ c*h0 + s*h1    s*h2 ]
        M(theta) =  [                     ]   c = cos(theta/2), s = sin(theta/2)
                    [     0         c*h0  ]

    Selecting with a one-hot blend rather than a Python ``if`` is what lets the
    branch index be a *traced* array, so all 3^R branches can be evaluated in one
    batched jax.vmap call instead of compiling 3^R separate programs. Both terms
    are always computed and multiplied by exactly 0.0 or 1.0, so one survives
    bit-for-bit. Same idiom as the did_reset blend in _sampled_reset and as tc's
    AbstractCircuit.select_gate.

    Note that *summing* the three branches instead of selecting one gives
    ``[[c+s, s], [0, c]]``: the coherent superposition, a different (non-CPTP)
    object. Adding the branch energies with their Born weights reproduces the
    channel; adding the branch amplitudes does not.
    """
    c = K.cos(theta / 2.0)
    s = K.sin(theta / 2.0)
    h0, h1, h2 = onehot3[0], onehot3[1], onehot3[2]
    zero = 0.0 * c                  # zeros_like without assuming a backend method
    m = K.stack([K.stack([c * h0 + s * h1, s * h2]),
                 K.stack([zero, c * h0])])
    return K.cast(m, tc.dtypestr)


def _branch_reset(qc, k, theta, branch_idx):
    """Apply one unnormalised Kraus branch of the reset at chain site k. Mutates qc.

    Unlike _sampled_reset this computes no Born probabilities and performs no
    division: the branch weight is carried implicitly by the norm of the state,
    since ``|| K_j |psi> ||^2`` IS the branch probability P_j. The caller
    recovers the exact channel energy ``sum_j P_j E_j`` as

        sum_j <phi_j|H|phi_j> / sum_j <phi_j|phi_j>

    with no normalisation anywhere, and hence no 0/0 on the (many) branches
    whose probability is exactly zero -- the case _sampled_reset has to guard
    with q_safe.

    Args:
        qc: tc.MPSCircuit being built.
        k: chain position of the system qubit to reset.
        theta: reset angle; p = sin^2(theta/2).
        branch_idx: scalar in {0, 1, 2}. May be a traced array, so that the
            enumeration over branches can be vmapped.
    """
    # position(k) first for the same reason as _sampled_reset: a single-site
    # operator applied at the orthogonality centre is the standard TEBD update,
    # and it leaves the canonical form -- and every bond dimension -- intact.
    # A reset therefore costs no SVD and no truncation at all.
    qc.position(k)
    T = qc._mps.tensors[k]                                    # (Dl, 2, Dr)
    onehot3 = K.cast(K.onehot(K.cast(branch_idx, "int32"), 3), tc.rdtypestr)
    qc._mps.tensors[k] = K.einsum("ab,ibj->iaj",
                                  _branch_reset_matrix(theta, onehot3), T)


def construct_branch_circuit_toriccodelattice_prob_resets(params, nlayers, nparams,
                                                          n_mps_qubits, mps_claws,
                                                          mps_reset_qubits,
                                                          active_reset_layers,
                                                          sys_mps_positions, branch,
                                                          split_conf=None,
                                                          cartan_mode="fused"):
    """Build ONE unnormalised Kraus branch of the reset-channel circuit.

    Same parameter layout and consumption order as the trajectory and purified
    builders (one theta per reset event), so the same params vector drives all
    three and their energies are directly comparable.

    Deliberately takes no ``normalize_state``: the surviving norm of each branch
    state IS its probability, so normalising -- per layer or at the end -- would
    throw the branch weights away. The unitary layers are norm-preserving anyway
    (up to truncation), and the residual truncation drift cancels exactly in the
    sum(num)/sum(den) ratio the caller forms. See
    VariationalAnsatz.enumerated_energy_from_params.

    Args:
        branch: one branch index in {0, 1, 2} per reset event, in
            circuit-application order (indexable by the same reset_idx as
            mps_reset_qubits). May be a traced array of shape (total_resets,).
    """
    if split_conf is None:
        split_conf = make_split_conf()
    if len(params) != nparams:
        raise ValueError(f"Parameter vector has wrong size: got {len(params)}, expected {nparams}.")

    qc = tc.MPSCircuit(n_mps_qubits, split=split_conf)
    paramindex = 0

    nresets_per_layer = len(mps_reset_qubits) // len(active_reset_layers) if active_reset_layers else 0
    reset_idx = 0

    for l in range(nlayers):
        qc, paramindex = onesetofunitaries(qc, mps_claws, params, paramindex, cartan_mode=cartan_mode)

        if l in active_reset_layers:
            for _ in range(nresets_per_layer):
                _branch_reset(qc, mps_reset_qubits[reset_idx],
                              params[paramindex], branch[reset_idx])
                paramindex += 1
                reset_idx += 1

    qc, paramindex = onelayerofsingleunitaries(qc, params, paramindex,
                                               qubit_indices=sys_mps_positions)
    return qc


def construct_traj_circuit_toriccodelattice_prob_resets(params, nlayers, nparams,
                                                        n_mps_qubits, mps_claws,
                                                        mps_reset_qubits, active_reset_layers,
                                                        sys_mps_positions, status,
                                                        split_conf=None,
                                                        normalize_state=True,
                                                        cartan_mode="fused",
                                                        force=None):
    """Trajectory-reset counterpart of construct_dyn_circuit_toriccodelattice_prob_resets.

    Same parameter layout and consumption order (one theta per reset event), so
    the same params vector drives both builders and their results can be
    compared directly. Differences: mps_reset_qubits holds bare system chain
    positions rather than (sys, coin, sink) triples, n_mps_qubits is just nq,
    and there is no toffoli_mode since no Toffolis are applied.

    Returns (qc, log_w) where log_w is the summed log-probability of every
    sampled branch. See _sampled_reset for why log_w matters.

    Args:
        status: one uniform in [0, 1) per reset event, in circuit-application
            order (indexable by the same reset_idx as mps_reset_qubits).
        force: None, or a list of forced branch indices (one per reset event)
            for exact enumeration in tests.
    """
    if split_conf is None:
        split_conf = make_split_conf()
    if len(params) != nparams:
        raise ValueError(f"Parameter vector has wrong size: got {len(params)}, expected {nparams}.")

    qc = tc.MPSCircuit(n_mps_qubits, split=split_conf)
    paramindex = 0
    log_w = K.cast(K.convert_to_tensor(0.0), tc.rdtypestr)

    nresets_per_layer = len(mps_reset_qubits) // len(active_reset_layers) if active_reset_layers else 0
    reset_idx = 0

    for l in range(nlayers):
        qc, paramindex = onesetofunitaries(qc, mps_claws, params, paramindex, cartan_mode=cartan_mode)

        if l in active_reset_layers:
            for _ in range(nresets_per_layer):
                log_w = log_w + _sampled_reset(
                    qc,
                    mps_reset_qubits[reset_idx],
                    params[paramindex],
                    status[reset_idx],
                    force=None if force is None else force[reset_idx],
                )
                paramindex += 1
                reset_idx += 1
        _normalize_mps_if_requested(qc, normalize_state)

    qc, paramindex = onelayerofsingleunitaries(qc, params, paramindex,
                                               qubit_indices=sys_mps_positions)
    _normalize_mps_if_requested(qc, normalize_state)
    return qc, log_w


def construct_dyn_circuit_toriccodelattice_prob_resets(params, nlayers, nparams,
                                                       n_mps_qubits, mps_claws,
                                                       mps_reset_qubits, active_reset_layers,
                                                       sys_mps_positions, split_conf=None,
                                                       normalize_state=True,
                                                       cartan_mode="separate",
                                                       toffoli_mode="decomposed"):
    """
    Construct a dynamic circuit for toric code lattice with probabilistic resets.

    All qubit indices are pre-remapped MPS chain positions, computed by
    ToricCodeAnsatz._build_interleaved_ordering().

    Args:
        params: Parameter vector
        nlayers: Number of layers
        nparams: Expected parameter count
        n_mps_qubits: Total qubits in the MPS chain
        mps_claws: List of (q0_mps, q1_mps) pairs for Cartan blocks
        mps_reset_qubits: List of (sys_mps, prob_mps, purif_mps) per reset,
                          ordered by layer then by reset qubit
        active_reset_layers: List of layer indices with resets
        sys_mps_positions: List of system qubit MPS positions for final single-qubit layer
        cartan_mode: "separate" (default) or "fused" -- see apply_cartanblock.
        toffoli_mode: "decomposed" (default) or "direct" -- see _decomposed_ccx.
    """
    if split_conf is None:
        split_conf = make_split_conf()
    if len(params) != nparams:
        raise ValueError(f"Parameter vector has wrong size: got {len(params)}, expected {nparams}.")
    if toffoli_mode not in ("decomposed", "direct"):
        raise ValueError(f"Unknown toffoli_mode: {toffoli_mode!r} (expected 'decomposed' or 'direct')")

    qc = tc.MPSCircuit(n_mps_qubits, split=split_conf)
    paramindex = 0

    nresets_per_layer = len(mps_reset_qubits) // len(active_reset_layers) if active_reset_layers else 0
    reset_idx = 0

    for l in range(nlayers):
        qc, paramindex = onesetofunitaries(qc, mps_claws, params, paramindex, cartan_mode=cartan_mode)

        if l in active_reset_layers:
            for _ in range(nresets_per_layer):
                sys_mps, prob_mps, purif_mps = mps_reset_qubits[reset_idx]

                qc.ry(prob_mps, theta=params[paramindex])
                paramindex += 1

                if toffoli_mode == "direct":
                    qc.toffoli(prob_mps, sys_mps, purif_mps)
                    qc.toffoli(prob_mps, purif_mps, sys_mps)
                else:
                    _decomposed_ccx(qc, prob_mps, sys_mps, purif_mps)
                    _decomposed_ccx(qc, prob_mps, purif_mps, sys_mps)

                reset_idx += 1
        _normalize_mps_if_requested(qc, normalize_state)

    qc, paramindex = onelayerofsingleunitaries(qc, params, paramindex,
                                                qubit_indices=sys_mps_positions)
    _normalize_mps_if_requested(qc, normalize_state)
    return qc



def construct_smallangle_init_toriccodelattice(params, Lx, Ly, nlayers=None,
                                               split_conf=None,normalize_state=True):
    """
    Construct a circuit for toric code lattice with small-angle initialization.
    
    Uses fixed CZ gates for two-qubit interactions and parameterized single-qubit rotations.
    The parameters are expected to be small angles for initialization near the identity.
    
    Args:
        params: Parameter vector for single-qubit rotations
        Lx, Ly: Lattice dimensions
        nlayers: Number of layers (default: 2)
    
    Returns:
        tc.MPSCircuit: The constructed circuit
    """
    if split_conf is None:
        split_conf = make_split_conf()
    toriccode = ToricCode(Lx, Ly)
    nplaquettes = (Lx - 1) * (Ly - 1)
    nq = 2 * Lx * Ly - Lx - Ly

    if nlayers is None:
        nlayers = 2

    # Calculate number of parameters needed:
    # - Single-qubit rotations: 3 parameters per qubit per layer (Ry-Rz-Ry)
    # - nlayers of rotations + CZ gates, plus final layer of rotations
    # - Total: 3 * nq * (nlayers + 1)
    nparams = 3 * nq * (nlayers + 1)
    
    if len(params) != nparams:
        raise ValueError(f"Parameter vector has wrong size: got {len(params)}, expected {nparams}.")
    
    # Create circuit with only system qubits (no ancillas needed)
    qc = tc.MPSCircuit(nq, split=split_conf)
    
    paramindex = 0
    
    # Prepare claws for two-qubit gates
    claws = toriccode.all_claws()
    claws = [claws[i::3][j] for i in range(3) for j in range((Lx - 1) * (Ly - 1))]
    
    # Apply layers
    for l in range(nlayers):
        # Apply single-qubit rotations
        for i in range(nq):
            qc, paramindex = universalsingle(qc,i,params,paramindex)
        # Apply fixed CZ gates on claws
        for cl in claws:
            qc.cz(cl[0], cl[1])
        _normalize_mps_if_requested(qc, normalize_state)
    
    # Final layer of single-qubit rotations
    for i in range(nq):
        qc, paramindex = universalsingle(qc,i,params,paramindex)
    _normalize_mps_if_requested(qc, normalize_state)
    
    return qc


def test_ansatz(param, n, nlayers, n_resets, split_conf=None,normalize_state=True):
    if split_conf is None:
        split_conf = make_split_conf()
    n = 2*n
    zz = np.kron(tc.gates._z_matrix, tc.gates._z_matrix)
    c = tc.MPSCircuit(n + n_resets, split=split_conf)
    paramc = tc.backend.cast(param, tc.dtypestr)  # We assume the input param with dtype float64
    counter = 0
    for i in range(int(n/2)):
        c.ry(i, theta=paramc[counter])
        counter += 1
        c.cnot(i, i+int(n/2))
    _normalize_mps_if_requested(c, normalize_state)
    for j in range(nlayers-1):
        for i in range(n - 1):
            c.exp1(i, i + 1, unitary=zz, theta=paramc[counter])
            counter += 1
        for i in range(n):
            c.rx(i, theta=paramc[counter])
            counter += 1
        _normalize_mps_if_requested(c, normalize_state)
    # Intermediate layer of resets
    counter_aux = 0
    for i in np.random.randint(0, int(n/2), n_resets).tolist():
        c.cnot(i, n+counter_aux)
        c.cnot(n+counter_aux, i)
        counter_aux+=1
    _normalize_mps_if_requested(c, normalize_state)
    # Final layer of rotations
    for i in range(n - 1):
        c.exp1(i, i + 1, unitary=zz, theta=paramc[counter])
        counter += 1
    for i in range(n):
        c.rx(i, theta=paramc[counter])
        counter += 1
    _normalize_mps_if_requested(c, normalize_state)

    return paramc, c

def make_split_conf(bond_dim=None):
    """Truncation config for tc.MPSCircuit's split rules, used when applying
    gates during circuit construction.

    bond_dim=None returns an empty dict: no cap, every singular value kept
    exactly. Pass bond_dim to impose a hard cap on the number of singular
    values kept (e.g. to reproduce a prior fixed-bond-dimension run).

    WARNING: bond_dim=None is not a drop-in "no cap, same as a huge cap"
    setting. tensorcircuit-ng's split_tensor (mpscircuit.py) branches on
    `len(split) > 0`: any non-empty split dict takes an SVD-based path
    (backend.svd); an empty dict takes a QR/RQ-based path instead. These are
    different decomposition algorithms with different autodiff/gradient
    behavior, not just "same math, different truncation." Verified
    empirically: the QR path (bond_dim=None) measurably changed training
    dynamics (e.g. reset-theta parameters failing to converge to their
    otherwise-stable value) compared to an explicit large bond_dim (e.g.
    1000), even though the *forward-pass* bond dimension was identical
    either way. If you want "no real cap" while keeping the known-good SVD
    path, pass an explicit large bond_dim rather than None.

    Deliberately not threshold-based (no max_truncation_err/max_truncation_error):
    that criterion bounds the cumulative Frobenius norm of the *discarded
    tail* of singular values, not each value individually, so for
    near-degenerate spectra -- e.g. the toric code's, which is exactly what
    this codebase targets -- it can discard a run of small-but-similar
    values whose combined norm exceeds the bound even though none of them
    individually would fail a per-value cutoff.
    """
    if bond_dim is None:
        return {}
    return {"max_singular_values": bond_dim}

def get_singular_values_per_cut(qc):
    """Singular-value spectrum at every bond of a canonicalized MPS circuit."""
    n_sites = len(qc.get_tensors())
    singular_values = []
    for site in range(n_sites - 1):
        qc.position(site)  # move orthogonality center so the cut is exact
        tensor = np.asarray(qc.get_tensors()[site])
        d_left, d_phys, d_right = tensor.shape
        matrix = tensor.reshape(d_left * d_phys, d_right)
        s = np.linalg.svd(matrix, compute_uv=False)
        singular_values.append(s)
    return singular_values

def construct_dissipative_ansatz_genresets(n, nlayers, param=None, split_conf=None,
                                           normalize_state=True):
    if split_conf is None:
        split_conf = make_split_conf()
    n_resets = nlayers*(n-1)
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = (nlayers+0.5)*n*15 + 3*n_resets
    if param is None:
        num_params = num_params.astype(int)
        paramc = tf.Variable(initial_value=tf.random.uniform(shape=[num_params]))#, stddev=0.1,))
    else:
        paramc = tc.backend.cast(param, tc.dtypestr)

    
    c = tc.MPSCircuit(n + 2*n_resets, split=split_conf)
    param_counter = 0
    counter_aux = 0
    for l in range(nlayers):
        resets = range(n-1)
        # Even
        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(0, n, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.rz(i, theta=paramc[param_counter+1])
            c.cnot(i+1, i)
            c.ry(i, theta=paramc[param_counter+2])
            c.cnot(i, i + 1)
            param_counter += 3

        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Even Resets; strength is a parameter as well
        for r in resets:
            if np.mod(r, 2) == 0:
                c.ry(int(n + counter_aux + n_resets), theta=paramc[param_counter]) # Strength of reset
                param_counter += 1 
                c.cnot(int(n + counter_aux + n_resets), int(n + counter_aux))
                c.ccx(int(r), int(n + counter_aux + n_resets), int(n + counter_aux))
                U = tc.gates.r_gate(theta=paramc[param_counter], alpha=paramc[param_counter+1], phi=0)
                c.multicontrol(int(n + counter_aux + n_resets), int(n + counter_aux), r, unitary=U,
                               ctrl=[int(n + counter_aux + n_resets), int(n + counter_aux)])
                counter_aux += 1
                param_counter += 2

        # Odd
        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(1, n-1, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.rz(i, theta=paramc[param_counter+1])
            c.cnot(i+1, i)
            c.ry(i, theta=paramc[param_counter+2])
            c.cnot(i, i + 1)
            param_counter += 3

        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Odd Resets
        for r in resets:
            if np.mod(r, 2) == 1:
                c.ry(int(n + counter_aux + n_resets), theta=paramc[param_counter]) # Strength of reset
                param_counter += 1 
                c.cnot(int(n + counter_aux + n_resets), int(n + counter_aux))
                c.ccx(int(r), int(n + counter_aux + n_resets), int(n + counter_aux))
                U = tc.gates.r_gate(theta=paramc[param_counter], alpha=paramc[param_counter+1], phi=0)
                c.multicontrol(int(n + counter_aux + n_resets), int(n + counter_aux), r, unitary=U,
                               ctrl=[int(n + counter_aux + n_resets), int(n + counter_aux)])
                counter_aux += 1
                param_counter += 2
        _normalize_mps_if_requested(c, normalize_state)

    # Coherent Layer
    # Even
    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(0, n, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.rz(i, theta=paramc[param_counter + 1])
        c.cnot(i + 1, i)
        c.ry(i, theta=paramc[param_counter + 2])
        c.cnot(i, i + 1)
        param_counter += 3

    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    # Odd
    for i in range(1, n - 1):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(1, n - 1, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.rz(i, theta=paramc[param_counter + 1])
        c.cnot(i + 1, i)
        c.ry(i, theta=paramc[param_counter + 2])
        c.cnot(i, i + 1)
        param_counter += 3

    for i in range(1, n - 1):
        c.r(i, theta=paramc[param_counter], phi=paramc[param_counter + 1], alpha=paramc[param_counter + 2])
        param_counter += 3
    _normalize_mps_if_requested(c, normalize_state)
    return c


# Density matrix implementation — not MPS-compatible (uses DMCircuit / Kraus channels).
def construct_dissipative_ansatz_dm(n, nlayers, param=None, dm_varqte=False, return_param_counter=False):
    def kraus_depolarizing(p):
        p = jnp.abs(jnp.cos(p))  # Ensure p\in[0,1[
        K0 = jnp.sqrt(1 - p) * jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(p / 3) * jnp.array([[0, 1], [1, 0]])
        K2 = jnp.sqrt(p / 3) * jnp.array([[0, 1j], [-1j, 0]])
        K3 = jnp.sqrt(p / 3) * jnp.array([[1, 0], [0, -1]])
        return K0, K1, K2, K3
    def kraus0(p, q):
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        q = jnp.abs(jnp.cos(q))  # Ensure q\in[0,1[
        K0 = jnp.sqrt(1-p-q)*jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(p) * jnp.array([[0, 0], [0, 1]])
        K2 = jnp.sqrt(p) * jnp.array([[0, 0], [1, 0]])
        K3 = jnp.sqrt(q) * jnp.array([[1, 0], [0, 0]])
        K4 = jnp.sqrt(q) * jnp.array([[0, 1], [0, 0]])
        return K0, K1, K2, K3, K4
    def kraus1(p): #Reset
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        K0 = jnp.sqrt(p)*jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(1-p)*jnp.array([[1, 0], [0, 0]])
        K2 = jnp.sqrt(1-p)*jnp.array([[0, 1], [0, 0]])
        return K0, K1, K2
    def kraus2(p):
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        K0 = jnp.array([[1, 0], [0, jnp.sqrt(1-p)]])
        K1 = jnp.array([[0, 0], [0, jnp.sqrt(p)]])
        return K0, K1
    def kraus3(p, r):
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        r = jnp.abs(jnp.cos(r))  # Ensure r\in[0,1[
        K0 = jnp.sqrt(1-p-r)*jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(p / 2)*jnp.array([[1, 0], [0, 0]])
        K2 = jnp.sqrt(p / 2)*jnp.array([[0, 1], [0, 0]])
        # K3 = jnp.sqrt(r / 2)*jnp.array([[0, 1], [1, 0]])
        K3 = jnp.sqrt(r) * jnp.array([[1, 0], [0, -1]])
        return K0, K1, K2, K3
    def kraus4(p, q, r):
        p = jnp.abs(jnp.cos(p)) # Ensure p\in[0,1[
        q = jnp.abs(jnp.cos(q)) # Ensure q\in[0,1[
        r = jnp.abs(jnp.cos(r))  # Ensure r\in[0,1[
        K0 = jnp.sqrt(1-p-q-r)*jnp.array([[1, 0], [0, 1]])
        K1 = jnp.sqrt(p/2)*jnp.array([[1, 0], [0, 0]])
        K2 = jnp.sqrt(p/2)*jnp.array([[0, 1], [0, 0]])
        K3 = jnp.sqrt(r/2)*jnp.array([[0, 1], [1, 0]])
        K4 = jnp.sqrt(q / 2) * jnp.array([[0, 0], [0, 1]])
        K5 = jnp.sqrt(q / 2) * jnp.array([[0, 0], [1, 0]])
        K6 = jnp.sqrt(r / 2) * jnp.array([[1, 0], [0, -1]])
        return K0, K1, K2, K3, K4, K5, K6
    n_resets = nlayers*(n-1)
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = int(nlayers * (12 * (n - 2) + 20 * n)*50)
    if param is None:
        paramc = np.random.rand(num_params)
        paramc = tc.backend.cast(paramc, tc.dtypestr)
    else:
        paramc = tc.backend.cast(param, tc.dtypestr)
    dmc = tc.DMCircuit(n)
    param_counter = 0
    for l in range(nlayers):
        resets = range(n - 1)
        # Even
        for i in range(n):
            dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(0, n, 2):
            dmc.cnot(i, i + 1)
            dmc.ry(i, theta=paramc[param_counter])
            dmc.rz(i, theta=paramc[param_counter + 1])
            dmc.cnot(i + 1, i)
            dmc.ry(i, theta=paramc[param_counter + 2])
            dmc.cnot(i, i + 1)
            param_counter += 3

        for i in range(n):
            dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Even Resets; strength is a parameter as well
        for r in resets:
            if np.mod(r, 2) == 0:
                K0, K1, K2 = kraus1(paramc[param_counter])  # Strength of reset
                dmc.general_kraus([K0, K1, K2], r)
                param_counter += 1
        # Odd
        for i in range(1, n - 1):
            dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(1, n - 1, 2):
            dmc.cnot(i, i + 1)
            dmc.ry(i, theta=paramc[param_counter])
            dmc.rz(i, theta=paramc[param_counter + 1])
            dmc.cnot(i + 1, i)
            dmc.ry(i, theta=paramc[param_counter + 2])
            dmc.cnot(i, i + 1)
            param_counter += 3

        for i in range(1, n - 1):
            dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Odd Resets
        for r in resets:
            if np.mod(r, 2) == 1:
                K0, K1, K2 = kraus1(paramc[param_counter])  # Strength of reset
                dmc.general_kraus([K0, K1, K2], r)
                param_counter += 1

    # Coherent Layer
    # Even
    for i in range(n):
        dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(0, n, 2):
        dmc.cnot(i, i + 1)
        dmc.ry(i, theta=paramc[param_counter])
        dmc.rz(i, theta=paramc[param_counter + 1])
        dmc.cnot(i + 1, i)
        dmc.ry(i, theta=paramc[param_counter + 2])
        dmc.cnot(i, i + 1)
        param_counter += 3

    for i in range(n):
        dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    # Odd
    for i in range(1, n - 1):
        dmc.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(1, n - 1, 2):
        dmc.cnot(i, i + 1)
        dmc.ry(i, theta=paramc[param_counter])
        dmc.rz(i, theta=paramc[param_counter + 1])
        dmc.cnot(i + 1, i)
        dmc.ry(i, theta=paramc[param_counter + 2])
        dmc.cnot(i, i + 1)
        param_counter += 3

    for i in range(1, n - 1):
        dmc.r(i, theta=paramc[param_counter], phi=paramc[param_counter + 1], alpha=paramc[param_counter + 2])
        param_counter += 3

    if dm_varqte:

        for i in range(0, n):
            K0, K1, K2, K3 = kraus_depolarizing(paramc[param_counter])  # Strength of depolarization
            dmc.general_kraus([K0, K1, K2, K3], i)
            param_counter += 1

    if return_param_counter:
        return param_counter
    else:
        return dmc

def construct_dissipative_ansatz_genresetsDM(n, nlayers, param=None,seed=None,split_conf=None):
    """Not MPS-compatible (uses DMCircuit / Kraus channels)."""
    if split_conf is None:
        split_conf = make_split_conf()
    n_resets = nlayers*(n-1)
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = int((nlayers+1)*(n-1)*9) + 3*n_resets + 3*n
    if param is None:
        paramc = tf.Variable(initial_value=tf.random.uniform(shape=[num_params]))#, stddev=0.1,))
    else:
        if param.shape != (num_params,):
            raise ValueError('The length of the parameter array should be ', num_params, "You have ", param.shape)
        else:
            paramc = tc.backend.cast(param, tc.dtypestr)
    c = tc.densitymatrix.DMCircuit(n, split=split_conf)
    param_counter = 0
    measindx = 0
    claws = []
    for i in range(n-1):
            claws.append([i,i+1])
    claws = [claws[i::2][j] for i in range(2) for j in range(n//2 - i)]

    for l in range(nlayers):
        c, param_counter, measindx = onesetofunitaries(c,claws,param,param_counter,measindx,measure=True,seed=seed)
    for l in range(1):
        c, param_counter, measindx = onesetofunitaries(c,claws,param,param_counter,measindx,measure=False,seed=seed)
    
    c, param_counter = onelayerofsingleunitaries(c, param, param_counter,n)
    return c

def construct_dissipative_ansatz(n, nlayers, resets_nlayer, param=None,
                                 return_param_counter=False, split_conf=None,
                                 normalize_state=True):
    if split_conf is None:
        split_conf = make_split_conf()
    n_resets = nlayers*resets_nlayer
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = int((nlayers+0.5)*n*15)
    if param is None:
        paramc = tf.Variable(initial_value=tf.random.uniform(shape=[num_params]))#, stddev=0.1,))
    else:
        if len(param) != num_params:
            raise ValueError('The length of the parameter array should be ', num_params)
        else:
            paramc = tc.backend.cast(param, tc.dtypestr)
    c = tc.MPSCircuit(n + n_resets, split=split_conf)
    param_counter = 0
    counter_aux = 0
    for l in range(nlayers):
        resets = np.random.randint(0, n, resets_nlayer)
        # Even
        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(0, n, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.rz(i, theta=paramc[param_counter+1])
            c.cnot(i+1, i)
            c.ry(i, theta=paramc[param_counter+2])
            c.cnot(i, i + 1)
            param_counter += 3

        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Even Resets
        for r in resets:
            if np.mod(r, 2) == 0:
                c.cnot(int(r), int(n + counter_aux))
                c.cnot(int(n + counter_aux), int(r))
                counter_aux += 1

        # Odd
        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(1, n-1, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.rz(i, theta=paramc[param_counter+1])
            c.cnot(i+1, i)
            c.ry(i, theta=paramc[param_counter+2])
            c.cnot(i, i + 1)
            param_counter += 3

        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        # Odd Resets
        for r in resets:
            if np.mod(r, 2) == 1:
                c.cnot(int(r), int(n + counter_aux))
                c.cnot(int(n + counter_aux), int(r))
                counter_aux += 1
        _normalize_mps_if_requested(c, normalize_state)

    #Coherent Layer
    # Even
    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3


    for i in range(0, n, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.rz(i, theta=paramc[param_counter + 1])
        c.cnot(i + 1, i)
        c.ry(i, theta=paramc[param_counter + 2])
        c.cnot(i, i + 1)
        param_counter += 3

    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    # Odd
    for i in range(1, n - 1):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(1, n - 1, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.rz(i, theta=paramc[param_counter + 1])
        c.cnot(i + 1, i)
        c.ry(i, theta=paramc[param_counter + 2])
        c.cnot(i, i + 1)
        param_counter += 3

    for i in range(1, n - 1):
        c.r(i, theta=paramc[param_counter], phi=paramc[param_counter + 1], alpha=paramc[param_counter + 2])
        param_counter += 3
    _normalize_mps_if_requested(c, normalize_state)
    if return_param_counter:
        return param_counter
    else:
        return c

def construct_simplified_dissipative_ansatz(n, nlayers, resets_nlayer, param=None,
                                            split_conf=None,normalize_state=True):
    if split_conf is None:
        split_conf = make_split_conf()
    n_resets = nlayers*resets_nlayer
    if np.mod(n, 2) != 0:
        raise ValueError('Please choose the number of qubits as a multiple of 2.')
    num_params = int((nlayers+0.5)*n*15)
    if param is None:
        paramc = tf.Variable(initial_value=tf.random.uniform(shape=[num_params]))#, stddev=0.1,))
    else:
        if len(param) != num_params:
            raise ValueError('The length of the parameter array should be ', num_params, "You have ", len(param))
        else:
            paramc = tc.backend.cast(param, tc.dtypestr)
    c = tc.MPSCircuit(n + n_resets, split=split_conf)
    param_counter = 0
    counter_aux = 0
    for l in range(nlayers):
        resets = np.random.randint(0, n, resets_nlayer)
        # Even
        for i in range(n):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(0, n, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.cnot(i+1, i)
            param_counter += 1

        # Even Resets
        for r in resets:
            if np.mod(r, 2) == 0:
                c.cnot(int(r), int(n + counter_aux))
                c.cnot(int(n + counter_aux), int(r))
                counter_aux += 1

        # Odd
        for i in range(1, n-1):
            c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
            param_counter += 3

        for i in range(1, n-1, 2):
            c.cnot(i, i+1)
            c.ry(i, theta=paramc[param_counter])
            c.cnot(i+1, i)
            param_counter += 1

        # Odd Resets
        for r in resets:
            if np.mod(r, 2) == 1:
                c.cnot(int(r), int(n + counter_aux))
                c.cnot(int(n + counter_aux), int(r))
                counter_aux += 1
        _normalize_mps_if_requested(c, normalize_state)

    # Coherent even layer
    for i in range(n):
        c.r(i, theta=paramc[param_counter], alpha=paramc[param_counter + 1], phi=paramc[param_counter + 2])
        param_counter += 3

    for i in range(0, n, 2):
        c.cnot(i, i + 1)
        c.ry(i, theta=paramc[param_counter])
        c.cnot(i + 1, i)
        param_counter += 1
    _normalize_mps_if_requested(c, normalize_state)
    return c

def construct_fldc_toriccodelattice(Lx,Ly):
    tc = ToricCode(Lx,Ly)
    nq = 2*Lx*Ly - Lx - Ly
    qr = QuantumRegister(nq)

    nblocks = (Lx-1)*(Ly-1)
    nparams = nblocks * 3 * 15

    params = ParameterVector('θ', nparams)
    paramindex = 0
    qc = QuantumCircuit(qr)
    claws = tc.all_claws()

    for cl in claws:
        inst, paramindex = cartanblock(params, paramindex)
        qc.append(inst, qargs=cl)

    return params, qc
