"""
NaN-gradient investigation and debugging playground for TensorCircuit's
MPS SVD-based gate splitting.

Context: training runs at large lattice sizes (see plot_training_curve_tc.py,
e.g. Lx=Ly=3, bond_dim=64) produce NaN gradients. This file was rebuilt from
scratch (superseding an earlier version of test.py) after digging into
TensorCircuit's and TensorNetwork's actual source and empirically testing each
piece with the real installed packages. Findings, in the order discovered:

1. Raw jnp.linalg.svd's default JAX gradient DOES blow up (huge, sometimes
   literal NaN, essentially a floating-point coin flip) for degenerate
   singular values -- but only for a loss that depends on U/V through an
   "external" contraction (e.g. against a fixed matrix), not one expressible
   purely via U^T U (a naive loss like sum(U**2 * S) is gauge-invariant and
   will NOT expose this -- an earlier attempt at Section 1 used exactly that
   wrong shape of loss and wrongly appeared to show no problem at all).
   Confirmed worst case: truncating a bond_dim cutoff straddles a tied block
   of singular values (bond_dim keeps some but not all of them) -- this is
   the scenario that actually matters for toric-code MPS truncation, since
   toric code ground states are stabilizer states with a provably FLAT
   entanglement spectrum.

2. TensorCircuit does NOT naively rely on jnp.linalg.svd's default gradient
   for MPS splits. At import time, tensorcircuit/backends/jax_backend.py:197
   monkeypatches tensornetwork.backends.jax.jax_backend.JaxBackend.svd
   (the actual method tensorcircuit/mps_base.py's FiniteMPS.apply_two_site_gate
   calls) with its own `_svd_jax`, which routes through `adaware_svd` in
   tensorcircuit/backends/jax_ops.py -- a jax.custom_vjp SVD with a
   Lorentzian-broadened backward pass (`_safe_reciprocal`, eps=1e-15) AND a
   proper complex phase-gauge correction term. So: the fix IS wired into the
   real MPS path (this contradicts what an earlier version of this file
   assumed -- tapping tc.backend alone is misleading either way, since MPS
   splits never go through tc.backend regardless of whether the fix is
   present).

3. Empirically (Section 2 below, using the REAL installed tensorcircuit),
   this fix correctly handles every degenerate case from Section 1, including
   truncation straddling a tied block, AND the real tc.MPSCircuit at an
   exactly degenerate gate angle -- giving the exact correct closed-form
   gradient, not NaN. It even survives 1000 chained near-degenerate splits at
   full float64 precision without blowing up (Section 3).

4. SHARP EDGE, worth knowing regardless of the above: `import tensorcircuit`
   itself forces jax_enable_x64=False as a side effect (cons.py's set_dtype()
   runs unconditionally at import with default dtype "complex64", which
   explicitly calls config.update("jax_enable_x64", False)). It is only
   restored by an explicit tc.set_dtype("complex128") call AFTER the import
   -- which is why every real entry point in this repo does exactly that
   (see generate_ansatz.py:23-24). This file follows the same order. Getting
   this wrong while debugging silently drops you to float32, which makes
   near-degenerate spectra far more dangerous and can produce misleading
   "blowup" results that don't exist at production precision (this happened
   once while building this file -- an apparent 1e6->1e11 compounding blowup
   over 200 chained splits turned out to be a complex64 artifact; at proper
   complex128 it doesn't happen at all, see Section 3).

So: in every controlled test here, TensorCircuit's existing fix works. That
means the real training NaNs are NOT simply "raw degenerate SVD gradient" --
something specific to the real, large, deep circuit is needed to explain it.
You've reported NaNs only show up ~100 steps into real training, not from the
initial parameters -- so a static repro can't show this; it needs an actual
instrumented training run (Section 4). That needs your real venv (this
sandbox's tensorflow/qiskit/cotengra stack has a numpy/pandas version
conflict that couldn't be resolved without risking the environment, so it
hasn't been run end-to-end here -- run it on Eddie / the scratch venv).

Section 4's working hypothesis is NOT about the reset gates specifically --
the probabilistic reset only touches a handful of ancilla pairs in specific
layers, a small local operation. The toric code's flat entanglement spectrum
is a property of the ground state itself, built up by the Cartan blocks
across every claw in every layer, over the whole lattice. A better
explanation for "NaN only ~100 steps in": at initialization the variational
state is far from the true ground state (generic, non-degenerate
entanglement); as training converges toward the true low-energy state --
exactly flat-spectrum by construction -- degeneracies should emerge and
sharpen across many system-qubit cuts simultaneously, independent of any
single gate's angle. That would also explain why Section 3's stress test
(many splits at a FIXED near-degenerate gap) didn't reproduce it: the real
failure mode may be the gap closing further over training, not just
compounding across cuts at a fixed gap. So Section 4 tracks lattice-wide
spectrum degeneracy (max tied multiplicity, smallest gap) alongside energy
and grad-norm at every step, and dumps a full snapshot the moment NaN hits.

Section 5 is a quick (no 100-step wait) static sanity check comparing SVD vs
QR paths on the real ansatz at increasing sizes. Section 6 has notes on
follow-ups (e.g. jaxlib/CUDA version) if Section 4 confirms GPU-specific
behavior via its FORCE_CPU toggle below.

Toggle sections independently via the RUN_SECTION_* flags below.
"""
import os

FORCE_CPU = False  # set True to run entirely on CPU (compares against GPU's
# cuSOLVER SVD, a different implementation from CPU LAPACK with its own known
# numerical-stability quirks for near-degenerate matrices). Must be set here,
# before jax is imported -- toggling it later in the process has no effect.
if FORCE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import tensorcircuit as tc
K = tc.set_backend("jax")
tc.set_dtype("complex128")  # must come AFTER importing tensorcircuit -- see note 4 above
from tensorcircuit.backends.jax_ops import adaware_svd

assert jnp.array([1.0], dtype=jnp.complex128).dtype == jnp.complex128, (
    "jax_enable_x64 did not stick after importing tensorcircuit -- "
    "everything below would silently run at float32. See docstring note 4."
)

RUN_SECTION_1 = False
RUN_SECTION_2 = False
RUN_SECTION_3 = False
RUN_SECTION_4 = True  # live training run -- needs the real venv, can be slow; see docstring
RUN_SECTION_5 = False  # static scaling sweep -- needs the real venv, quick
RUN_SECTION_6_NOTES = False

# Section 4 (live training diagnostic) config -- defaults match the failing
# job in logs/2026-08-11_15-41-44_tc_3x3_bd_64_print_grad_3593082
LIVE_LX = 3
LIVE_LY = 3
LIVE_NLAYERS = 2
LIVE_HOWOFTEN_TORESET = 7
LIVE_RESET_LAYERS = [1]
LIVE_BOND_DIM = 64
LIVE_MAXITER = 200            # you reported NaN ~100 steps in; this gives margin
LIVE_PRINT_EVERY = 5          # spectrum check is a forward-only extra pass, so not every step
LIVE_HISTORY_KEEP = 10        # how many recent steps to show in the snapshot dump
LIVE_USE_OPTIMAL_ORDERING = False  # skip the slow qubit-ordering search; doesn't change the physics
LIVE_NORMALIZE_STATE = True  # A/B toggle: run the same LIVE_SEED once False, once True
LIVE_TRIALS = 5  # match the original run's trial count. IMPORTANT: with trials=1, the
# same LIVE_SEED only reproduces trial 0's initial params (jax.random.uniform(key,
# shape=(1,N)) is bit-identical to row 0 of shape=(trials,N) for the same key -- verified
# empirically in this repo's venv -- but trial 0 isn't necessarily the trial that NaN'd.
# find_gs.py's tqdm postfix reports jnp.min(value) across all trials each step, and NaN
# propagates through jnp.min, so "Current value: nan" in a job log just means *some*
# trial went NaN, not which one. Keep trials=5 here so every trial from the original run
# replays identically; the loop below reports which trial index(es) actually NaN.
LIVE_SEED = 22405  # reproducing job 3606420 (logs/2026-08-13_17-01-43_not_normed_nr_3x3_bd64_3606420),
# which NaN'd around step 20-21 of 1200. None -> draw + log a fresh seed each run (was previously always the case,
# silently and unlogged -- ToricCodeAnsatz now logs "parameter init seed: N" on construction).
# Once a run reproduces a NaN, set LIVE_SEED = N (that exact value) here to replay the identical
# initialization deterministically instead of hoping to get unlucky again.

# Section 5 (static sweep) config
LATTICE_SIZES = [(2, 2), (2, 3), (3, 3)]  # extend e.g. with (3, 4)
SECTION5_BOND_DIM = 8  # a real bond_dim -> forces the SVD split path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _symmetric_test_matrix(key, m, singular_values):
    """m x m symmetric matrix (like a reduced density matrix) with a chosen
    exact singular-value spectrum -- Q @ diag(S) @ Q^T for random orthogonal Q."""
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (m, m)))
    S = jnp.diag(jnp.array(singular_values))
    return (Q @ S @ Q.T).astype(jnp.complex128)


def _external_svd_loss(svd_fn):
    """A loss depending on U through an EXTERNAL fixed matrix C, mimicking how
    the split tensor gets contracted with the rest of an MPS chain. This is
    the shape of loss that actually exposes the degenerate-singular-value
    gradient issue -- a loss built only from U^T U (e.g. sum(U**2 * S)) is
    gauge-invariant and will NOT show any problem, degenerate or not."""
    def loss(A, keep, C):
        u, s, vh = svd_fn(A)
        return jnp.real(jnp.sum(u[:, :keep] * C) + jnp.sum(s[:keep]))
    return loss


def _bell_pair_energy(theta, split_conf):
    """3-qubit MPS: ry(theta) then two CNOTs. At theta=pi/2 the first bond's
    Schmidt spectrum is exactly degenerate [1/sqrt2, 1/sqrt2] -- same
    structural pattern as the reset gate (ry(theta) then two CCXs) in
    construct_dyn_circuit_toriccodelattice_prob_resets
    (generate_ansatz.py:296-307). The true energy is cos(theta) (closed
    form), so the true gradient -sin(theta) is finite everywhere, including
    at theta=pi/2 -- any NaN seen here would be a pure AD artifact.
    """
    qc = tc.MPSCircuit(3, split=split_conf)
    qc.ry(0, theta=theta)
    qc.cnot(0, 1)
    qc.cnot(1, 2)
    return K.real(qc.expectation((tc.gates.z(), [2])))


_bell_pair_grad = K.grad(_bell_pair_energy, argnums=0)


def _report(label, g):
    g = jnp.asarray(g)
    print(f"  {label}: nan={bool(jnp.isnan(g).any())}  "
          f"max_abs={float(jnp.nan_to_num(jnp.max(jnp.abs(g)), nan=-1)):.4e}")


# ---------------------------------------------------------------------------
# Section 1 -- raw jnp.linalg.svd's default gradient at degenerate spectra
# ---------------------------------------------------------------------------

def section1_raw_svd_degeneracy():
    print("\n" + "=" * 70)
    print("SECTION 1: raw jnp.linalg.svd default gradient at degenerate spectra")
    print("=" * 70)

    def raw_svd(A):
        return jnp.linalg.svd(A, full_matrices=False)

    loss = _external_svd_loss(raw_svd)
    grad_fn = jax.grad(loss)
    key, key_c = jax.random.PRNGKey(0), jax.random.PRNGKey(1)

    print("-- non-degenerate vs a 2-fold-degenerate pair (both within 'keep') --")
    A_nondeg = _symmetric_test_matrix(key, 6, [2.0, 1.5, 1.0, 0.8, 0.5, 0.3])
    A_deg = _symmetric_test_matrix(key, 6, [2.0, 2.0, 1.5, 1.0, 0.7, 0.3])
    C = jax.random.normal(key_c, (6, 3)).astype(jnp.complex128)
    _report("non-degenerate", grad_fn(A_nondeg, 3, C))
    _report("degenerate    ", grad_fn(A_deg, 3, C))

    print("-- truncation cutting THROUGH a 6-fold-tied block (the realistic case: "
          "toric code's flat spectrum wider than bond_dim) --")
    A_straddle = _symmetric_test_matrix(key, 8, [2.0] * 6 + [1.0, 0.5])
    C3 = jax.random.normal(key_c, (8, 3)).astype(jnp.complex128)
    C6 = jax.random.normal(key_c, (8, 6)).astype(jnp.complex128)
    _report("keep=3 (straddles the tied block)", grad_fn(A_straddle, 3, C3))
    _report("keep=6 (whole tied block kept)   ", grad_fn(A_straddle, 6, C6))


# ---------------------------------------------------------------------------
# Section 2 -- TensorCircuit's real fix (adaware_svd), on the same cases
# ---------------------------------------------------------------------------

def section2_adaware_svd_fix():
    print("\n" + "=" * 70)
    print("SECTION 2: TensorCircuit's adaware_svd on the same degenerate cases")
    print("=" * 70)

    loss = _external_svd_loss(adaware_svd)
    grad_fn = jax.grad(loss)
    key, key_c = jax.random.PRNGKey(0), jax.random.PRNGKey(1)

    A_deg = _symmetric_test_matrix(key, 6, [2.0, 2.0, 1.5, 1.0, 0.7, 0.3])
    C = jax.random.normal(key_c, (6, 3)).astype(jnp.complex128)
    _report("2-fold degenerate (Section 1 gave nan/huge)", grad_fn(A_deg, 3, C))

    A_straddle = _symmetric_test_matrix(key, 8, [2.0] * 6 + [1.0, 0.5])
    C3 = jax.random.normal(key_c, (8, 3)).astype(jnp.complex128)
    _report("truncation straddles 6-tied block (Section 1 gave nan)", grad_fn(A_straddle, 3, C3))

    print("\n-- real tc.MPSCircuit, degenerate reset-like angle, SVD vs QR path --")
    called, called_tn, restore = tap_backends()
    try:
        for label, split_conf in [
            ("QR path  (bond_dim=None)", make_split_conf_local(None)),
            ("SVD path (bond_dim=8)   ", make_split_conf_local(8)),
        ]:
            for theta_label, theta in [("generic 0.7", 0.7), ("degenerate pi/2", jnp.pi / 2)]:
                g = _bell_pair_grad(theta, split_conf)
                true_grad = -jnp.sin(theta)
                print(f"  {label} | {theta_label}: grad={float(g): .6f}  "
                      f"true={float(true_grad): .6f}  nan={bool(jnp.isnan(g))}")
    finally:
        restore()
    print(f"\n  tc.backend calls (expected ~0, wrong backend for MPS splits): {called}")
    print(f"  tensornetwork JaxBackend calls (the real, patched ones):        {called_tn}")


def make_split_conf_local(bond_dim):
    """Same logic as generate_ansatz.make_split_conf, duplicated here so this
    file has no dependency on the rest of the repo for Sections 1-3."""
    if bond_dim is None:
        return {}
    return {"max_singular_values": bond_dim}


def tap_backends():
    """Wrap svd/qr/eigh/rq on tc.backend and TensorNetwork's JaxBackend to
    count calls. tc.backend is NOT what performs MPS two-qubit-gate splits --
    FiniteMPS.apply_two_site_gate calls self.backend.svd, and self.backend is
    the (monkeypatched, see docstring note 2) tensornetwork JaxBackend class.
    Returns (called, called_tn, restore); call restore() to undo the patch.
    """
    from tensornetwork.backends.jax.jax_backend import JaxBackend as TNJaxBackend

    called = {"svd": 0, "qr": 0, "eigh": 0, "rq": 0}
    called_tn = {"svd": 0, "qr": 0, "eigh": 0, "rq": 0}
    originals = []

    for name in list(called):
        if hasattr(tc.backend, name):
            orig = getattr(tc.backend, name)
            originals.append((tc.backend, name, orig))
            def make(n, f):
                def w(*a, **k):
                    called[n] += 1
                    return f(*a, **k)
                return w
            setattr(tc.backend, name, make(name, orig))
        if hasattr(TNJaxBackend, name):
            orig_tn = getattr(TNJaxBackend, name)
            originals.append((TNJaxBackend, name, orig_tn))
            def make_tn(n, f):
                def w(self, *a, **k):
                    called_tn[n] += 1
                    return f(self, *a, **k)
                return w
            setattr(TNJaxBackend, name, make_tn(name, orig_tn))

    def restore():
        for obj, name, orig in originals:
            setattr(obj, name, orig)

    return called, called_tn, restore


# ---------------------------------------------------------------------------
# Section 3 -- does the fix survive compounding (many sequential near-
# degenerate splits, as in one forward pass of a deep circuit)?
# ---------------------------------------------------------------------------

def section3_compounding_stress_test():
    print("\n" + "=" * 70)
    print("SECTION 3: adaware_svd under many chained near-degenerate splits")
    print("=" * 70)
    print("  (an earlier attempt at this showed a 1e6->1e11 'blowup' over 200 "
          "splits that turned out to be a complex64 artifact from the "
          "tensorcircuit x64-reset gotcha -- see docstring note 4. Rerun here "
          "at confirmed complex128.)")

    m = 8
    key = jax.random.PRNGKey(0)
    C = jax.random.normal(jax.random.PRNGKey(99), (m, m)).astype(jnp.complex128)

    def chained_loss(theta, n_splits):
        x = jnp.eye(m, dtype=jnp.complex128)
        for i in range(n_splits):
            Q, _ = jnp.linalg.qr(jax.random.normal(jax.random.fold_in(key, i), (m, m)))
            Q = Q.astype(jnp.complex128)
            svals = jnp.array([2.0, 2.0 + theta] + [1.5 - 0.1 * j for j in range(m - 2)],
                               dtype=jnp.complex128)
            A = x @ Q @ jnp.diag(svals) @ Q.conj().T
            u, s, v = adaware_svd(A)
            x = u
        return jnp.real(jnp.sum(x * C))

    for n_splits in [1, 20, 100, 500, 1000]:
        theta = 1e-8  # near-exact degeneracy injected at every single split
        g = jax.grad(chained_loss)(theta, n_splits)
        print(f"  n_splits={n_splits:5d}: grad={float(g):.6e}  "
              f"nan={bool(jnp.isnan(g))}  inf={bool(jnp.isinf(g))}")


# ---------------------------------------------------------------------------
# Section 4 -- LIVE instrumented training run (primary diagnostic): does the
# real ToricCodeAnsatz actually NaN partway through training, and what does
# the lattice-wide singular-value spectrum look like as it happens? See the
# docstring for why this is centered on lattice-wide spectrum degeneracy vs.
# convergence, not on any single gate's angle. Needs the real venv
# (tensorflow/qiskit/cotengra) -- not runnable in every sandbox.
# ---------------------------------------------------------------------------

def _spectrum_degeneracy_summary(spectra, atol=1e-8, floor_rel=1e-6):
    """Lattice-wide degeneracy summary across ALL cuts.

    Filters out singular values below floor_rel * (largest value in that
    cut) first -- otherwise a near-zero noise-floor tail (values the
    circuit's bond_dim cap already suppressed towards zero) masquerades as
    'degenerate': two values that are both ~1e-18 are numerically close to
    each other for a totally uninteresting reason, not because of a real
    near-degenerate Schmidt pair. (An earlier version of this function
    didn't floor the tail and reported min_gap~1e-18 at basically every
    step regardless of training progress -- almost certainly this artifact,
    not a real signal.)

    max_multiplicity is the largest run of consecutive (sorted-descending,
    post-floor) values within atol of their neighbour, anywhere in any cut
    -- not just relative to that cut's largest value, which an earlier
    version of this function incorrectly did. min_gap is the smallest gap
    >= atol seen anywhere, i.e. how close the nearest near-miss is.
    """
    max_multiplicity = 0
    min_gap = np.inf
    for s in spectra:
        s = np.sort(np.asarray(s))[::-1]
        if len(s) == 0:
            continue
        s = s[s > floor_rel * s[0]]
        if len(s) == 0:
            continue
        max_multiplicity = max(max_multiplicity, 1)
        run = 1
        for gap in -np.diff(s):
            if gap < atol:
                run += 1
                max_multiplicity = max(max_multiplicity, run)
            else:
                run = 1
                min_gap = min(min_gap, float(gap))
    return max_multiplicity, min_gap


def section4_live_training_diagnostic():
    print("\n" + "=" * 70)
    print("SECTION 4: live training run -- watching for the moment it NaNs")
    print("=" * 70)
    print(f"  jax devices: {jax.devices()}")
    print(f"  config: Lx={LIVE_LX} Ly={LIVE_LY} nlayers={LIVE_NLAYERS} "
          f"howoften_toreset={LIVE_HOWOFTEN_TORESET} reset_layers={LIVE_RESET_LAYERS} "
          f"bond_dim={LIVE_BOND_DIM} maxiter={LIVE_MAXITER} "
          f"normalize_state={LIVE_NORMALIZE_STATE}")
    print("  (first step pays a one-off cotengra contraction-path search + JIT "
          "compile -- can take ~10-20 minutes; steps after that are much "
          "faster. Consider running this inside a job script.)")
    print("  NOTE: if jax devices above shows only CpuDevice even though you "
          "requested a GPU, jaxlib's CUDA init silently failed on this node "
          "and everything ran on CPU -- check stderr.err for a "
          "'jax_plugins.xla_cuda12' error. That happened on crannog05 for "
          "job 3594457; it means that run was never actually the GPU-vs-CPU "
          "comparison it was meant to be.")

    import optax
    from src.diagnostics.mps_numerics import (
        format_magnitude_summary,
        magnitude_summary,
        singular_value_summary,
    )
    from src.utilities.ansatz_classes import ToricCodeAnsatz
    from src.utilities.generate_ansatz import get_singular_values_per_cut

    ansatz = ToricCodeAnsatz(
        Lx=LIVE_LX, Ly=LIVE_LY, nlayers=LIVE_NLAYERS,
        howoften_toreset=LIVE_HOWOFTEN_TORESET, reset_layers=LIVE_RESET_LAYERS,
        trials=LIVE_TRIALS, bond_dim=LIVE_BOND_DIM, use_optimal_ordering=LIVE_USE_OPTIMAL_ORDERING,
        normalize_state=LIVE_NORMALIZE_STATE,
        seed=LIVE_SEED,
    )
    params = jnp.array(ansatz.initparams)
    optimizer = optax.adam(learning_rate=ansatz.learning_rate)
    opt_state = optimizer.init(params)
    reset_slice = ansatz.reset_param_slice

    history = []  # rolling window of recent per-step summaries

    def _fmt_gap(g):
        return f"{g:.3e}" if g is not None and np.isfinite(g) else "n/a"

    def _print_step(r, extra=""):
        print(f"  step {r['step']:4d}: energy={r['energy']:.6f}  grad_norm={r['grad_norm']:.4e}  "
              f"energy_nan={r['energy_nan']}  grad_nan={r['grad_nan']}  nan_trials={r['nan_trials']}  "
              f"tensor_nan={r['tensor_nan']}  max_tied_multiplicity={r['max_multiplicity']}  "
              f"min_gap={_fmt_gap(r['min_gap'])}{extra}")
        if r["center_tensor"] is not None:
            print(f"    center tensor {r['center_position']}: "
                  f"{format_magnitude_summary(r['center_tensor'])}")
        if r["singular_values"] is not None:
            sv = r["singular_values"]
            print(f"    singular values: min_nonzero={sv['min_nonzero']:.3e}  "
                  f"max_condition={sv['max_condition']:.3e}  "
                  f"zeros={sv['zero_count']}  nonfinite={sv['nonfinite_count']}")

    def _dump_snapshot(reason, spectra=None):
        print(f"\n  *** {reason} -- dumping snapshot ***")
        print(f"  seed={ansatz.seed} (set LIVE_SEED = {ansatz.seed} at the top of this file "
              f"to replay this exact initialization deterministically)")
        print(f"  recent history ({len(history)} steps):")
        for r in history:
            _print_step(r)
        if spectra is not None:
            print(f"\n  full per-cut spectra:")
            for cut_idx, s in enumerate(spectra):
                print(f"    cut {cut_idx}: {np.array2string(np.asarray(s), precision=6)}")
        if reset_slice is not None:
            report_trial = history[-1]["nan_trials"][0] if history and history[-1]["nan_trials"] else 0
            thetas = np.asarray(params[report_trial])[reset_slice]
            print(f"\n  reset thetas (trial {report_trial}): {thetas}")
            print(f"  |theta - pi/2| per reset: {np.abs(thetas - np.pi / 2)}")

    for step in range(LIVE_MAXITER):
        value, grad = ansatz._cost_vvag(params)
        value_np = np.asarray(value)
        grad_np = np.asarray(grad)
        energy_nan = bool(np.isnan(value_np).any())
        grad_nan = bool(np.isnan(grad_np).any())
        grad_norm = float(np.linalg.norm(grad_np))

        # Which trial(s) -- out of LIVE_TRIALS independent starts sharing this one
        # LIVE_SEED -- actually went NaN. Defaults to trial 0 for the periodic
        # (non-NaN) diagnostic checks below; only meaningful once nan_trials is non-empty.
        nan_trials = np.where(np.isnan(value_np) | np.isnan(grad_np).any(axis=-1))[0]
        trial_idx = int(nan_trials[0]) if len(nan_trials) else 0

        record = dict(step=step, energy=float(np.real(value_np).mean()), grad_norm=grad_norm,
                      energy_nan=energy_nan, grad_nan=grad_nan, tensor_nan=None,
                      max_multiplicity=None, min_gap=None, center_position=None,
                      center_tensor=None, singular_values=None,
                      nan_trials=nan_trials.tolist())

        # The forward-only spectrum diagnostic (building qc + a per-cut numpy
        # SVD) is itself capable of failing outright -- job 3594457 hit
        # numpy.linalg.LinAlgError: SVD did not converge here, which killed
        # the whole process before this step's line could even print. That
        # failure is itself a useful signal (LAPACK's SVD choking usually
        # means the tensor it was fed is already NaN/Inf or wildly
        # ill-conditioned) but only if we survive it and report it.
        do_check = (step % LIVE_PRINT_EVERY == 0) or energy_nan or grad_nan
        spectra = None
        if do_check:
            try:
                qc = ansatz._circuit(params[trial_idx])
                if ansatz.normalize_state:
                    # Match energy_from_params(): inspect the same normalized
                    # MPS that is passed to the Hamiltonian expectations.
                    qc.normalize()
                tensors = [np.asarray(t) for t in qc.get_tensors()]
                record["tensor_nan"] = any(bool(np.isnan(t).any() or np.isinf(t).any()) for t in tensors)

                # Capture the actual orthogonality-centre tensor before
                # get_singular_values_per_cut() moves the centre across every
                # site. For a canonical MPS, this is also where the full state
                # norm is concentrated, so its scale is especially useful.
                center_position = qc.get_center_position()
                record["center_position"] = center_position
                if center_position is not None:
                    record["center_tensor"] = magnitude_summary(tensors[center_position])

                spectra = get_singular_values_per_cut(qc)
                record["max_multiplicity"], record["min_gap"] = _spectrum_degeneracy_summary(spectra)
                record["singular_values"] = singular_value_summary(spectra)
            except Exception as e:
                history.append(record)
                _print_step(record, extra=f"  [spectrum diagnostic raised {type(e).__name__}: {e}]")
                print("\n  the forward-only spectrum diagnostic itself failed -- a stronger "
                      "signal than a plain NaN gradient; usually means a raw MPS tensor is "
                      "already NaN/Inf or extremely ill-conditioned at this step.")
                try:
                    for i, t in enumerate(tensors):
                        print(f"    tensor {i}: shape={t.shape}  "
                              f"{format_magnitude_summary(magnitude_summary(t))}")
                except NameError:
                    print("    (circuit construction itself failed before tensors could be inspected)")
                _dump_snapshot(f"exception during spectrum diagnostic at step {step}")
                return

        history.append(record)
        if len(history) > LIVE_HISTORY_KEEP:
            history.pop(0)
        if do_check:
            _print_step(record)

        if energy_nan or grad_nan:
            _dump_snapshot(f"NaN detected at step {step}", spectra=spectra)
            return

        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)

    print(f"\n  no NaN encountered in {LIVE_MAXITER} steps")


# ---------------------------------------------------------------------------
# Section 5 -- static sanity check: does NaN appear at initial params as
# lattice size grows (quick, no 100-step wait)? Needs the real venv
# (tensorflow/qiskit/cotengra) -- not runnable in every sandbox.
# ---------------------------------------------------------------------------

def section5_real_ansatz_scaling_sweep():
    print("\n" + "=" * 70)
    print("SECTION 5: NaN onset vs lattice size (real ToricCodeAnsatz)")
    print("=" * 70)
    print("  (first call at each size pays a one-off cotengra contraction-path "
          "search + JIT compile -- can take a while)")

    from src.utilities.ansatz_classes import ToricCodeAnsatz
    from src.utilities.generate_ansatz import get_singular_values_per_cut

    for Lx, Ly in LATTICE_SIZES:
        for path_label, bond_dim in [("SVD path", SECTION5_BOND_DIM), ("QR path", None)]:
            ansatz = ToricCodeAnsatz(
                Lx=Lx, Ly=Ly, nlayers=1, reset_layers=[0], trials=1,
                bond_dim=bond_dim, use_optimal_ordering=False,
            )
            params = ansatz.initparams.at[:, ansatz.reset_param_slice].set(jnp.pi / 2.0)

            val, grad = ansatz._cost_vvag(params)
            has_nan = bool(np.isnan(np.asarray(grad)).any())

            qc = ansatz._circuit(params[0])
            spectra = get_singular_values_per_cut(qc)
            multiplicities = [int(np.isclose(s, s[0], atol=1e-8).sum()) for s in spectra if len(s) > 0]
            max_multiplicity = max(multiplicities, default=0)

            print(f"\nLx={Lx} Ly={Ly} {path_label} ({ansatz.lattice.num_qubits} system qubits, "
                  f"{ansatz.n_mps_qubits} MPS qubits): "
                  f"nan_in_grad={has_nan}  max_degenerate_multiplicity={max_multiplicity}")
            for cut_idx, s in enumerate(spectra):
                print(f"    cut {cut_idx}: {np.array2string(np.asarray(s), precision=4)}")


# ---------------------------------------------------------------------------
# Section 6 -- notes on follow-ups if Section 4 confirms NaN still happens
# despite Sections 1-3 showing the fix works in every controlled test here
# ---------------------------------------------------------------------------

SECTION_6_NOTES = """
SECTION 6: follow-ups if Section 4 still NaNs

1. CPU vs GPU consistency is now a direct toggle: set FORCE_CPU = True at the
   top of this file (must be set before jax is imported) and rerun Section 4.
   JAX's GPU SVD dispatches to cuSOLVER, a different implementation from CPU
   LAPACK, with its own known numerical-stability quirks for near-degenerate/
   singular matrices (mpscircuit.py even has a comment about a *different*
   backend's SVD implementation, Eigen::BDCSVD, failing outright in this
   exact "AD + MPS" context). If NaN only appears on GPU, the bug is likely
   in cuSOLVER's SVD forward pass itself (not the custom_vjp gradient rule
   Sections 1-3 exercised) -- worth a minimal cuSOLVER repro reported against
   tensorcircuit-ng / jax, since none of adaware_svd's regularization can fix
   a forward pass that already returns NaN/garbage U, V.

2. If confirmed GPU-specific: check whether upgrading jaxlib/CUDA changes
   anything (cuSOLVER SVD bugs for degenerate spectra have been fixed before
   in specific CUDA/cuDNN/jaxlib version combinations).

3. If Section 4's snapshot shows max_tied_multiplicity/min_gap were NOT
   unusual right before the NaN step, the lattice-wide-degeneracy hypothesis
   is wrong and it's worth looking elsewhere -- e.g. Adam's own moment
   accumulators overflowing after many steps of a merely-large (not NaN)
   gradient, or something outside the SVD path entirely (the Toffoli/CCX
   decomposition, the Cartan block's rxx/ryy/rzz parameterization at large
   angles, etc).
"""


def section6_print_notes():
    print("\n" + "=" * 70)
    print(SECTION_6_NOTES)


if __name__ == "__main__":
    if RUN_SECTION_1:
        section1_raw_svd_degeneracy()
    if RUN_SECTION_2:
        section2_adaware_svd_fix()
    if RUN_SECTION_3:
        section3_compounding_stress_test()
    if RUN_SECTION_4:
        section4_live_training_diagnostic()
    if RUN_SECTION_5:
        section5_real_ansatz_scaling_sweep()
    if RUN_SECTION_6_NOTES:
        section6_print_notes()
