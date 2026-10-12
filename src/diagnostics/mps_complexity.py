"""Rebuild a trained MPS outside the JAX path and measure how complex it really is.

Run the self-test (the gate for trusting anything else here):
    python -m src.diagnostics.mps_complexity --self-test --lattice 3x2

WHAT THIS IS FOR
----------------
bond_dim_requirement.py answers "what bond dimension does each reset arm need" by
walking a ladder of hard caps and reporting the first rung whose discarded weight
passes a gate. That is a rung, not a number, and it is only ever evaluated at two
or three points along a run. This module answers the same question differently:
truncate to a fixed *error tolerance* and read off the bond dimension the state
actually settles at. One build gives one integer, so it is cheap enough to run at
every snapshot of a training run.

WHY NOT UNDER JAX
-----------------
Error-tolerance truncation cannot run under jit at all: the number of singular
values kept is data-dependent, so the tensor shapes change from call to call.
That is exactly why make_split_conf only ever emits `max_singular_values` -- the
production forward pass has to have static shapes to be traced, vmapped and
differentiated.

Nothing here is traced, vmapped or differentiated, so none of that applies. We go
further and swap TensorCircuit onto its NumPy backend, which makes the rebuild
plain LAPACK on real numpy arrays: no XLA, no autodiff tape, and no exposure to
the GPU batched-Jacobi SVD failures that NaN training runs. It is also about 10x
faster than the same rebuild under JAX-eager, because there is no dispatch
overhead on thousands of small tensors.

THE TWO TRUNCATION PATHS (and why the shim below exists)
--------------------------------------------------------
MPSCircuit truncates in two different places, which disagree on the keyword name:

    two-qubit gates      apply_adjacent_double_gate -> FiniteMPS.apply_two_site_gate(**split)
                         names it  max_truncation_err
    Toffolis, MPO gates  apply_MPO -> mpscircuit.split_tensor -> backend.svd(**split)
                         names it  max_truncation_error

Only the purified arm applies Toffolis (generate_ansatz.py, the `toffoli_mode`
branch), which is why the trajectory arm accepts a tolerance today and the
purified arm dies with

    TypeError: _svd_jax() got an unexpected keyword argument 'max_truncation_err'

Both underlying implementations support the tolerance; they just spell it
differently. `numpy_mps_mode` wraps split_tensor to rename the key, so one split
dict drives both paths and both arms can be measured the same way.

A NOTE ON THRESHOLD TRUNCATION
------------------------------
make_split_conf's docstring declines to use a truncation threshold *for training*,
because the criterion bounds the norm of the discarded tail rather than each value
individually, and can therefore cut a run of small-but-similar singular values --
which the toric code's near-degenerate spectrum produces. That caution is about
differentiating through the result. Here we are measuring the state rather than
differentiating it, and the tolerance is the measurement. The defence against the
near-degeneracy problem is to sweep the tolerance: if the reported bond dimension
is not monotone in it, degeneracy is biting and the number has to be reported with
that caveat rather than quoted flat.
"""

import argparse
import contextlib
import io
import sys
import types

# TensorFlow is only needed by unrelated legacy constructors in generate_ansatz;
# stub it unconditionally -- some environments have a TF/sklearn build compiled
# against a different NumPy ABI than the installed NumPy, which crashes hard on
# import rather than raising cleanly. (Same reason as bond_dim_requirement.py.)
sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np
import tensorcircuit as tc

from src.utilities.ansatz_classes import ToricCodeAnsatz

NLAYERS = 2
RESET_LAYERS = [1]
SEED = 2026

# Modes this module knows how to rebuild. "none" (the reset-free control used by
# bond_dim_requirement) shares the trajectory path with zero reset events.
MODES = ("traj", "pur", "enum", "none")

# Ancilla-free arms stay at nq sites; purified spends 2 per reset event.
ANCILLA_FREE = ("traj", "enum", "none")


# ---------------------------------------------------------------------------
# Backend swap and truncation shim
# ---------------------------------------------------------------------------

def _renaming_split_tensor(original):
    """split_tensor that accepts the apply_two_site_gate spelling of the tolerance.

    split_tensor forwards **split straight to backend.svd, whose parameter is
    `max_truncation_error`. apply_two_site_gate takes `max_truncation_err` and
    translates it itself. Renaming here lets a single split dict, written in the
    apply_two_site_gate spelling, drive both paths.
    """
    def split_tensor(tensor, center_left=True, split=None):
        if split and "max_truncation_err" in split:
            split = dict(split)
            split["max_truncation_error"] = split.pop("max_truncation_err")
        return original(tensor, center_left=center_left, split=split)
    return split_tensor


@contextlib.contextmanager
def numpy_mps_mode(dtype="complex128"):
    """Run circuit construction on the NumPy backend, with tolerance truncation.

    Three things have to move together, and missing any one of them leaves a
    half-switched process that still silently produces JAX arrays:

      1. TensorCircuit's global backend, which is what MPSCircuit and the
         underlying FiniteMPS dispatch their linear algebra through.
      2. `generate_ansatz.K` and `find_gs.K`. Both modules do `K = tc.set_backend("jax")`
         at import time, binding the backend *object* into a module global. Later
         calls to tc.set_backend do not touch those bindings, so every `K.einsum`,
         `K.sin`, `K.real` in the circuit builders and in _energy_of_circuit would
         keep running on JAX.
      3. mpscircuit.split_tensor, per _renaming_split_tensor above.

    Everything is restored on exit, so a caller can measure and then go back to
    the normal JAX path in the same process -- which the self-test relies on.
    """
    from tensorcircuit import mpscircuit as mpsmod
    import src.utilities.generate_ansatz as ga
    import src.find_gs as fgs

    prev_backend_name = tc.backend.name
    # tc.dtypestr is the complex dtype ("complex128"); tc.rdtypestr is its real
    # counterpart. set_dtype takes the complex one, so that is what to save.
    prev_dtype = tc.dtypestr
    prev_ga_K, prev_fgs_K = ga.K, fgs.K
    prev_split_tensor = mpsmod.split_tensor

    numpy_backend = tc.set_backend("numpy")
    tc.set_dtype(dtype)
    ga.K = numpy_backend
    fgs.K = numpy_backend
    mpsmod.split_tensor = _renaming_split_tensor(prev_split_tensor)
    try:
        yield numpy_backend
    finally:
        mpsmod.split_tensor = prev_split_tensor
        ga.K = prev_ga_K
        fgs.K = prev_fgs_K
        tc.set_backend(prev_backend_name)
        tc.set_dtype(prev_dtype)


def error_split_conf(tol, cap, relative=True):
    """Split rules that truncate to a tolerance, under a safety cap.

    `cap` is NOT the measurement. It exists because a 20-site purified chain with
    no cap will happily chase its exact bond dimension of 1024, at which point a
    single snapshot costs minutes and hundreds of megabytes. Every measurement
    records whether the cap actually bound; if it did, that row is a ceiling and
    must not be quoted as a bond-dimension requirement.

    `relative=True` scales the tolerance by the largest singular value at each
    bond, so it means "discard this fraction of the local weight" rather than an
    absolute size -- the right reading when the state norm drifts under
    normalize_state=False.
    """
    if tol is None:
        return {"max_singular_values": cap}
    return {"max_singular_values": cap, "max_truncation_err": tol, "relative": relative}


# ---------------------------------------------------------------------------
# Building and rebuilding
# ---------------------------------------------------------------------------

def build_ansatz(Lx, Ly, mode, nlayers=NLAYERS, reset_layers=RESET_LAYERS,
                 seed=SEED, cap=512, use_optimal_ordering=True,
                 max_reset_branches=3 ** 8, quiet=True):
    """One ansatz per (lattice, mode). Truncation is re-pointed afterwards.

    The ordering search inside __post_init__ runs up to 2M random permutations,
    so it must not be repeated per snapshot. Build once, then call
    `set_split_conf` for each measurement -- the circuit builders read
    `self.split_conf` when they are called rather than capturing it at
    construction, which is what makes that safe. Same pattern as
    bond_dim_requirement.build / set_bond_dim.

    The search is seeded (`_rng.seed(42)` in _build_interleaved_ordering), so the
    chain ordering this reproduces is bit-for-bit the one the training run used.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    sink = io.StringIO() if quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        ansatz = ToricCodeAnsatz(
            Lx=Lx, Ly=Ly, nlayers=nlayers,
            # An empty list, not None: None means "resets on every layer", []
            # means no reset events at all, which is the `none` control.
            reset_layers=([] if mode == "none" else reset_layers),
            trials=1, seed=seed, bond_dim=cap,
            use_optimal_ordering=use_optimal_ordering,
            use_trajectory_resets=(mode in ("traj", "none")),
            use_enumerated_resets=(mode == "enum"),
            max_reset_branches=max_reset_branches,
            # The whole point: with normalisation on, the norm is forced back to
            # 1 after every layer and the discarded weight vanishes from view.
            normalize_state=False,
            sparse=False, use_mps=True,
        )
    ansatz._build_log = sink.getvalue() if quiet else ""
    return ansatz


def set_split_conf(ansatz, split_conf):
    """Re-point an already-built ansatz at different truncation rules."""
    ansatz.split_conf = split_conf


def rebuild(ansatz, params, mode, status=None, branch=None):
    """Build one circuit at these parameters. Returns (qc, log_w or None).

    Dispatches to the same entry points the training loop uses, so the state
    measured here is the state training actually built -- not a reimplementation
    that could drift from it.
    """
    params = np.asarray(params)
    if mode in ("traj", "none"):
        if status is None:
            status = np.zeros(ansatz.total_resets)
        qc, log_w = ansatz._circuit_traj(params, np.asarray(status))
        return qc, float(np.real(log_w))
    if mode == "enum":
        if branch is None:
            raise ValueError("mode='enum' needs an explicit branch index vector")
        return ansatz._circuit_branch(params, np.asarray(branch)), None
    return ansatz._circuit(params), None


# ---------------------------------------------------------------------------
# Complexity metrics
# ---------------------------------------------------------------------------

def bond_profile(qc):
    """Bond dimension at every internal cut, left to right.

    tn's `bond_dimensions` includes the trivial dimension-1 legs at both ends of
    the chain; those are not cuts and are dropped, so the result has one entry per
    cut and lines up with the per-cut Schmidt spectra.
    """
    dims = [int(d) for d in np.asarray(qc.get_bond_dimensions())]
    n_sites = len(qc.get_tensors())
    if len(dims) == n_sites + 1:
        dims = dims[1:-1]
    return dims


def mps_entries(qc):
    """Total complex entries stored, sum over sites of D_left * 2 * D_right.

    This is the metric that survives a comparison between arms. Max bond
    dimension does not: the trajectory chain is 12 sites at 3x3 and the purified
    chain is 20, so "the widest bond" is not the same question on each. Total
    entries is the actual memory footprint and is chain-length independent.
    """
    return int(sum(int(np.prod(np.asarray(t).shape)) for t in qc.get_tensors()))


def _entropies(spectrum):
    """Von Neumann and Renyi-2 entanglement entropy from one Schmidt spectrum.

    The spectrum comes off an unnormalised state (normalize_state=False), so it
    is renormalised here before being read as a probability distribution --
    otherwise truncation drift in the norm would show up as a fake entropy shift.
    """
    s = np.asarray(spectrum, dtype=float)
    p = s ** 2
    total = p.sum()
    if total <= 0:
        return 0.0, 0.0
    p = p / total
    p = p[p > 1e-300]                      # 0*log 0 = 0, and log(0) would warn
    vn = float(-(p * np.log(p)).sum())
    renyi2 = float(-np.log((p ** 2).sum()))
    return vn, renyi2


def complexity(ansatz, qc, cap, spectra=True):
    """Every per-snapshot metric, from one built circuit.

    Order matters. Bond dimensions, norm and energy are read BEFORE the spectrum
    walk, which moves the orthogonality centre across the whole chain. Those moves
    carry no truncation (MPSCircuit.position calls FiniteMPS.position with no
    split rules) so they do not change the bond dimensions, but reading first
    keeps that independent of a future change to `position`.
    """
    dims = bond_profile(qc)
    max_bd = max(dims) if dims else 1
    entries = mps_entries(qc)
    norm2 = float(np.real(np.asarray(qc.get_norm())) ** 2)
    energy = float(np.real(ansatz._energy_of_circuit(qc)))

    out = {
        "max_bond_dim": max_bd,
        "bond_profile": dims,
        "mps_entries": entries,
        # complex128 is 16 bytes; the production dtype throughout this repo.
        "mps_bytes": 16 * entries,
        "n_sites": len(qc.get_tensors()),
        # 1 - ||psi||^2 is the fraction of the state truncation threw away, and
        # with the state unnormalised the fractional error in the raw energy
        # equals it exactly.
        "discarded_weight": 1.0 - norm2,
        # MPSCircuit accumulates prod(1 - sum(err^2)) over every split it makes,
        # which is an independent readout of the same loss: it counts what each
        # truncation dropped, where discarded_weight measures what survived.
        "fidelity": float(np.real(qc._fidelity)),
        "energy": energy,
        "energy_normalised": energy / norm2 if norm2 > 1e-12 else float("nan"),
        # A row where the cap bound is a ceiling, not a measurement.
        "cap_bound": bool(max_bd >= cap),
    }

    if spectra:
        from src.utilities.generate_ansatz import get_singular_values_per_cut
        vns, r2s = [], []
        for s in get_singular_values_per_cut(qc):
            vn, r2 = _entropies(s)
            vns.append(vn)
            r2s.append(r2)
        out["entropy_vn"] = vns
        out["entropy_renyi2"] = r2s
        out["max_entropy_vn"] = max(vns) if vns else 0.0
        out["max_entropy_renyi2"] = max(r2s) if r2s else 0.0
    return out


def measure(ansatz, params, mode, tol, cap, status=None, branch=None, spectra=True):
    """Set the truncation, rebuild, and measure. The single-snapshot workhorse."""
    set_split_conf(ansatz, error_split_conf(tol, cap))
    qc, _ = rebuild(ansatz, params, mode, status=status, branch=branch)
    out = complexity(ansatz, qc, cap, spectra=spectra)
    out["tolerance"] = tol
    out["cap"] = cap
    out["mode"] = mode
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def tensor_modules(qc):
    """Which array libraries the MPS tensors actually come from.

    _sampled_reset writes its reset site with raw jnp.stack / jnp.where calls
    rather than going through K, so a JAX array can survive the backend swap and
    sit in an otherwise-NumPy chain. That is not wrong -- the values agree to
    machine precision either way -- but it means the rebuild would not be purely
    NumPy, and the self-test reports it rather than letting it pass silently.
    """
    return sorted({type(t).__module__.split(".")[0] for t in qc.get_tensors()})


def self_test(Lx, Ly, verbose=True):
    """Prove the NumPy rebuild reproduces the JAX path, then check known values.

    Everything downstream rests on this. It runs at a lattice small enough that
    both arms are exactly representable, so any disagreement is a bug in the
    rebuild rather than a truncation difference.
    """
    key = f"{Lx}x{Ly}"
    nq = 2 * Lx * Ly - Lx - Ly
    R = (Lx - 1) * (Ly - 1)
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
        if not ok:
            failures.append(name)

    print(f"\n{'=' * 78}\nself-test at {key}: nq={nq} R={R}\n{'=' * 78}")

    exact_cap = {"traj": 2 ** (nq // 2), "pur": 2 ** ((nq + 2 * R) // 2)}
    # Measured requirements from docs/three_reset_implementations.md section 2,
    # found there by ladder search. Tolerance truncation should land on the same
    # numbers in a single build.
    published = {("3x2", "traj"): 8, ("3x2", "pur"): 16}

    for mode in ("traj", "pur"):
        cap = min(exact_cap[mode], 512)
        ansatz = build_ansatz(Lx, Ly, mode, cap=cap)
        params = np.asarray(ansatz.initparams[0])
        status = np.random.default_rng(7).uniform(size=ansatz.total_resets)
        print(f"\n--- {key} {mode}: chain={ansatz.n_mps_qubits} exact cap={exact_cap[mode]}")

        # (1) JAX-eager reference at a hard cap, no tolerance.
        set_split_conf(ansatz, {"max_singular_values": cap})
        qc_jax, _ = rebuild(ansatz, params, mode, status=status)
        jax_dims = bond_profile(qc_jax)
        jax_energy = float(np.real(ansatz._energy_of_circuit(qc_jax)))

        # (2) The same thing on NumPy. Same params, same cap, same status.
        with numpy_mps_mode():
            set_split_conf(ansatz, {"max_singular_values": cap})
            qc_np, _ = rebuild(ansatz, params, mode, status=status)
            np_dims = bond_profile(qc_np)
            np_energy = float(np.real(ansatz._energy_of_circuit(qc_np)))
            modules = tensor_modules(qc_np)

            check(f"{mode}: bond profile matches JAX", np_dims == jax_dims,
                  f"numpy={np_dims} jax={jax_dims}")
            dE = abs(np_energy - jax_energy)
            check(f"{mode}: energy matches JAX to 1e-10", dE < 1e-10,
                  f"numpy={np_energy:.12f} jax={jax_energy:.12f} |dE|={dE:.2e}")
            # Informational, not a failure: see tensor_modules.
            print(f"  [ -- ] {mode}: tensor modules present: {modules}"
                  + ("" if modules == ["numpy"] else "   (jax arrays survived the swap)"))

            # (3) Tolerance truncation, swept. Monotonicity is the guard against
            # the near-degeneracy failure make_split_conf warns about.
            tols = [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
            bds = []
            for tol in tols:
                m = measure(ansatz, params, mode, tol, cap, status=status, spectra=False)
                bds.append(m["max_bond_dim"])
                if verbose:
                    print(f"         tol={tol:<8g} bd={m['max_bond_dim']:<5d} "
                          f"entries={m['mps_entries']:<8d} "
                          f"discarded={m['discarded_weight']:+.2e} "
                          f"fid={m['fidelity']:.12f} E={m['energy']:+.9f}")
            check(f"{mode}: bond dim monotone non-decreasing as tolerance tightens",
                  all(a <= b for a, b in zip(bds, bds[1:])), f"{dict(zip(tols, bds))}")

            if (key, mode) in published:
                want = published[(key, mode)]
                got = bds[tols.index(1e-8)]
                check(f"{mode}: tol=1e-8 reproduces the published requirement {want}",
                      got == want, f"got {got}")

            # (4) At a tolerance this tight the state is exact, so the energy
            # must match the untruncated one.
            m_tight = measure(ansatz, params, mode, 1e-12, cap, status=status, spectra=False)
            check(f"{mode}: tol=1e-12 energy matches untruncated",
                  abs(m_tight["energy"] - np_energy) < 1e-9,
                  f"|dE|={abs(m_tight['energy'] - np_energy):.2e}")

            # (5) Entropies are finite and bounded by log(bond dimension).
            m_full = measure(ansatz, params, mode, 1e-10, cap, status=status)
            bound = np.log(max(m_full["max_bond_dim"], 2))
            check(f"{mode}: max entropy finite and <= log(chi)",
                  np.isfinite(m_full["max_entropy_vn"])
                  and m_full["max_entropy_vn"] <= bound + 1e-9,
                  f"S={m_full['max_entropy_vn']:.4f} log(chi)={bound:.4f}")

    # (6) The enumerated arm's Kraus set is trace preserving, so the branch norms
    # sum to 1 without truncation. This validates the branch walk before
    # traj_approximation_quality uses it as the exact reference.
    import itertools
    ansatz = build_ansatz(Lx, Ly, "enum", cap=min(exact_cap["traj"], 512))
    params = np.asarray(ansatz.initparams[0])
    with numpy_mps_mode():
        set_split_conf(ansatz, error_split_conf(1e-12, min(exact_cap["traj"], 512)))
        total = 0.0
        for branch in itertools.product(range(3), repeat=ansatz.total_resets):
            qc, _ = rebuild(ansatz, params, "enum", branch=np.asarray(branch, dtype=int))
            total += float(np.real(np.asarray(qc.get_norm())) ** 2)
        check(f"enum: {3 ** ansatz.total_resets} branch norms sum to 1",
              abs(total - 1.0) < 1e-9, f"sum={total:.12f}")

    print(f"\n{'=' * 78}")
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s): {failures}")
    else:
        print("SELF-TEST PASSED")
    print("=" * 78)
    return not failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--lattice", default="3x2")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("nothing to do; pass --self-test (this module is a library)")
    Lx, Ly = (int(v) for v in args.lattice.split("x"))
    sys.exit(0 if self_test(Lx, Ly) else 1)


if __name__ == "__main__":
    main()
