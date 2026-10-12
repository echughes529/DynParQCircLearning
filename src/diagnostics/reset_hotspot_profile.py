"""Where does the purified arm's extra time and its NaNs actually come from?

Run:
    python -m src.diagnostics.reset_hotspot_profile --lattice 3x3 --modes traj,pur
    python -m src.diagnostics.reset_hotspot_profile --lattice 3x3 --params-from <checkpoint.npz>
    python -m src.diagnostics.reset_hotspot_profile --lattice 3x3 --natural-ordering

WHAT THIS ADDS OVER swap_counting.py
------------------------------------
`swap_counting.py` answers "how many SWAPs does each arm issue" (3x3, optimised
ordering: traj 64, pur 148). It does NOT answer "how much of the runtime is
that", because a SWAP on the purified 20-site chain and a SWAP on the
trajectory 12-site chain are not the same amount of arithmetic. Nor does it say
anything about numerics.

This module attributes both, from the SAME instrumented forward pass:

  (a) COST.  Every MPS bond split -- SVD or QR -- is tagged with the circuit
      construct that issued it (Cartan block / SWAP / Toffoli MPO /
      canonicalisation / normalisation), and for each we record the matrix
      shape and the measured wall time. Shape gives a hardware-independent
      work model (an m x n SVD is ~ m*n*min(m,n)); wall time is the empirical
      cross-check on this box. Non-split time -- the einsum contractions that
      apply the gate, plus Python -- is the residual of the total build time.

  (b) NUMERICS.  Every split's full singular-value spectrum is kept, so each
      split can be classified by the two DIFFERENT ways the SVD backward pass
      blows up. tensorcircuit's jaxsvd_bwd (backends/jax_ops.py) contains

          F_ij   = _safe_reciprocal(s_i^2 - s_j^2)      [off-diagonal]
          Sinv_i = _safe_reciprocal(s_i)                [diagonal]

      with `_safe_reciprocal(x) = x / (x*x + eps)`, eps = 1e-15. That is a
      Lorentzian, and it peaks at x = sqrt(eps) = 3.16e-8 with value
      1/(2*sqrt(eps)) = 1.58e7. Two distinct failure channels follow:

        DEGENERACY  -- two kept singular values with |s_i^2 - s_j^2| near
                       sqrt(eps). This is the flat-toric-code-spectrum case
                       the Lorentzian was put there for, and it is BOUNDED:
                       the broadening caps the amplification at 1.58e7.

        RANK DEFICIENCY -- a kept singular value s_i itself near sqrt(eps).
                       Two sub-cases that behave completely differently:
                       s_i EXACTLY 0 is harmless (0/(0+eps) = 0, and the
                       forward u,vh columns it multiplies are also zeroed);
                       s_i in the 1e-9..1e-6 band is the dangerous one -- it
                       sits ON the Lorentzian peak, and it enters BOTH F (via
                       s_i^2 - s_j^2 ~ -s_j^2) and Sinv.

      Both bands are reported separately, per arm, per issuing construct. The
      forward kernel's own rank sensitivity (project memory: batched-Jacobi
      NaNs one vmapped trial on rank-deficient matrices) is measured as
      `deficient` -- splits whose numerical rank is below the number of
      singular values the split is forced to keep, which is what a fixed
      `max_singular_values` guarantees will happen once the true Schmidt rank
      falls below the cap.

WHY EAGER, AND WHAT THAT COSTS
------------------------------
The build runs eagerly (no jit) because attribution needs one Python frame per
split. The SPLIT COUNTS and SHAPES are exact -- they are properties of the
traced program, identical to what the jitted run issues (swap_counting.py's
--demo-jit shows why). The measured SECONDS are eager seconds on this host and
include per-call dispatch, so treat the work model as primary and the timings
as a same-host relative check, not as a prediction of GPU wall time.
"""

import argparse
import contextlib
import sys
import time
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

import numpy as np

from src.diagnostics.mps_complexity import MODES, build_ansatz

# _safe_reciprocal's Lorentzian peaks here; see module docstring.
SVD_EPS = 1e-15
DANGER = np.sqrt(SVD_EPS)          # 3.16e-8
# Bands, in units of the largest singular value of the same split (the split is
# scale-free only up to the state norm, so absolute and relative are both kept).
EXACT_ZERO = 1e-14                  # below this, treated as a structural zero
DANGER_LO, DANGER_HI = 1e-11, 1e-5  # the window where the Lorentzian bites


# ---------------------------------------------------------------------------
# Tagging: which circuit construct issued this split
# ---------------------------------------------------------------------------

class Tagger:
    """A stack of tags maintained by wrappers around MPSCircuit's methods.

    Splits are attributed to the INNERMOST tag, which is what makes SWAPs come
    out as "swap" even though consecutive_swap is reached from inside
    apply_double_gate: consecutive_swap pushes, apply_adjacent_double_gate does
    not (it only pushes when nothing above it has, i.e. when a genuinely
    adjacent gate is applied directly).
    """

    def __init__(self):
        self.stack = []

    @contextlib.contextmanager
    def push(self, tag, only_if_empty=False):
        if only_if_empty and self.stack:
            yield
            return
        self.stack.append(tag)
        try:
            yield
        finally:
            self.stack.pop()

    def current(self):
        return self.stack[-1] if self.stack else "untagged"


TAGS = ("cartan", "swap", "toffoli_mpo", "reset_1q", "canonicalise", "normalise", "untagged")


def instrument():
    """Patch MPSCircuit + the tensornetwork Jax backend. Returns (records, restore).

    `records` is a list of dicts, one per bond split, in issue order.

    Patching classes rather than instances is forced: the circuits are created
    inside the ansatz builders, so there is no instance to reach in from here.
    Always restore in a finally -- leaving these wrappers installed would
    silently slow and miscount every later build in the process.
    """
    import jax
    from tensorcircuit.mpscircuit import MPSCircuit
    from tensornetwork.backends.jax.jax_backend import JaxBackend

    tagger = Tagger()
    records = []
    # Which bond the split is on. Set by the gate wrappers, read by the svd/qr
    # wrappers -- the backend never sees a site index, so it has to be carried
    # across. `site` is the LEFT site of the bond being split.
    site = {"cur": None}

    orig = {
        "cs": MPSCircuit.consecutive_swap,
        "adg": MPSCircuit.apply_adjacent_double_gate,
        "dg": MPSCircuit.apply_double_gate,
        "nq": MPSCircuit.apply_nqubit_gate,
        "sg": MPSCircuit.apply_single_gate,
        "pos": MPSCircuit.position,
        "svd": JaxBackend.svd,
        "qr": JaxBackend.qr,
    }

    def consecutive_swap(self, index_from, index_to, split=None):
        with tagger.push("swap"):
            return orig["cs"](self, index_from, index_to, split=split)

    def apply_adjacent_double_gate(self, gate, index1, index2,
                                   center_position=None, split=None):
        prev = site["cur"]
        site["cur"] = min(int(index1), int(index2))
        try:
            with tagger.push("cartan", only_if_empty=True):
                return orig["adg"](self, gate, index1, index2,
                                   center_position=center_position, split=split)
        finally:
            site["cur"] = prev

    def apply_double_gate(self, gate, index1, index2, split=None):
        with tagger.push("cartan", only_if_empty=True):
            return orig["dg"](self, gate, index1, index2, split=split)

    def apply_nqubit_gate(self, gate, *index, split=None):
        # Only the terminal (already-ascending) call is the real application;
        # apply_nqubit_gate re-enters itself once when indices are unsorted.
        with tagger.push("toffoli_mpo", only_if_empty=True):
            return orig["nq"](self, gate, *index, split=split)

    def apply_single_gate(self, gate, index):
        # Single-site gates need no split, but the reset's ry can force a
        # canonical-form move, and that move does split.
        with tagger.push("reset_1q", only_if_empty=True):
            return orig["sg"](self, gate, index)

    def position(self, site):
        with tagger.push("canonicalise", only_if_empty=True):
            return orig["pos"](self, site)

    def _record(kind, tensor, pivot_axis, max_sv, out, t0):
        left = int(np.prod(tensor.shape[:pivot_axis]))
        right = int(np.prod(tensor.shape[pivot_axis:]))
        rec = {
            "site": site["cur"],
            "kind": kind,
            "tag": tagger.current(),
            "m": left,
            "n": right,
            "k": min(left, right),
            "max_sv": max_sv,
            "seconds": time.perf_counter() - t0,
            "s": None,
        }
        records.append(rec)
        return rec

    def svd(self, tensor, pivot_axis=-1, max_singular_values=None,
            max_truncation_error=None, relative=False):
        t0 = time.perf_counter()
        out = orig["svd"](self, tensor, pivot_axis, max_singular_values,
                          max_truncation_error, relative=relative)
        jax.block_until_ready(out)
        rec = _record("svd", tensor, pivot_axis, max_singular_values, out, t0)
        # out = (u, s, vh, s_rest): keep the whole spectrum, kept and discarded,
        # since the backward pass only ever sees the kept block but the
        # discarded tail is what says whether the matrix was rank deficient.
        s_kept = np.abs(np.asarray(out[1]))
        s_rest = np.abs(np.asarray(out[3]))
        rec["s"] = s_kept
        rec["s_rest"] = s_rest
        return out

    def qr(self, tensor, pivot_axis=-1, non_negative_diagonal=False):
        t0 = time.perf_counter()
        out = orig["qr"](self, tensor, pivot_axis, non_negative_diagonal)
        jax.block_until_ready(out)
        _record("qr", tensor, pivot_axis, None, out, t0)
        return out

    MPSCircuit.consecutive_swap = consecutive_swap
    MPSCircuit.apply_adjacent_double_gate = apply_adjacent_double_gate
    MPSCircuit.apply_double_gate = apply_double_gate
    MPSCircuit.apply_nqubit_gate = apply_nqubit_gate
    MPSCircuit.apply_single_gate = apply_single_gate
    MPSCircuit.position = position
    JaxBackend.svd = svd
    JaxBackend.qr = qr

    def restore():
        MPSCircuit.consecutive_swap = orig["cs"]
        MPSCircuit.apply_adjacent_double_gate = orig["adg"]
        MPSCircuit.apply_double_gate = orig["dg"]
        MPSCircuit.apply_nqubit_gate = orig["nq"]
        MPSCircuit.apply_single_gate = orig["sg"]
        MPSCircuit.position = orig["pos"]
        JaxBackend.svd = orig["svd"]
        JaxBackend.qr = orig["qr"]

    return records, restore


@contextlib.contextmanager
def profiling():
    records, restore = instrument()
    try:
        yield records
    finally:
        restore()


# ---------------------------------------------------------------------------
# Running one forward pass
# ---------------------------------------------------------------------------

def forward(ansatz, params, mode, status=None):
    """Build one circuit, eagerly, through the same entry point training uses.

    `status` is the uniform randomness the trajectory sampler consumes. It is
    NOT free to leave at zero: _sampled_reset feeds `1 - u` to
    probability_sample, so u=0 deterministically selects the LAST branch --
    "reset fired, measured 1" -- at every single reset event, which is the one
    branch that projects a qubit and so understates the state's entanglement.
    The default here is the branch training actually spends almost all its time
    in (see three_reset_implementations.md 6.1: P(no reset fires) rises from
    0.10 to 0.999 over a run), and --traj-status overrides it.
    """
    params = np.asarray(params)
    if mode in ("traj", "none"):
        if status is None:
            # u = 1 - 1e-9 -> "1 - u" ~ 0 -> branch j=0, "no reset fired".
            status = np.full(ansatz.total_resets, 1.0 - 1e-9)
        return ansatz._circuit_traj(params, np.asarray(status))
    if mode == "enum":
        return ansatz._circuit_branch(params, np.zeros(ansatz.total_resets, dtype=int))
    return ansatz._circuit(params)


def profile_forward(ansatz, params, mode, warm=True):
    """Returns (records, total_seconds). One warm-up build first, uninstrumented."""
    if warm:
        forward(ansatz, params, mode)
    with profiling() as records:
        t0 = time.perf_counter()
        forward(ansatz, params, mode)
        total = time.perf_counter() - t0
    return records, total


# ---------------------------------------------------------------------------
# (a) cost attribution
# ---------------------------------------------------------------------------

def work_units(rec):
    """Hardware-independent arithmetic model for one split.

    Both SVD and QR of an m x n matrix are O(m*n*min(m,n)); the constants
    differ (a QR is roughly 4-8x cheaper per unit than a full SVD on the same
    shape), so the two kinds are reported separately and never summed.
    """
    return rec["m"] * rec["n"] * rec["k"]


def cost_table(records, total_seconds):
    """Per-tag split count, work units, measured seconds."""
    rows = {}
    for r in records:
        key = (r["tag"], r["kind"])
        row = rows.setdefault(key, {"count": 0, "work": 0, "seconds": 0.0,
                                    "shapes": {}})
        row["count"] += 1
        row["work"] += work_units(r)
        row["seconds"] += r["seconds"]
        row["shapes"][(r["m"], r["n"])] = row["shapes"].get((r["m"], r["n"]), 0) + 1
        row.setdefault("ks", []).append(r["k"])
    split_seconds = sum(r["seconds"] for r in records)
    return rows, split_seconds


# ---------------------------------------------------------------------------
# (b) numerics: which blow-up channel, if any, each split sits in
# ---------------------------------------------------------------------------

def numerics(rec):
    """Classify one split's kept spectrum. Returns a dict of scalars.

    `s` here is the KEPT block, which is exactly what jaxsvd_bwd differentiates
    through; `s_rest` is the discarded tail and only tells us about truncation.

    The central distinction this makes is between a gap |s_i^2 - s_j^2| that is
    small because two GENUINE singular values are nearly degenerate, and one
    that is small because both entries are structural zeros. Only the first is
    the flat-spectrum case the Lorentzian was put in for. The second is a
    rank-deficiency artefact of forcing the split to keep `max_singular_values`
    entries whatever the true rank is -- and it is BENIGN in the backward pass,
    because _safe_reciprocal(0) = 0 exactly. Lumping them together (which is
    what a naive gap histogram does) makes the purified arm look as if it were
    drowning in degeneracies when what it is really doing is padding.
    """
    s = rec["s"]
    if s is None or s.size == 0:
        return None
    smax = float(np.max(s))
    zero_cut = max(EXACT_ZERO, 1e-13 * smax)
    nz = s[s > zero_cut]
    smin_nonzero = float(nz.min()) if nz.size else 0.0
    n_exact_zero = int(s.size - nz.size)
    # ON-PEAK singular values: nonzero, but small enough that 1/s is Lorentzian
    # limited. These are the ones no regularisation can help.
    n_danger_s = int(np.sum((s > zero_cut) & (s < DANGER_HI)))

    # Gaps in s^2. Sorted order makes adjacent pairs the minimum, so the full
    # O(k^2) matrix is unnecessary. Restricted to pairs of NONZERO values.
    def gapstats(v):
        if v.size < 2:
            return 0.0, 0, 0.0
        g = np.abs(np.diff(v.astype(float) ** 2))
        pos = g[g > 0]
        mn = float(pos.min()) if pos.size else 0.0
        n_band = int(np.sum((g > 0) & (g < DANGER_HI)))
        amp = float(np.max(np.abs(g / (g * g + SVD_EPS)))) if g.size else 0.0
        return mn, n_band, amp

    min_gap_all, n_danger_gap_all, amp_gap_all = gapstats(s.astype(float))
    min_gap_nz, n_danger_gap_nz, amp_gap_nz = gapstats(nz.astype(float))

    sv = s.astype(float)
    amp_s_all = float(np.max(np.abs(sv / (sv * sv + SVD_EPS)))) if sv.size else 0.0
    nzv = nz.astype(float)
    amp_s_nz = float(np.max(np.abs(nzv / (nzv * nzv + SVD_EPS)))) if nzv.size else 0.0

    rank = int(nz.size)
    return {
        "smax": smax, "smin_nonzero": smin_nonzero,
        "n_kept": int(s.size), "rank": rank, "deficient": int(s.size) - rank,
        "n_exact_zero": n_exact_zero, "n_danger_s": n_danger_s,
        "min_gap_all": min_gap_all, "n_danger_gap_all": n_danger_gap_all,
        "min_gap_nz": min_gap_nz, "n_danger_gap_nz": n_danger_gap_nz,
        # The amplification the backward pass can actually reach, split into the
        # channel it comes through.
        "amp_gap_nz": amp_gap_nz, "amp_s_nz": amp_s_nz,
        "amp_all": max(amp_gap_all, amp_s_all),
        "discarded_weight": float(np.sum(rec.get("s_rest", np.zeros(0)) ** 2)),
    }


def numerics_table(records):
    """Aggregate the per-split classification, per tag."""
    rows = {}
    for r in records:
        if r["kind"] != "svd":
            continue
        d = numerics(r)
        if d is None:
            continue
        row = rows.setdefault(r["tag"], {
            "splits": 0, "kept": 0, "rank": 0, "deficient_splits": 0,
            "exact_zeros": 0, "danger_s_splits": 0,
            "danger_gap_nz_splits": 0, "danger_gap_all_splits": 0,
            "max_amp_nz": 0.0, "max_amp_all": 0.0,
            "min_smin": np.inf, "min_gap_nz": np.inf,
        })
        row["splits"] += 1
        row["kept"] += d["n_kept"]
        row["rank"] += d["rank"]
        row["deficient_splits"] += int(d["deficient"] > 0)
        row["exact_zeros"] += d["n_exact_zero"]
        row["danger_s_splits"] += int(d["n_danger_s"] > 0)
        row["danger_gap_nz_splits"] += int(d["n_danger_gap_nz"] > 0)
        row["danger_gap_all_splits"] += int(d["n_danger_gap_all"] > 0)
        row["max_amp_nz"] = max(row["max_amp_nz"],
                                max(d["amp_gap_nz"], d["amp_s_nz"]))
        row["max_amp_all"] = max(row["max_amp_all"], d["amp_all"])
        if d["smin_nonzero"] > 0:
            row["min_smin"] = min(row["min_smin"], d["smin_nonzero"])
        if d["min_gap_nz"] > 0:
            row["min_gap_nz"] = min(row["min_gap_nz"], d["min_gap_nz"])
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_cost(mode, ansatz, records, total_seconds):
    rows, split_seconds = cost_table(records, total_seconds)
    svd_work = sum(v["work"] for (t, k), v in rows.items() if k == "svd")
    print(f"\n  [{mode}]  chain={ansatz.n_mps_qubits}  "
          f"build={total_seconds * 1e3:.0f} ms  "
          f"in splits={split_seconds * 1e3:.0f} ms "
          f"({100 * split_seconds / total_seconds:.0f}%)  "
          f"elsewhere={1e3 * (total_seconds - split_seconds):.0f} ms")
    print(f"    {'construct':>13} {'kind':>4} {'splits':>7} {'work':>13} "
          f"{'work %':>7} {'seconds':>9} {'time %':>7} "
          f"{'min k':>6} {'med k':>6} {'max k':>6}")
    order = sorted(rows.items(), key=lambda kv: -kv[1]["work"])
    for (tag, kind), v in order:
        ks = np.array(v["ks"])
        wpct = 100 * v["work"] / svd_work if (kind == "svd" and svd_work) else float("nan")
        print(f"    {tag:>13} {kind:>4} {v['count']:>7} {v['work']:>13,} "
              f"{wpct:>6.1f}% {v['seconds']:>8.3f}s "
              f"{100 * v['seconds'] / split_seconds:>6.1f}% "
              f"{ks.min():>6} {int(np.median(ks)):>6} {ks.max():>6}")
    print(f"    (k = min(m, n) of the split matrix, i.e. how many singular "
          f"values the SVD computes)")
    return rows, svd_work, split_seconds


def print_numerics(mode, records):
    rows = numerics_table(records)
    print(f"\n  [{mode}]  singular-value health, per construct "
          f"(kept block only -- that is what jaxsvd_bwd differentiates)")
    print(f"    {'construct':>13} {'splits':>7} {'kept sv':>8} {'true rank':>10} "
          f"{'padded 0s':>10} {'rank-def splits':>16} "
          f"{'s on peak':>11} {'true degen':>11} {'min s>0':>10} "
          f"{'min gap>0':>11} {'max amp':>9}")
    tot = {"kept": 0, "rank": 0}
    for tag, v in sorted(rows.items(), key=lambda kv: -kv[1]["splits"]):
        tot["kept"] += v["kept"]
        tot["rank"] += v["rank"]
        print(f"    {tag:>13} {v['splits']:>7} {v['kept']:>8} {v['rank']:>10} "
              f"{v['exact_zeros']:>10} "
              f"{str(v['deficient_splits']) + '/' + str(v['splits']):>16} "
              f"{str(v['danger_s_splits']) + '/' + str(v['splits']):>11} "
              f"{str(v['danger_gap_nz_splits']) + '/' + str(v['splits']):>11} "
              f"{v['min_smin']:>10.2e} {v['min_gap_nz']:>11.2e} "
              f"{v['max_amp_nz']:>9.2e}")
    if tot["kept"]:
        print(f"    -> {tot['rank']}/{tot['kept']} = "
              f"{100 * tot['rank'] / tot['kept']:.0f}% of the singular values the "
              f"SVDs compute are nonzero; the rest are cap padding.")
    print(f"    'true degen'  = splits with two NONZERO singular values whose "
          f"|s_i^2-s_j^2| < {DANGER_HI:g}  (the Lorentzian's job)")
    print(f"    's on peak'   = splits with a nonzero kept s < {DANGER_HI:g}  "
          f"(the case the Lorentzian cannot fix)")
    return rows


def counterfactuals(mode, ansatz, records, cap):
    """Separate the two candidate levers, using only measured per-split shapes.

    The headline ratio (purified SVD work / trajectory SVD work) mixes two
    completely different causes, and the fix for one is not the fix for the
    other:

      LOCALITY  -- purified issues more SWAPs, because its Cartan pairs sit
                   further apart on a longer chain.
      GEOMETRY  -- every split purified does is on a bigger matrix, because a
                   20-site chain at cap chi has a PLATEAU of bonds sitting at
                   chi where a 12-site chain has a pyramid that only touches
                   chi once.

    Three counterfactuals separate them. Each reuses the recorded shapes, so
    none of them is a model of the circuit -- only a re-summation of measured
    numbers.

      no-SWAP     drop every SWAP-issued split. This is the BEST any ordering
                  could ever do (every Cartan pair already adjacent) and so
                  upper-bounds what locality work is worth.
      rank-only   replace min(m,n) by the split's true numerical rank. This is
                  the work that would remain if the bond dimension were not
                  padded past the state's actual Schmidt rank.
      both        the two together.
    """
    svd = [r for r in records if r["kind"] == "svd"]
    total = sum(work_units(r) for r in svd)
    no_swap = sum(work_units(r) for r in svd if r["tag"] != "swap")
    rank_only = 0
    for r in svd:
        d = numerics(r)
        rk = d["rank"] if d else r["k"]
        rank_only += r["m"] * r["n"] * rk
    both = 0
    for r in svd:
        if r["tag"] == "swap":
            continue
        d = numerics(r)
        rk = d["rank"] if d else r["k"]
        both += r["m"] * r["n"] * rk

    n = ansatz.n_mps_qubits
    at_cap = max(0, n - 2 * int(np.log2(cap)) - 1)
    return {"total": total, "no_swap": no_swap, "rank_only": rank_only,
            "both": both, "chain": n, "at_cap": at_cap}


def print_counterfactuals(cf):
    print(f"\n  Counterfactual SVD work (same measured shapes, re-summed)")
    print(f"    {'arm':>6} {'chain':>6} {'bonds at cap':>13} {'as built':>14} "
          f"{'no SWAPs':>14} {'no zero padding':>17} {'neither':>14}")
    for mode, d in cf.items():
        print(f"    {mode:>6} {d['chain']:>6} {d['at_cap']:>13} "
              f"{d['total']:>14,} {d['no_swap']:>14,} "
              f"{d['rank_only']:>14,} {d['both']:>14,}")
    if len(cf) == 2 and "traj" in cf and "pur" in cf:
        t, pu = cf["traj"], cf["pur"]
        print(f"\n    purified / trajectory, under each counterfactual:")
        for key, name in (("total", "as built"), ("no_swap", "if no SWAPs were needed"),
                          ("rank_only", "if bonds were not padded past rank"),
                          ("both", "if neither")):
            print(f"      {name:>36}: {pu[key] / t[key]:>6.1f}x")
        print(f"\n    what each lever removes from purified's own bill:")
        print(f"      {'all SWAPs':>36}: "
              f"{100 * (1 - pu['no_swap'] / pu['total']):>5.1f}%")
        print(f"      {'all zero padding':>36}: "
              f"{100 * (1 - pu['rank_only'] / pu['total']):>5.1f}%")
        print(f"      {'both':>36}: "
              f"{100 * (1 - pu['both'] / pu['total']):>5.1f}%")
        print(f"    'bonds at cap' = max(0, n - 2*log2(chi) - 1): how many bonds "
              f"of an n-site\n    chain the truncation actually pins at chi, "
              f"rather than the light cone pinning them lower.")


def totals(records):
    svd = [r for r in records if r["kind"] == "svd"]
    return {
        "splits": len(svd),
        "qr": len(records) - len(svd),
        "work": sum(work_units(r) for r in svd),
        "seconds": sum(r["seconds"] for r in records),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lattice", default="3x3")
    p.add_argument("--modes", default="traj,pur")
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--reset-layers", default="[1]")
    p.add_argument("--bond-dim", type=int, default=64)
    p.add_argument("--natural-ordering", action="store_true")
    p.add_argument("--toffoli-mode", default="direct", choices=("direct", "decomposed"))
    p.add_argument("--params-from", default=None,
                   help="checkpoint.npz to read all_params from, instead of using "
                        "the random initialisation")
    p.add_argument("--param-step", type=int, default=-1,
                   help="which stored snapshot to use from --params-from")
    p.add_argument("--param-trial", type=int, default=0)
    args = p.parse_args()

    import ast
    reset_layers = ast.literal_eval(args.reset_layers)
    Lx, Ly = (int(v) for v in args.lattice.split("x"))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        p.error(f"unknown mode(s) {bad}; choose from {MODES}")

    external = None
    if args.params_from:
        z = np.load(args.params_from)
        allp = z["all_params"]
        if not np.any(allp):
            raise SystemExit(f"{args.params_from}: all_params is all zeros "
                             "(the source run needs TRACK_PARAMS=true)")
        external = np.asarray(allp[args.param_step][args.param_trial])
        print(f"params from {args.params_from} "
              f"step={args.param_step} trial={args.param_trial}")

    print(f"{args.lattice}  nlayers={args.nlayers}  reset_layers={reset_layers}  "
          f"bond_dim={args.bond_dim}  toffoli={args.toffoli_mode}  "
          f"ordering={'natural' if args.natural_ordering else 'optimised'}")
    print(f"Lorentzian: eps={SVD_EPS:g}, peak at sqrt(eps)={DANGER:.2e}, "
          f"max amplification 1/(2 sqrt(eps))={1 / (2 * DANGER):.2e}")

    summary = {}
    cfs = {}
    for mode in modes:
        ansatz = build_ansatz(Lx, Ly, mode, nlayers=args.nlayers,
                              reset_layers=reset_layers, cap=args.bond_dim,
                              use_optimal_ordering=not args.natural_ordering)
        ansatz.toffoli_mode = args.toffoli_mode
        params = external if external is not None else np.asarray(ansatz.initparams[0])
        records, total = profile_forward(ansatz, params, mode)
        qc = forward(ansatz, params, mode)
        if isinstance(qc, tuple):
            qc = qc[0]
        bonds = [int(t.shape[2]) for t in qc._mps.tensors[:-1]]
        print(f"\n  [{mode}]  final bond profile: {bonds}   max={max(bonds)}")
        print_cost(mode, ansatz, records, total)
        print_numerics(mode, records)
        cfs[mode] = counterfactuals(mode, ansatz, records, args.bond_dim)
        summary[mode] = (ansatz, totals(records), total)

    print_counterfactuals(cfs)

    if len(summary) >= 2 and "traj" in summary and "pur" in summary:
        (_, t_tot, t_wall), (_, p_tot, p_wall) = summary["traj"], summary["pur"]
        print(f"\n  purified / trajectory:  splits {p_tot['splits'] / t_tot['splits']:.2f}x   "
              f"SVD work {p_tot['work'] / t_tot['work']:.2f}x   "
              f"split seconds {p_tot['seconds'] / t_tot['seconds']:.2f}x   "
              f"whole build {p_wall / t_wall:.2f}x")


if __name__ == "__main__":
    main()
