# Why the toric-code ground-state search produces garbage gradients (and NaNs)

**Investigation of the NaN energies observed after ~20–100 Adam steps on the 3×3 toric code.**

Everything below was measured, not reasoned about. The scripts that produce every
number are in `tests/nan_investigation/`; each claim names the script that proves it.

---

## 0. Verdict in one paragraph

Your instinct was right, and it is worse than you thought. `generate_ansatz.py` puts
`split={"max_singular_values": 2, "fixed_choice": 1}` on almost every circuit, which makes
tensorcircuit run a **differentiated SVD on every two-qubit gate**. tensorcircuit's SVD
backward pass regularises the `1/(s_i² − s_j²)` degeneracy factor with
`_safe_reciprocal(x) = x/(x² + 1e-15)`. That regulariser has a **maximum of 1.58 × 10⁷ at
x = √(1e-15) = 3.16 × 10⁻⁸**. Your whole simulation runs in **complex64**, where the
floating-point residual of `s_i² − s_j²` at a degeneracy is ~5 × 10⁻⁸ — i.e. it lands
essentially *exactly on the peak of the regulariser*. So instead of damping the
singularity, the regulariser maximally amplifies float32 rounding noise. Measured
consequence: a gradient component of **2.32 × 10⁷ where the true value is 0**. The
degeneracies sit at every multiple of π/2 in the `rxx`/`ryy`/`rzz` angles — which is
precisely where a *stabilizer* target state like the toric-code ground state drags the
optimiser. In a real 800-step run, **58 steps out of 800 had gradient errors above 1 %,
with a worst case 13 000× larger than the true gradient**, and every single one coincided
with a two-qubit angle sitting within ~10⁻⁶ of a multiple of π/2. The same run in
complex128: **0 out of 800**. With every two-qubit angle parked on a Clifford value the
gradient reaches **10⁸** (true value ~0.8), and at θ exactly 0 it is a hard **NaN**.

The split buys you nothing: it makes the gradient evaluation **1.6× slower** and the XLA
compile **much** longer (§7).

---

## 1. Your configuration, established by measurement

### 1.1 The whole simulation runs in single precision

`src/find_gs.py:21` and `src/utilities/generate_ansatz.py:12` both call
`config.update("jax_enable_x64", True)`. That controls **JAX**, not **tensorcircuit**.
tensorcircuit keeps its own dtype, set at import time in `tensorcircuit/cons.py:244`
(`set_dtype()` with no argument → `"complex64"`, which also *disables* x64 —
`cons.py:225` — hence the order dependence you may have noticed).

Nothing in the repo ever calls `tc.set_dtype("complex128")`. Measured (`t02_dtype.py`):

```
tc.dtypestr  = complex64
tc.rdtypestr = float32
gate tensor dtype : complex64
statevector dtype : complex64
a plain jnp float : float64     <- x64 IS on, tensorcircuit just doesn't use it
```

So: **parameters and the Adam state are float64, but every circuit tensor, the SVD, and
the energy are float32.** This single fact is what turns the SVD issue from a curiosity
into a training-killer.

### 1.2 Which ansätze go through the splitter

`split_conf` (`generate_ansatz.py:410`) is passed to `tc.Circuit(..., split=split_conf)` in:

| builder | line | split? |
|---|---|---|
| `construct_dyn_circuit_toriccodelattice` (the default toric-code ansatz) | 162 | **yes** |
| `construct_dyn_circuit_toriccodelattice_prob_resets` | 268 | **yes** |
| `construct_smallangle_init_toriccodelattice` | 353 | **yes** |
| `construct_dyn_circuit_brickwork` / `construct_unitary_circuit_brickwork` | 101 / 136 | **yes** |
| `construct_unitary_circuit_toriccodelattice` | 209 | **no** |

So `ToricCodeAnsatz(unitary=True)` is the *only* toric-code path that is free of this bug.
If your NaNs came from `unitary=True`, this report is not your cause — see §8.

The split survives `qc.append(...)`, which is how `onesetofunitaries` applies every
Cartan block: `append` rebuilds the circuit through `from_qir(..., self.circuit_param)`
and `circuit_param` carries `split`. Measured (`t05_split_propagation.py`): after
appending one Cartan block you get **6 rank-3 nodes and 0 rank-4 nodes** — i.e. all three
two-qubit rotations were SVD-split.

### 1.3 The exact code path

```
generate_ansatz.cartanblock  ->  qc.append(...)
  -> tensorcircuit/basecircuit.py:231   apply_general_gate, noe == 2
  -> tensorcircuit/simplify.py:98       _split_two_qubit_gate -> tn.split_node(max_singular_values=2)
  -> tensornetwork/network_operations.py:243   u,s,vh = backend.svd(...);  sqrt_s = backend.sqrt(s)
  -> tensorcircuit/backends/jax_backend.py:76  _svd_jax -> adaware_svd
  -> tensorcircuit/backends/jax_ops.py:33      jaxsvd_bwd   <-- the problem lives here
```

`jax_ops.py:24`:

```python
def _safe_reciprocal(x, epsilon: float = 1e-15):
    return x / (x * x + epsilon)
```

`jax_ops.py:42`:

```python
F = s * s - (s * s)[:, None]
F = _safe_reciprocal(F) - jnp.diag(jnp.diag(_safe_reciprocal(F)))
```

---

## 2. The forward pass is fine — the split is *exact* for your gates

This matters, because it means the bug is purely a gradient bug and the energies you
*read off* are trustworthy right up until the optimiser is poisoned.

Operator-Schmidt spectra in the exact reshaping tensorcircuit uses
(`t01_schmidt_spectra.py`):

| gate | singular values |
|---|---|
| `rxx(θ)`, `ryy(θ)`, `rzz(θ)` | `[2·abs(cos(θ/2)), 2·abs(sin(θ/2)), 0, 0]` |
| `cnot`, `cz` | `[√2, √2, 0, 0]` |
| `swap`, `iswap` | `[1, 1, 1, 1]` |

Every gate the toric-code ansätze actually use has operator-Schmidt rank ≤ 2, so keeping
2 singular values is lossless. Verified (`t26_misc_checks.py`):

```
rxx(1.1)   max|U_split - U_exact| = 5.97e-08
rzz(1.1)   max|U_split - U_exact| = 1.69e-07
cnot       max|U_split - U_exact| = 5.96e-08
cz         max|U_split - U_exact| = 5.96e-08
cry(0.7)   max|U_split - U_exact| = 4.77e-07
swap       max|U_split - U_exact| = 1.00e+00   <<< LOSSY
iswap      max|U_split - U_exact| = 1.00e+00   <<< LOSSY
```

and the state norm stays 1 (`t06_fast_equivalence.py`: `0.9999995`, float32 roundoff).

> **Landmine for later:** `max_singular_values=2` silently discards half of any
> two-qubit gate of Schmidt rank > 2. Add a `swap`, an `iswap`, or a generic
> `c.unitary(...)` anywhere in these circuits and you get *silently wrong physics* with no
> warning — measured error 1.0 on the unitary, i.e. a completely different gate.

---

## 3. Where the degeneracies are — and why your problem walks straight into them

For `rxx`/`ryy`/`rzz` the two kept singular values are `2|cos(θ/2)|` and `2|sin(θ/2)|`.
Two things can go degenerate:

* **D1 — the two *kept* values collide:** `s₀ = s₁` ⟺ `cos θ = 0` ⟺ **θ ≡ π/2 (mod π)**.
* **D2 — the smaller kept value collides with the *discarded* zeros:** `s₁ = 0` ⟺
  `sin θ = 0` ⟺ **θ ≡ 0 (mod π)**.

So **every multiple of π/2 is a degeneracy.** `cnot` and `cz` sit permanently on D1
(`[√2, √2]`), but they are constant gates so their cotangent is discarded — measured, they
are harmless (`t13_nan_hunt.py`: the `cnot`/`cnot` reset gadget gives 0 non-finite values
and worst gradient discrepancy 1.3e-7).

**The problem is that multiples of π/2 are exactly the Clifford angles.** The toric-code
ground state is a stabilizer state; the circuit that prepares it is Clifford; so the
optimiser is actively driving your `rxx`/`ryy`/`rzz` angles *onto* the degeneracies. That
is the mechanism behind "it works for 20–100 steps and then dies" — it takes that many
Adam steps at `lr=1e-2` to get there.

---

## 4. The blow-up, dissected

`t24_epsilon_sweep.py` instruments tensorcircuit's own backward pass, under jit, in
complex64, at θ = π/2:

```
[bwd] s = [1.4142135  1.4142135  0.  0.]        <- the two kept values are BITWISE EQUAL
      F0 = s*s - (s*s)[:,None] = 5.0752206e-08  <- but this is NOT zero in float32/XLA
      max|F| = 14193299.0                       <- _safe_reciprocal peak is 1.58e7
      max|Sinv| = 0.707
      |dAs| = 0.32   |dAu| = 2.32e7   |dAv| = 2.32e7   |dAc| = 0.25
=> dE/dtheta = 2.316601e+07        (the true value is ~0)
```

Read the third line carefully: **that is the entire bug.** `_safe_reciprocal` is a
Lorentzian whose peak sits at `x = √ε = 3.16e-8` with height `1/(2√ε) = 1.58e7`. In
float32 the residual of `s² − s'²` at a degeneracy is ~5e-8. The regulariser designed to
*cap* the singularity instead sits you on its maximum.

That this is the knob is proved by sweeping ε (`t24_epsilon_sweep.py`; note you must call
`jax.clear_caches()` and rebuild `adaware_svd_jit`, or the patch never reaches the
compiled code):

| ε | peak `1/(2√ε)` | gradient error at θ = π/2 |
|---|---|---|
| **1e-15 (tensorcircuit's value)** | 1.58e7 | **2.317e+07** |
| 1e-12 | 5.00e5 | 8.262e+04 |
| 1e-9 | 1.58e4 | 8.263e+01 |
| 1e-7 | 1.58e3 | 6.243e-01 |
| 1e-6 | 5.00e2 | 1.212e-01 |
| 1e-5 | 1.58e2 | 1.957e-01 |

The error tracks the peak height exactly until it saturates at the O(0.2) "true"
discretisation error. In float64 the residual is ~1e-16 ≪ √ε, so
`_safe_reciprocal(1e-16) ≈ 0.1` and everything is damped. **ε = 1e-15 is correctly tuned
for float64 and pathologically mistuned for float32.**

### 4.1 Gradient error vs distance from the degeneracy

`t17_rzz_degeneracy_scan.py`. Three qubits, a `cnot`, then `rzz(θ)`, observable `Z₀Z₁`.
The reference is the same circuit with the splitter off (mathematically identical, since
`rzz` has Schmidt rank 2). The true gradient goes to 0 linearly as θ → π/2.

| θ − π/2 | error, **complex64** (repo) | error, complex128 |
|---:|---:|---:|
| 1e-2 | 6.0e-08 | 1.9e-16 |
| 1e-3 | 6.0e-08 | 1.1e-16 |
| 1e-4 | **8.2e-05** | 1.4e-16 |
| 1e-5 | **3.4e-02** | 5.6e-17 |
| 1e-6 | **7.9e-03** | 9.4e-16 |
| 1e-7 | **2.3e+07** | 1.9e-12 |
| 1e-8 | **2.3e+07** | 2.5e-09 |
| 1e-10 | **2.3e+07** | 4.0e-07 |
| 0 (exact) | **2.3e+07** | 3.6e-01 |

**This is precisely the picture you described**: not infinite at exact degeneracy, but
absolutely enormous — and the "almost degenerate" region is *wider*, not narrower, than
the exact one. float64 shrinks the dangerous window by ~6 orders of magnitude and caps the
worst case at O(1) instead of 10⁷.

`jit` makes it markedly worse — XLA's reassociation changes which side of the peak you
land on (`t19_jit_effect.py`):

| θ − π/2 | grad, no jit | grad, **with jit** | truth |
|---:|---:|---:|---:|
| 1e-4 | −2.134e-05 | −1.036e-04 | −2.133e-05 |
| 1e-5 | −5.159e-05 | +3.440e-02 | −2.133e-06 |
| 1e-7 | −2.040e-01 | **+2.317e+07** | −2.133e-08 |
| 0 | −2.040e-01 | **+2.317e+07** | ~0 |

### 4.2 The other degeneracy, and one genuine NaN

`t22_degeneracies_and_fixes.py`, complex64:

* **θ → π** (D2, the `Sinv = _safe_reciprocal(s)` route): error **8.6e-02** for
  |θ−π| ≤ 1e-7 — a 40 % relative error, permanently, in a whole neighbourhood.
* **θ = 0 exactly**: **hard NaN**. `tn.split_node` computes `sqrt_s = backend.sqrt(s)`
  and `s₁ = 2|sin(θ/2)|` is exactly 0 there; the VJP of `sqrt` at 0 is `0.5/√0 = inf`.
  In the *real* toric-code circuit with every two-qubit angle set to 0
  (`t25_many_degeneracies.py`): `E = 0.181426` (finite, forward is fine) but
  **48 of 165 gradient components are NaN**. Scanned 400 161 angles
  (`t20_zero_singular_values.py`): `s₁ == 0` happens **only** at θ = 0.0 exactly, in both
  precisions — so this is a real but improbable route under Adam, which will not land on
  exactly 0.0.
* **`max_singular_values=4` ("just don't truncate") is much worse, not better**: it keeps
  the two zero singular values and differentiates `sqrt(0)`, giving **NaN almost
  everywhere**. Do not "fix" it this way.

---

## 5. What this does to a real training run

`t21_train_track.py`. Real toric-code dynamic ansatz (3×2, 11 qubits, 165 params,
48 two-qubit angles per trial), 12 trials, 800 Adam steps at `lr=1e-2`, `h=0.1`, repo
settings. Every step the gradient is computed **both** ways — through the splitter and
with the splitter off — and compared.

```
DONE mode=c64 seed=0: steps with rel-error>1e-2:  58/800; worst rel error 1.322e+04;
                      first non-finite step None; final Emin -7.310377
DONE mode=c64 seed=1: steps with rel-error>1e-2: 126/800; worst rel error 1.075e+03;
                      first non-finite step None; final Emin -7.310410
```

Worst offenders:

```
step 470  max|g|=3.218e+03  | split-vs-ref: abs 3.218e+03  rel 1.322e+04 | closest 2q angle to k*pi/2: 3.49e-07
step 332  max|g|=1.200e+02  | split-vs-ref: abs 1.200e+02  rel 1.185e+03 | closest 2q angle to k*pi/2: 1.89e-06
step 623  max|g|=4.002e+01  | split-vs-ref: abs 4.002e+01  rel 1.366e+02 | closest 2q angle to k*pi/2: 1.54e-07
step 377  max|g|=1.254e+01  | split-vs-ref: abs 1.254e+01  rel 3.571e+01 | closest 2q angle to k*pi/2: 1.40e-06
```

The correlation with proximity to a Clifford angle is monotone and total:

| closest two-qubit angle to a multiple of π/2 | steps | median gradient error | max gradient error |
|---|---:|---:|---:|
| [0, 1e-7) | 21 | 1.38e-02 | 6.74e+00 |
| [1e-7, 1e-6) | 35 | 9.85e-03 | **3.22e+03** |
| [1e-6, 1e-5) | 18 | 4.79e-03 | 1.20e+02 |
| [1e-5, 1e-4) | 13 | 1.67e-05 | 3.62e-02 |
| [1e-4, 1e-3) | 2 | 3.09e-05 | 5.82e-05 |
| ≥ 1e-3 | 4 | 3.58e-06 | 4.77e-06 |

Note also that the corrupted steps become *dense* once the run converges (Emin plateaus at
−7.3103 against an exact −7.310437) — which is exactly the "it starts fine and degrades"
signature, because convergence means angles piling onto Clifford values.

### 5.1 The same run in complex128

Identical script, identical seed, one line changed (`MODE=c128`):

```
DONE mode=c128 seed=0: steps with rel-error>1e-2: 0/800; worst rel error 3.320e-08;
                       first non-finite step None; final Emin -7.310424
```

**Zero corrupted steps out of 800**, worst relative error 3.3e-08 (against 1.3e+04), and it
also lands closer to the exact ground state (−7.310424 vs −7.310377, exact −7.310437).

### 5.2 How big does it get when many angles are Clifford at once?

`t25_many_degeneracies.py`, real toric-code circuit (3×2, 9 qubits, 48 two-qubit angles),
complex64, setting the first *k* two-qubit angles to exactly π/2:

| # angles at π/2 | E (split) | E (reference) | max\|g\| split | max\|g\| true | # components > 1e3 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.210642 | 0.210643 | 5.14e-01 | 5.14e-01 | 0 |
| 1 (`rxx`) | 0.230269 | 0.230270 | 5.18e-01 | 5.18e-01 | 0 |
| 2 (`+ryy`) | 0.095572 | 0.095572 | 5.18e-01 | 5.18e-01 | 0 |
| 4 (`+rzz`) | 0.457283 | 0.457284 | **3.23e+07** | 5.05e-01 | 1 |
| 8 | 0.631377 | 0.631379 | **4.94e+07** | 5.75e-01 | 2 |
| 16 | 0.642329 | 0.642331 | **4.00e+07** | 8.35e-01 | 5 |
| 24 | −0.180541 | −0.180541 | **5.75e+07** | 8.61e-01 | 8 |
| 32 | −0.451496 | −0.451498 | **8.35e+07** | 9.07e-01 | 10 |
| 48 (all) | −1.643542 | −1.643550 | **1.03e+08** | 8.26e-01 | 16 |

Three things to read off this:

1. The **energies are correct throughout** (agreement to ~1e-6) — the damage is purely in
   the gradient.
2. The blow-up is triggered by the **`rzz`** angles specifically (`k=1,2` are the `rxx` and
   `ryy` of the first Cartan block and are clean; `k=4` is the first one that includes an
   `rzz`). The number of corrupted components tracks the number of `rzz` gates sitting on
   the degeneracy.
3. The magnitude grows **additively, not multiplicatively**, saturating around 10⁸ — each
   gate's blown-up cotangent terminates at its own gate matrix and does not propagate
   through the network. That bounds how bad a single step can be, and is why §6 says what
   it says.

---

## 6. What I could **not** prove

**I did not reproduce a literal NaN energy in a full training run.** 800 steps × 12 trials
at 11 qubits gave 58 catastrophically corrupted steps but zero non-finite values. Being
precise about why: `optax.adam` normalises by `√v̂`, so even a 10⁸ gradient produces a
step of order `lr`; to get NaN out of Adam in float64 you need `|g| ≳ 1e154` so that `g²`
overflows. §5.2 shows the spikes saturate around 10⁸ and grow additively rather than
multiplicatively, so a single step cannot get you there.

So the honest statement is: **this mechanism definitely destroys your gradients, and it is
overwhelmingly the most likely cause of your training pathology, but I have proven
gradient corruption up to 10⁴–10⁸×, plus a hard NaN whenever a two-qubit angle is exactly
0 — not the final step from "10⁸" to "NaN energy" under Adam.** Your 3×3 run has ~12×
more two-qubit angles per trial (576 vs 48) and a larger Hamiltonian norm, so both the
frequency and the magnitude of the spikes will be larger there. §9 tells you how to close
that last gap on your own machine.

Other candidates that I could not test here and that you should rule out (§9):

1. **GPU SVD.** Your `environment.yml` is a CUDA build. `jnp.linalg.svd` on GPU goes
   through cuSOLVER, and batched float32 SVD there has different (and less benign) failure
   modes than CPU LAPACK, including returning NaN on non-convergence. Under
   `K.vmap(..., vectorized_argnums=0)` you are running a *batched* SVD of nearly-degenerate
   4×4 float32 matrices thousands of times per step. This is the single most likely place
   for the last step from "10⁷" to "NaN".
2. **Version skew.** I tested `tensorcircuit-ng 1.9.1` + `jax 0.10.2`; you pin
   `tensorcircuit-nightly 1.5.0.dev20260313` / `tensorcircuit-ng>=1.1.0` + `jax 0.4.37`.
   `jax_ops.py` and `simplify.py` have been stable across these, but confirm with test T1
   in §9.
3. **No gradient clipping.** `find_gs.py:254` calls `optimizer.update(gradient, opt_state)`
   with no guard at all. Whatever the source, one inf reaching Adam poisons `v` and every
   subsequent parameter forever.

---

## 7. The split is not even buying you speed

`t23_epsilon_and_perf.py`, toric code 3×2, 9 qubits, complex64:

| | TN nodes | jit compile | per gradient eval |
|---|---:|---:|---:|
| `split={"max_singular_values": 2}` (repo) | 222 | 239.6 s | **0.55 ms** |
| no split | 174 | 262.8 s | **0.35 ms** |

Splitting makes the gradient **1.6× slower**. For a statevector simulation the cost is
dominated by the 2ⁿ state, and turning one rank-4 tensor into two rank-3 tensors just adds
a contraction step plus an SVD. There is no reason to pay for it here.

(Separately, `qc.append` in `onesetofunitaries` rebuilds the *entire* circuit from its QIR
on every call, so tracing is O(M²) in the number of Cartan blocks. For 3×3 with
`nlayers=12` that is 192 appends → ~55 000 traced SVD ops that XLA then has to
dead-code-eliminate. That is why your compiles are slow. Fixing it is a one-line change:
apply the gates in place instead of building a sub-circuit and appending it —
`tests/nan_investigation/fastansatz.py` does exactly this and is **bit-identical** to your
version, `t06_fast_equivalence.py`: `max|state diff| = 0.000e+00`, and 20× faster to
build.)

---

## 8. Fixes, in the order I would apply them

### Fix 1 — turn the splitter off (one line, removes the bug entirely)

`src/utilities/generate_ansatz.py:410`:

```python
split_conf = None
```

Measured effect (`t22_degeneracies_and_fixes.py`): gradient error **exactly 0.0** at every
θ tested, including the degeneracies. Faster, too (§7). Because every gate you use has
Schmidt rank ≤ 2 the forward result is unchanged.

### Fix 2 — run in double precision (do this regardless)

In `src/find_gs.py`, right after `K = tc.set_backend("jax")`:

```python
K = tc.set_backend("jax")
tc.set_dtype("complex128")          # <-- this is what actually turns on float64
```

`tc.set_dtype("complex128")` also sets `jax_enable_x64=True` for you (`cons.py:223`), so it
must come *after* the `tc.set_backend` call and can replace the manual
`config.update("jax_enable_x64", True)` lines. Do the same in
`src/utilities/generate_ansatz.py` and `generate_ising_hamiltonian.py`, and be careful about
import order — `import tensorcircuit` resets x64 to `False`.

Measured effect (§4.1): worst-case error drops from 2.3e7 to 3.6e-1, and the dangerous
window narrows from |θ−π/2| ≲ 1e-4 to ≲ 1e-10. On the full 800-step training run (§5.1) it
takes the corrupted-step count from **58/800 to 0/800** and the worst relative gradient
error from **1.3e+04 to 3.3e-08**, while converging closer to the exact ground state. You
are also currently reporting toric-code energies of order −12 computed entirely in float32;
double precision is worth it on its own.

Fixes 1 and 2 are independent and you want both. Fix 1 removes the mechanism; Fix 2 makes
the rest of the pipeline trustworthy and protects you if a splitter reappears.

### Fix 3 — a safety net in the optimiser (cheap, catches everything)

`src/find_gs.py`, in `optimize`:

```python
optimizer = optax.chain(
    optax.zero_nans(),                      # kill NaN/inf before they reach the state
    optax.clip_by_global_norm(1.0),         # cap the spikes
    optax.adam(learning_rate=self.learning_rate),
)
```

and assert on the way in so a poisoned step is loud rather than silent:

```python
value, gradient = self._cost_vvag(params)
if not bool(jnp.all(jnp.isfinite(gradient))):
    raise FloatingPointError(f"non-finite gradient at step {i}")
```

Note `optax.zero_nans` sits *before* adam so the moment estimates never see the NaN.

### Fix 4 — do **not** do this

* `max_singular_values=4` → NaN nearly everywhere (§4.2).
* Raising `_safe_reciprocal`'s ε by monkey-patching → helps (§4), but you are patching a
  library internal, you must remember `jax.clear_caches()` + rebuilding
  `jax_ops.adaware_svd_jit`, and it still leaves an O(0.2) error at the degeneracy. Only
  worth it if you have a reason to keep the splitter.

### Optional cleanups found along the way

* `ToricCodeAnsatz.__hash__` omits `unitary`, `use_prob_resets` and
  `use_small_angle_initialization`, so two structurally different ansätze hash equal
  (verified, `t27_hash_eq.py`). `find_gs.purity`/`purity_vec` pass the ansatz as a jit
  `static_argnums`, so this matters. It does not currently give wrong answers — `__eq__`
  falls through to a `__dict__` comparison that happens to return `False` first — but it is
  one field-reordering away from either a wrong cache hit or a `ValueError` from comparing
  jax arrays. Add those three fields to `__hash__` and give `__eq__` an explicit field list.
* `VariationalAnsatz.energy_from_params` calls `self._circuit(params, seed)` positionally,
  but `ToricCodeAnsatz._circuit(self, params, *args, seed=None)` takes `seed` as
  keyword-only — so `seed` lands in `*args` and the keyword stays `None`. Harmless today
  (the toric-code builders ignore it), but the noisy path cannot actually seed the circuit.
* `find_gs.py:12` imports `scipy.optimize.minimize` and never uses it.

### Reference energies for checking convergence

Exact diagonalisation (`t26_misc_checks.py`), for `H = −(1−h)(Σ stars + Σ plaquettes) − h Σ Zᵢ`:

| lattice | qubits | stars | plaquettes | E₀ (h=0) | E₀ (h=0.1) | gap (h=0.1) |
|---|---:|---:|---:|---:|---:|---:|
| 3×3 | 12 | 9 | 4 | −13.000000 | **−11.820764** | 1.679172 |
| 3×2 | 7 | 6 | 2 | −8.000000 | −7.310437 | 1.830417 |
| 2×2 | 4 | 4 | 1 | −5.000000 | −4.584886 | 1.969772 |

The ground state is non-degenerate in every case, so the search target is unambiguous.

---

## 9. Tests to run on your GPU machine

I could only test CPU, `tensorcircuit-ng 1.9.1`, `jax 0.10.2`. These four take minutes and
close the remaining gaps. All scripts are in `tests/nan_investigation/`.

**T1 — confirm the mechanism on your stack (2 min).**
```bash
python tests/nan_investigation/t17_rzz_degeneracy_scan.py            # DT=complex64 default
DT=complex128 python tests/nan_investigation/t17_rzz_degeneracy_scan.py
```
Expect the §4.1 table. If your worst-case error at θ = π/2 is ~1e7 in complex64 and ~1e-16
in complex128, the diagnosis transfers to your versions unchanged.

**T2 — find your actual first NaN (the important one).**
Add this to `find_gs.optimize`, immediately after `value, gradient = self._cost_vvag(params)`:
```python
g = np.asarray(gradient)
if not np.all(np.isfinite(g)) or np.max(np.abs(g)) > 1e3:
    bad = np.argwhere(~np.isfinite(g)) if not np.all(np.isfinite(g)) else np.argwhere(np.abs(g) > 1e3)
    tr, idx = bad[0]
    th = float(np.asarray(params)[tr, idx])
    print(f"step {i}: trial {tr} param {idx}  |g|={g[tr, idx]}  theta={th!r}  "
          f"theta/(pi/2)={th/(np.pi/2)!r}  block-offset={idx % 9}")
    np.save(f"blowup_step{i}.npy", np.asarray(params))
    break
```
`block-offset` 6, 7 or 8 means the culprit is an `rxx`, `ryy` or `rzz` angle (each Cartan
block is 9 params, the last three are the two-qubit rotations). If `theta/(pi/2)` prints as
a near-integer, this report is your cause. If the first non-finite value appears with all
angles far from multiples of π/2, it is something else — send me that dump.

**T3 — rule the mechanism in or out by elimination (one run each).**
Run your 3×3 case three times, changing exactly one thing:
1. as-is → expect NaN as before;
2. `split_conf = None` in `generate_ansatz.py` → expect no NaN;
3. `tc.set_dtype("complex128")` after `tc.set_backend` → expect no NaN.

If (2) and (3) both survive and (1) dies, you are done. (On the 3×2 lattice I ran
exactly this comparison for (1) vs (3) — 58/800 corrupted steps vs 0/800, §5.1.)

**T4 — check whether GPU cuSOLVER adds its own failure (this is the one I can't do).**
```python
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
# 4x4 operator-Schmidt matrices of rzz(theta) for theta on a fine grid around pi/2
I2 = np.eye(2); Z = np.diag([1., -1.])
th = np.pi/2 + np.linspace(-1e-5, 1e-5, 200001)
U = (np.cos(th/2)[:,None,None,None,None]*np.einsum('ac,bd->abcd', I2, I2)
     - 1j*np.sin(th/2)[:,None,None,None,None]*np.einsum('ac,bd->abcd', Z, Z))
M = np.ascontiguousarray(U.transpose(0,1,3,2,4).reshape(-1,4,4))
for dt in (jnp.complex64, jnp.complex128):
    s = jnp.linalg.svd(jnp.asarray(M, dt), compute_uv=False)
    s = np.asarray(s)
    print(dt.__name__, "non-finite rows:", int(np.sum(~np.isfinite(s).all(axis=1))),
          " s1==0 rows:", int(np.sum(s[:,1] == 0)),
          " max s0-s1:", float(np.max(s[:,0]-s[:,1])))
```
On CPU this prints `0` non-finite and `s1==0` only at exact θ=0. If your GPU prints a
nonzero count for either, cuSOLVER is contributing an *additional*, independent NaN source
and Fix 1 (removing the splitter) becomes the mandatory fix rather than merely the best one.

---

## 10. Script index

| script | proves |
|---|---|
| `t01_schmidt_spectra.py` | operator-Schmidt spectra of every gate used; where the degeneracies are |
| `t02_dtype.py` | the whole simulation runs in complex64; the `_safe_reciprocal` peak |
| `t05_split_propagation.py` | `qc.append` routes every 2q gate through the splitter; lossiness for rank-4 gates |
| `t06_fast_equivalence.py` | the fast in-place circuit builder is bit-identical to the repo's |
| `t13_nan_hunt.py` | dense θ scan; NaN only at θ=0; first sighting of the 2.3e7 for `rzz` |
| `t17_rzz_degeneracy_scan.py` | **the main result**: error vs distance to degeneracy, c64 vs c128 |
| `t19_jit_effect.py` | jit makes the blow-up ~10⁸× worse |
| `t20_zero_singular_values.py` | `s₁ == 0` happens only at θ = 0.0 exactly |
| `t21_train_track.py` | **real training run**: 58/800 corrupted steps in complex64, 0/800 in complex128 |
| `t25_many_degeneracies.py` | magnitude vs number of simultaneously-Clifford angles (up to 1.03e8); NaN at θ=0 |
| `t22_degeneracies_and_fixes.py` | the θ→π degeneracy; `max_sv=4` is worse; the fixes work |
| `t24_epsilon_sweep.py` | ε is the knob; instrumented backward pass under jit |
| `t23_epsilon_and_perf.py` | the split is 1.6× slower |
| `t26_misc_checks.py` | exact reference energies; split lossiness table |
| `t27_hash_eq.py` | ansatz `__hash__` collisions |

---

## 11. Findings outside the SVD path (added after the first pass)

### 11.1 `perform_noisy_simulations=True` applies no noise at all

`find_gs.py:169-173` calls

```python
noise_conf.add_noise("depolarizing", [self.noise_rate*0.1],
                     ["x","y","z","h","s","t","rx","ry","rz"])
```

but the signature is `NoiseConf.add_noise(gate_name, kraus, qubit=None)`. All three
arguments are in the wrong slot, and `add_noise` does `zip(qubit, kraus)` internally, so
a one-element `kraus` truncates the nine-element gate list to one. Measured
(`t30_noise_config.py`):

```
repo's config  -> 2 rule(s): [('depolarizing', 'x'), ('depolarizing', 'cnot')]
correct config -> 7 rule(s): ['rx', 'ry', 'rz', 'x', 'y', 'z', 'h']

noiseless <Z0 Z1>            : -0.0120958686
repo's noisy path (nmc=200)  : -0.0120958686
|difference|                 :  0.000e+00      <-- bitwise identical

density-matrix sim with real depolarizing noise : -0.0083966851  (shift 3.70e-03)
```

tensorcircuit warns `gate name depolarizing not in the common gate set that tc supported`
on every call — that warning is the bug announcing itself. Net effect: the run costs
`nmc` times more and returns the **noiseless** answer. `src/examples/find_gs_tc_example.py`
uses this path with `noise_rate=5e-2, number_of_shots=2000`.

### 11.2 The per-step seeds do nothing

`optimize()` builds `all_seeds` and threads a seed through `energy_with_seed`, but
`K.set_random_state(seed)` is a Python-level side effect that runs once at trace time, and
the channel randomness inside `expectation_noisfy` comes from `backend.implicit_randu`,
not from the `status` argument the code passes (that one is the *shot* randomness).
Measured (`t28_pipeline_consistency.py`, part d): five different seeds give **bitwise
identical** energies and gradients. Even with 11.1 fixed, every optimisation step would
sample the same noise realisation.

### 11.3 What a corrupted gradient does to Adam — measured, in two phases

`t29_adam_after_spike.py`. One parameter, honest gradient 0.1, `lr=1e-2`, one corrupted
step of size `G` of the wrong sign:

| G | \|step\| at t+1 | \|step\| at t+100 | \|step\| at t+350 | steps to recover | progress lost by step 60000 |
|---:|---:|---:|---:|---:|---:|
| 1e+02 | 6.4e-03 | 1.2e-04 | 2.2e-04 | 5,810 | 56 of 600 rad |
| 1e+04 | 6.4e-03 | 9.1e-07 | 2.2e-06 | 15,012 | 147 of 600 rad |
| 1e+06 | 6.4e-03 | 3.2e-07 | 2.2e-08 | 24,218 | 239 of 600 rad |
| 2.3e+07 | 6.4e-03 | 3.3e-07 | 9.4e-10 | 30,486 | 302 of 600 rad |
| 1e+08 | 6.4e-03 | 3.3e-07 | 2.2e-10 | 33,424 | 331 of 600 rad |

Adam is scale invariant, so the spike does **not** produce a giant jump — `m` and `v` are
inflated together. Instead:

* **Phase 1**, about `ln(G/g)/ln(1/0.9)` ≈ 110–200 steps: `m` is still dominated by the
  spike, so the parameter marches at roughly the full learning rate in whatever
  (meaningless) direction the corrupted gradient pointed.
* **Phase 2**, about `ln(1e-3 G²/g²)/ln(1/0.999)` ≈ 16,000–34,000 steps: `m` has decayed
  back to the honest gradient but `v` has not, so the effective step collapses by a factor
  of ~10⁻⁸ and the parameter is **frozen**.

With `maxiter=501`, a single spike anywhere in the first half of the run ends that
parameter's optimisation permanently. No NaN required.

### 11.4 Checks that came back clean

`t28_pipeline_consistency.py`:

* `sparse=True` and `sparse=False` agree to float32 roundoff (≤1.2e-07) on all three
  toric-code ansatz branches — the two Hamiltonian construction routes are consistent.
* `self.nancillas` matches the circuit width for all four branches (unitary, dynamic,
  prob-resets, small-angle) at 3×3.
* The `claws[i::4][j]` reordering is a genuine permutation for 2×2, 3×2, 3×3, 4×3 and 4×4
  — no gates silently dropped or duplicated.
