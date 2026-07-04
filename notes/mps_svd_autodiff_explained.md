# Why MPS training produced NaNs: forward passes, reverse-mode autodiff, and VJPs

You said you're not confident on forward passes, reverse-mode autodiff, or
VJPs, so this document builds those up from nothing, then uses them to
explain precisely why `tc.MPSCircuit` broke and `tc.Circuit` doesn't, at the
same system size. The last section lays out what this implies for whether
MPS is a good fit for a 2D lattice-plus-ancillas system like yours — as
things to weigh, not a verdict, since you asked to understand the mechanism
before deciding that yourself.

---

## 1. Two different questions your training loop asks

Every step of `ansatz.optimize()` needs two things from
`K.value_and_grad(energy_fn)(params)`:

1. **"What is the energy, given these parameters?"** — run the circuit,
   compute a number. This is the **forward pass**: plug in concrete numbers,
   follow the computation through, get an output. Nothing clever here, it's
   just "run the program."
2. **"If I nudge each of these 381 parameters slightly, how does the energy
   change?"** — the **gradient**. This is what the optimizer (Adam) actually
   uses to decide how to update the parameters.

Both of your NaN failure modes map directly onto these two questions:
**forward NaN** = question 1 itself returns garbage. **backward NaN** =
question 1 is fine, but question 2 (the gradient) is garbage. These are
genuinely separate failures with separate causes — that's why the fix had
two independent parts. We'll do the forward one second, because it's the
easier of the two; the real conceptual weight is in the gradient, so let's
build up to that properly.

---

## 2. The chain rule, refreshed with actual numbers

Everything below is the calculus chain rule, applied mechanically and
repeatedly. Quick refresher with a trivial example, because everything else
builds on this:

Suppose `b = 2a` and `c = b + 1`, so overall `c = 2a + 1`. If `a` increases
by a tiny amount `da`, then `b` increases by `2·da` (since `b=2a`), and `c`
increases by that same amount again (since `c=b+1`, a 1-to-1 relationship).
So: *how much `c` changes per unit change in `a`* is `2 × 1 = 2`. That's the
chain rule: **multiply the local sensitivities together, step by step,
through however many intermediate steps there are.**

In symbols, if `c` depends on `b` which depends on `a`:
```
dc/da = (dc/db) × (db/da)
```
Every operation in your circuit — every gate, every matrix multiply, every
SVD — is one link in a much longer chain like this, and the chain rule is
the *only* tool being used to get a gradient out the other end. The whole
of "autodiff" is just: mechanically apply this multiplication, link by
link, for a chain with potentially thousands of links, without you having
to derive the combined formula by hand.

---

## 3. Two ways to walk the chain: forward-mode vs. reverse-mode

Here's the part that isn't obvious: **you can walk that chain of
multiplications in either direction**, and the direction you pick has a
huge effect on cost when there are many inputs and few outputs (your case
exactly: 381 parameters in, 1 energy number out).

Picture a computation as a chain of boxes, each one a step in your circuit,
with your 381 parameters entering on the left and one energy number coming
out on the right.

### Forward-mode

Start at the leftmost input, ask "how does the output change if I wiggle
*this one* parameter, holding all others fixed?", and propagate that
sensitivity forward through every box, left to right, until it pops out the
other end as one number: `d(energy)/d(param_i)`.

That gives you the answer for *one* parameter. To get the full gradient
(all 381 partial derivatives) you'd have to repeat this entire left-to-right
sweep **381 separate times**, once per parameter. Cost scales with
**number of inputs**.

### Reverse-mode ("backprop")

Instead: first run the whole computation forward once, left to right,
computing and *remembering* every intermediate value along the way (this
remembered trail is often called the **tape**). Then, starting from the
*output* end, ask "how sensitive is the final energy to this box's output?"
and propagate that sensitivity **backward**, right to left, one box at a
time, until it reaches every input simultaneously.

Because there's only **one** output (the energy is a single number), this
backward sweep needs to happen only **once**, and at the end you have all
381 partial derivatives simultaneously — not one at a time. Cost scales
with **number of outputs**.

### Why this matters for you concretely

You have 381 parameters and 1 output. Forward-mode: 381 sweeps. Reverse-mode:
1 sweep. This is exactly why `jax.grad` / `K.value_and_grad` default to
reverse-mode, and it's the same reason every neural network in existence is
trained with "backpropagation" rather than forward-mode differentiation —
backprop *is* reverse-mode autodiff, same algorithm, different name because
it was invented independently in the neural-net literature first.

This also tells you something important for later: **reverse-mode requires
that forward pass to have actually happened and been remembered first.**
You can't run the backward sweep without the tape from the forward sweep.
Keep that in mind for §5.

---

## 4. Reverse mode in detail: what a "VJP" actually is

Now let's be precise about what happens at *each box* during that backward
sweep, because "VJP" is just the name for exactly this per-box operation.

**Cotangent** is the term JAX/autodiff literature uses for "the sensitivity
signal flowing backward" at a given point in the chain — i.e. "how much does
the final loss change per unit change in *this* intermediate quantity."
It's not a fundamentally new concept, just a name for "the running product
of local derivatives so far," accumulated as the backward sweep passes
through each box.

**VJP = Vector-Jacobian Product.** For any operation `y = f(x)` where `x`
and `y` might be vectors/matrices (not just single numbers), the **Jacobian**
`J` is the full matrix of *every* partial derivative of *every* output
w.r.t. *every* input (`J[i,j] = dy_i/dx_j`). Forming this whole matrix
explicitly would be expensive and usually unnecessary. Instead, reverse-mode
autodiff only ever needs: *given an incoming cotangent vector `v` (sensitivity
w.r.t. the outputs `y`), produce the outgoing cotangent `v @ J` (sensitivity
w.r.t. the inputs `x`)* — this product, computed *without ever building `J`
itself*, is the VJP. Every JAX primitive operation (add, multiply, matmul,
reshape, SVD, ...) has to define this VJP rule; that collection of rules is
the entire machinery `jax.grad` is built on.

Two concrete examples so this isn't abstract:

- **Multiply**, `y = a * b`. If the incoming cotangent (sensitivity w.r.t.
  `y`) is `v`, the VJP rule says: sensitivity w.r.t. `a` is `v * b`,
  sensitivity w.r.t. `b` is `v * a`. (This is just the product rule from
  calculus, packaged as a reusable recipe.)
- **Matrix multiply**, `y = A @ x`. Incoming cotangent `v` (same shape as
  `y`). VJP rule: sensitivity w.r.t. `x` is `A^T @ v`, sensitivity w.r.t.
  `A` is the outer product `v @ x^T`. Notice: *no division anywhere*. This
  will matter a lot in a moment.

Reverse-mode autodiff is nothing more than: walk the recorded tape backward,
and at every box, apply that box's VJP rule to turn "sensitivity w.r.t. my
output" into "sensitivity w.r.t. my input(s)," handing that result to the
box before it. Do this all the way back to the 381 parameters, and you have
your gradient. `jax.custom_vjp` (used in the stashed fix) is JAX letting you
*override* this rule for one specific operation — normally JAX derives it
automatically, but you can hand-write your own instead.

---

## 5. Applying this to SVD: where `1/(σᵢ²−σⱼ²)` actually comes from

SVD decomposes a matrix `A` into `A = U · S · Vᵀ`, where `S` is a diagonal
matrix of **singular values** `σ₁ ≥ σ₂ ≥ ... ≥ 0` (think of these as "how
much information/entanglement lives along each of these directions" — in
MPS terms, they're exactly the **Schmidt coefficients** across that bond).
`U` and `V` are matrices whose *columns* are directions in input/output
space, orthogonal to each other.

In your MPS circuit, applying a 2-qubit gate merges two neighboring MPS
tensors into one bigger tensor, and SVD is used to **split it back into two
smaller tensors while truncating** — keeping only the largest `bond_dim`
singular values and discarding the rest. That SVD call is one box in the
forward pass, with **three outputs**: `U`, `S`, `V`.

Now here's the part that's genuinely different from the multiply/matmul
examples above. During the backward sweep, this SVD box receives **three
separate incoming cotangents** — `dU`, `dS`, `dV` (sensitivity of the final
loss w.r.t. each of the three outputs) — and has to produce **one** combined
answer: sensitivity w.r.t. the *original* matrix `A`. That means the VJP
rule has to figure out: "given that wiggling `U` a bit and wiggling `V` a
bit both affect the loss, how much did the *original input matrix* `A` need
to change to cause exactly those wiggles in `U` and `V`?"

This is where it stops being "just apply the product rule" and becomes a
genuine linear-algebra derivation (first worked out in the numerical linear
algebra literature decades ago, and independently rediscovered in the ML
literature once people started backpropagating through PCA/whitening-style
layers). The derivation involves a matrix `F` where:
```
F[i,j] = 1 / (σᵢ² − σⱼ²)     for i ≠ j
```
**Why does a *difference of squared singular values* show up at all?**
Intuitively: `U` and `V`'s columns are only defined *up to rotation within
any subspace of equal singular values*. If `σᵢ = σⱼ`, the SVD could just as
validly have used a *rotated* pair of columns for `i` and `j` — there's no
way to tell, from `A` alone, "how much" of any wiggle in `U`/`V` came from a
genuine change to `A` versus an arbitrary, physically meaningless rotation
within that degenerate subspace. The VJP formula has to "undo" these
rotations to recover the true gradient w.r.t. `A`, and the amount of
rotation is inversely proportional to how far apart `σᵢ` and `σⱼ` are. As
`σᵢ → σⱼ`, that denominator `σᵢ² − σⱼ²` goes to zero and `F[i,j]` blows up
toward infinity — **this is a genuine mathematical singularity in the SVD
gradient, not a bug or a missing edge case in the implementation.**

TensorCircuit's built-in SVD gradient rule regularizes this by computing
`x / (x² + ε)` instead of the bare `1/x` (this smooths out the *exact*
`σᵢ = σⱼ` case, where the formula would divide by literal zero), with a
tiny `ε = 1e-15`. That's enough to avoid `1/0`, but does almost nothing
once `σᵢ` and `σⱼ` are merely *close* rather than *identical* — and "close"
turns out to be extremely common (§8). The stashed fix used the same
`x/(x²+ε)` shape but with `ε = 0.5`, a much bigger number, which caps how
large `F[i,j]` can ever get, at the cost of the gradient becoming an
*approximation* rather than mathematically exact whenever singular values
are near-degenerate — a deliberate trade of exactness for not-getting-NaN.

**And here's the compounding problem, in the numbers you already had:**
capping `F_max ≈ 0.71` per SVD step still means each sequential SVD can
amplify the gradient signal by roughly ×3.83. One `distance-6` gate needs
up to 11 chained SWAP+SVD steps just to bring two logically-distant qubits
adjacent in the 1D MPS chain (more on why in §8) — and `3.83^11 ≈ 1.5
million`. Do that across many gates, many layers, across a whole training
step, and even a "safely regularized" per-step amplification factor
compounds into enormous numbers extremely fast. This is exactly why the fix
needed such an aggressively large `ε` — anything smaller still overflows
once you chain enough of these together, and that's before considering
that some of these compounding steps might hit *exactly* degenerate
singular values (§8), for which no finite `ε` fully fixes the problem.

---

## 6. The other failure mode: forward-pass NaN (separate cause)

This one is simpler and doesn't involve gradients at all. **Rank** of a
matrix is, informally, "how many genuinely independent directions of
information it contains." If you ask SVD for `bond_dim = 16` singular
values/vectors, but the actual two-site tensor you're decomposing only has,
say, rank 4 (common early in a circuit, or near a chain boundary, before
much entanglement has built up), then 12 of the 16 "singular vectors" you
asked for don't correspond to any real direction — they live in what's
called the matrix's **null space**.

Different SVD implementations handle this null space differently.
CPU LAPACK routines tend to fill it with an arbitrary (but at least
*finite*) orthonormal basis. The GPU library used here, **cuSOLVER**, can
instead return literal `NaN` values in those extra columns. That NaN
value is then a real number sitting in your MPS tensor from that point
forward — it doesn't need a gradient to be a problem, the *energy itself*
comes out as NaN the moment that NaN-containing tensor is used in the next
contraction. The fix for this half was much simpler: `nan_to_num` the raw
SVD output before using it (those NaN columns correspond to zero singular
values anyway, so replacing them with zero is not just a hack — it's the
mathematically correct thing to do, since they contribute nothing to the
represented state).

---

## 7. Why `tc.Circuit` (dense statevector) never hits either of these

This is the direct comparison you asked for, and now you have the machinery
to see exactly why the two approaches diverge.

A dense statevector simulation represents your whole system as one big
vector of `2^n` complex numbers, and applies each gate as a **matrix
multiply / tensor contraction** directly against that vector (or, for
tensorcircuit's contraction-based simulation, a sequence of tensor
contractions equivalent to one). There is no compression step anywhere —
the state is exact, full-sized, all the time.

Walk back through §4: the VJP rules needed for a dense circuit are things
like the matmul example — multiply, add, contract, reshape. **None of these
ever involve a division that can blow up.** Multiply's VJP is another
multiply. Matmul's VJP is another matmul. These are the single
most-optimized, most numerically battle-tested operations in all of
scientific computing and deep learning — every neural network on Earth
backpropagates through millions of matmuls without this class of problem,
because matmul's gradient rule has no singularity in it, structurally.
There is no equivalent of "two singular values happen to be close together"
lurking anywhere in a plain matmul-based pipeline.

So the direct answer to "why don't we get errors there at the same system
size": **it's not that the dense method is somehow more numerically robust
in general — it's that it simply never calls the one specific operation
(SVD, used for truncation) whose gradient has a genuine mathematical
singularity built into it.** Same lattice, same number of qubits, same
gates — the only difference is dense statevectors never compress the state,
so they never need the operation that's causing all of this.

The trade-off, which is *why* MPS was attractive in the first place: a
dense statevector's memory cost is `O(2^n)` — exponential in qubit count,
which is exactly the wall you hit in the earlier GPU-OOM investigation.
MPS's memory cost is roughly `O(n · bond_dim²)` — polynomial, controllable
via `bond_dim`. MPS buys you that scaling *specifically* by introducing
truncation (SVD), and it's that same truncation step that turns out to be
the numerically fragile one to differentiate through. There's no free
lunch here: the thing that makes MPS memory-cheap is mechanically the same
thing that makes its gradient fragile.

---

## 8. Why bigger bond dimension makes it worse — and why the toric code
   might be an especially bad case, not just an unlucky one

Two separate effects push in the same direction as you scale up:

**Effect 1 — more singular values, more chances for a close pair.** With
`bond_dim = 4` you only ever have 4 singular values per bond; the odds any
two of them land close together are low. With `bond_dim = 32`, you have 32
numbers packed into the same range `[0, 1]` — by basic pigeonhole reasoning,
some pair is much more likely to be close. This alone explains "worked at
2×2, strange at 3×2, NaN once we pushed bond dimension up" as a smooth,
predictable trend rather than a one-off bug.

**Effect 2 — this specific physical system may produce *exactly* degenerate
singular values, not just statistically likely near-misses.** This is the
part worth taking seriously before concluding anything about MPS in
general. The toric code ground state (and stabilizer states more broadly)
has a well-known property in the quantum-information literature: its
**entanglement spectrum is exactly flat** — meaning, for a bipartition
(a "cut" splitting the system in two, which is exactly what each MPS bond
represents), *every nonzero Schmidt coefficient across that cut is
identical*. This isn't a numerical coincidence, it falls directly out of the
stabilizer formalism: the reduced state on either side of the cut is
*maximally mixed* on its support, and a maximally mixed state has, by
definition, all its nonzero eigenvalues equal.

If that's right, then as your ansatz trains *toward* something close to the
toric code ground state, you're not occasionally unlucky enough to hit two
close singular values — you should *expect*, as a structural feature of the
target state itself, large groups of *exactly or near-exactly* degenerate
singular values at many bonds. No finite regularization `ε` fully rescues
an *exact* degeneracy; you can only ever trade how large the resulting
gradient error is, never eliminate the underlying singularity. This is a
meaningfully different (and harder) situation than "SVD gradients are
occasionally fragile" — it suggests the specific state you're trying to
learn is close to a worst case for this method.

**Effect 3 — geometry mismatch compounds both of the above.** An MPS is
fundamentally a **1D chain** — gates apply cheaply only between
*neighboring* sites in that chain. Your system is a 2D lattice plus
ancillas; mapping it onto a 1D ordering necessarily means some pairs of
qubits that are adjacent *in the lattice* end up far apart *in the MPS
chain*. To apply a gate between them, the code has to `SWAP` them next to
each other first — and each `SWAP` on an MPS is itself another SVD-based
split step. That's where "up to 11 SWAP+SVD steps for one distance-6 claw
gate" comes from: it's not 1 fragile operation per gate, it's potentially a
dozen, and every single one is another link in the compounding chain from
§5. A geometry better suited to a 1D chain (e.g. a genuinely 1D spin chain)
would need far fewer of these per gate; a 2D lattice with long-range
"claw" terms is close to a worst case for how many SVDs get chained
together per logical gate.

---

## 9. Why "turning up epsilon" didn't rescue it, in hindsight

Put §5, §8-Effect-2, and §8-Effect-3 together and the outcome you saw
stops looking surprising: you were chaining together dozens of SVD steps
per training call, on a system whose target state structurally produces
exactly-degenerate singular values, at a bond dimension high enough to make
near-misses common even ignoring the exact-degeneracy issue. Any fixed
regularization constant has to trade off two failure directions at once —
too small, and near-degenerate cases still overflow after enough chained
multiplications; too large, and the gradient becomes so distorted from the
true value that training either doesn't converge or converges to the wrong
thing (a large `ε` isn't "safe," it's "differently wrong"). There may not
be a single `ε` that survives both an exactly-flat target spectrum *and*
a dozen-deep SWAP+SVD chain per gate — which is a structural argument, not
a "you didn't tune it enough" one.

---

## 10. What this means for "is MPS a good fit here" — questions to weigh

You asked to understand the mechanism before drawing this conclusion
yourself, so here's the shape of the decision rather than an answer:

- Does your target state (or states close to it along the optimization
  trajectory) plausibly have a flat or near-flat entanglement spectrum at
  the bonds you're truncating? If yes, that's a structural argument against
  *any* naive gradient-based SVD-truncated method for *this specific
  system*, independent of implementation quality.
- Is the 2D-lattice-into-1D-chain SWAP overhead avoidable, e.g. by choosing
  the MPS ordering to minimize the maximum gate "distance" (the
  "interleaved ordering... optimal qubit to MPS ordering" work mentioned in
  the `e07e2c6` commit looks like exactly this attempt)? Even a good
  ordering can't remove the flat-spectrum problem, only Effect 3. Tensor
  network geometries designed natively for 2D lattices do exist (e.g.
  **PEPS**, Projected Entangled Pair States — the natural 2D generalization
  of MPS), but they come with their own, generally *harder*,
  differentiability and contraction-cost challenges — worth being aware
  this isn't a free escape hatch.
- Is there a way to avoid differentiating *through* the truncation at all —
  e.g. treating `bond_dim` truncation as a fixed, non-differentiated
  approximation and estimating gradients some other way (parameter-shift,
  finite differences, or a custom estimator that doesn't route through
  SVD's VJP)? This trades one set of problems for another (usually cost or
  variance) but sidesteps this specific singularity entirely.
- Given the dense-statevector approach already works correctly and the
  memory wall only bit at ~29 qubits, is the actual near-term need for MPS
  as strong as it seemed when the memory OOM was the only known problem?
  Now that there's a second, structural obstacle specific to this physical
  system, the cost/benefit may look different than it did when MPS was
  first tried.

---

## 11. Rules of thumb + where to learn more

- **Any operation whose gradient formula involves dividing by a difference
  of two things that *can* be equal (eigenvalues, singular values, sorted
  quantities generally) is a candidate for this exact failure mode.** SVD,
  eigendecomposition, and "top-k" style operations are the classic examples
  across all of ML, not just here.
- **A "forward NaN" and a "backward (gradient) NaN" are different bugs with
  different causes, even when they show up in the same operation** — always
  check which one you're actually looking at (does the *loss value* print
  as NaN, or only the *gradient*?) before reaching for a fix.
- **Search terms for further reading** (rather than links, since exact doc
  locations move around): *"differentiable SVD gradient instability"*,
  *"backprop through eigendecomposition degenerate eigenvalues"*,
  *"forward-mode vs reverse-mode automatic differentiation"*, *"vector
  Jacobian product explained"*, *"entanglement spectrum stabilizer states
  flat"*, *"PEPS vs MPS 2D tensor networks"*.
- **A cheap sanity probe you can run without touching any real training
  code:** take a small already-known highly-symmetric state (even just 2-3
  qubits of a toy stabilizer state) and directly inspect its Schmidt
  spectrum (`numpy.linalg.svd` on the reshaped statevector) across a cut.
  If the nonzero singular values really are numerically identical, that's
  a fast, concrete confirmation of the §8-Effect-2 hypothesis before
  concluding anything about the full lattice.
