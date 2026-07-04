# Why the H200 run OOM'd during training (and the JAX concepts behind it)

This explains the crash from `logs/2026-07-04_15-45-38_4x3_test_3529279/stderr.err`,
where training died on the very first step with a 280GB allocation failure,
even on a 141GB H200. The short version: **this isn't "not enough memory," it's
JAX accidentally making dozens-of-GB-sized copies of your Hamiltonian that it
doesn't need to make.** The rest of this doc builds up the concepts needed to
see exactly why, from the ground up.

No prior JAX-internals knowledge assumed. Skim the headers, read what's new to you.

---

## 1. The one-sentence summary

Your Hamiltonian matrix (`self.fullham`, a huge sparse array) gets silently
"baked into" the compiled training-step program as a giant literal constant
instead of being treated as reusable data — and because that program is also
being vectorized over your 5 trials and differentiated (for gradients), JAX
ends up baking in *multiple copies* of it. 5 trials × ~35GB × (a couple of
copies for forward and backward passes) ≈ the 601GB the warning reported.

Everything below is unpacking what each piece of that sentence actually means.

---

## 2. Background: what does `jit` actually do?

Normal Python runs line by line, on the CPU, one operation at a time — call
`+`, Python computes it immediately, moves on. That's far too slow for a GPU:
launching one GPU instruction at a time has huge overhead, and the GPU wants
one big fused program to chew on, not a thousand tiny handoffs from Python.

`jax.jit` (used here as `K.jit`, `K` being tensorcircuit's JAX backend
wrapper) solves this by **tracing** your Python function *once*, recording
every operation it performs, and handing the *whole recorded sequence* to a
compiler (XLA — more on that in §7) which turns it into one optimized GPU
program. After that, calling the jitted function again just runs that
pre-built program directly on the GPU — no Python involved, very fast.

The catch: to trace your function, JAX doesn't run it with your real numbers.
It runs it with **tracers**.

---

## 3. Tracers: fake numbers used to "record the recipe"

A **tracer** is a placeholder JAX substitutes for a real array when tracing.
It knows the array's *shape* and *dtype* (e.g. "a 381-length vector of
float64") but not its actual values. Think of it like a stunt double: same
size and shape as the real actor, stands in the right position, but you
can't ask it to actually say the lines.

When your function runs on a tracer instead of a real array, every
operation you do to it (`x + 1`, `jnp.sin(x)`, `matrix @ x`, ...) gets
*recorded* as a step in a graph, instead of *computed*. At the end, JAX has
a full graph of "first do this, then that, then that" — this graph is
called a **jaxpr** (JAX + Program), and it's what gets handed to the
compiler. This is why JAX documentation says `jit` requires your function to
be "pure": it only ever *sees* tracers, so anything that depends on real
concrete values happening during tracing (like an `if x > 0` branch on a
traced value) won't work the way plain Python expects.

**Key fact for what follows:** only the things you explicitly *pass in* as
arguments to the jitted/vmapped function get replaced by tracers. Anything
else your function reaches out and touches is not a tracer — it's a real,
concrete value. That distinction is the whole story.

---

## 4. Traced arguments vs. closed-over constants

This is the crux of the bug, so slow down here.

### Traced argument
```python
cost_vvag = K.jit(K.vmap(K.value_and_grad(energy_fn, argnums=0), vectorized_argnums=0))
value, grad = cost_vvag(params)
```
`params` is passed *into* the jitted call. During tracing, JAX replaces it
with a tracer. The compiled program says, in effect, "whatever real buffer
shows up in argument slot 0 at call time, run the recorded steps on it."
The actual data for `params` lives in exactly one place in GPU memory and is
handed to the compiled program *by reference* — it is never duplicated just
because it's an argument.

### Closed-over constant
Now look at [find_gs.py:199](../src/find_gs.py#L199):
```python
def energy_from_params(self, params, seed=None):
    qc = self._circuit(params, seed)
    return K.real(sparse_expectation(qc, self.fullham))
```
`self.fullham` is *not* a parameter of `energy_from_params` — it's reached
via `self`, a Python object that the function's closure remembers from when
it was defined (`energy_fn = ansatz.energy_from_params` — a **closure** is
just "a function bundled with references to variables from its enclosing
scope," here `self`). Since `self.fullham` was never listed as an official
input, JAX never gives it a tracer. During tracing, JAX sees a real,
concrete ~30GB array being used, and has only one option: treat it as a
**constant** and write its literal contents directly into the jaxpr/compiled
program.

This is the difference between:
- *"Insert whatever buffer arrives in slot 2 here"* (traced argument — cheap,
  no copying)
- *"Insert this exact 30GB of numbers, right here, forever"* (closed-over
  constant — the data gets physically embedded in the compiled program)

Normally this is harmless for closed-over constants if they're small (a
learning rate, a boolean flag, a small lookup table) — and even for a large
one, JAX/XLA usually keep exactly one physical copy and reference it
wherever it's used, so a single closed-over 30GB array by itself shouldn't
be catastrophic. The catastrophe comes from combining this with `vmap`,
next.

---

## 5. `vmap`: batching without writing a loop

Suppose you have a function that computes one trial's energy from one set
of parameters, and you want to do this for 5 trials. Two ways to do it:

1. **Python loop:** call the function 5 times, once per trial. Correct, but
   launches 5 separate GPU programs — slow, and doesn't let the GPU exploit
   the fact that all 5 are doing "the same shape of work."
2. **`vmap`** ("**v**ectorizing **map**"): rewrite the function, automatically,
   so it operates on a *batch* of 5 inputs at once, as if you'd manually
   added a new leading axis of size 5 everywhere and used batched matrix ops
   from the start. One GPU program, one launch, much better utilization.

Think of it like an Excel spreadsheet: writing `=A1*B1` in one cell and
dragging it down 5 rows is the "loop" — same formula, executed 5 separate
times. `vmap` is closer to rewriting the formula once so it multiplies two
entire *columns* together in one shot.

**The catch:** for `vmap` to do this rewrite, every low-level operation
inside your function (matrix multiply, add, reshape, ...) needs a **batching
rule** — a recipe for "here's how to reinterpret this single-example
operation as a batched operation." Dense ops (normal matrix multiply, adds,
etc.) have fast, well-tested batching rules. Sparse ops often don't, or only
have partial ones — which brings us to the actual data structure involved.

---

## 6. Sparse arrays and `BCOO`: why your Hamiltonian isn't a normal matrix

Your Hamiltonian, as a plain (**dense**) matrix, would need to store every
entry of a 2²⁹ × 2²⁹ grid — over 10¹⁷ numbers. That's meaningless to even
attempt; there isn't enough memory on Earth. But almost all of those entries
are zero (a Pauli string only touches a handful of qubits), so instead the
code stores it as a **sparse** matrix: only the *non-zero* entries are kept,
as a list of `(row, column, value)` triples.

`BCOO` ("**B**atched **CO**O**rdinate**" format) is JAX's sparse array type
— it's literally two dense arrays: `indices` (the list of `(row, col)`
positions with something non-zero) and `data` (the corresponding values).
This is what `qu.PauliStringSum2COO` builds and what `self.fullham` actually
is.

Sparse formats are a great memory trade-off, but a genuinely awkward fit for
GPUs, which are built to crunch large, *regular*, uniform grids of numbers
in parallel — "go do this same operation to every element" is the whole
premise of GPU parallelism. Sparse data is irregular by nature (the nonzero
positions are data-dependent, different every row), so:

- Fewer sparse operations are supported at all.
- The ones that are supported often fall back to something inefficient
  internally (sometimes literally converting to dense temporarily).
- **Autodiff and `vmap` support for sparse arrays are the least mature
  parts of this stack** — tellingly, this lives under
  `jax.experimental.sparse` in JAX's own source. "Experimental" is JAX
  telling you directly: batching and gradient rules here haven't had the
  years of hardening that dense linear algebra has.

---

## 7. Autodiff, `value_and_grad`, and what "batching rule" really costs you

`K.value_and_grad(energy_fn)` asks JAX for both the energy value *and* its
gradient with respect to `params`. JAX computes gradients by walking the
jaxpr (the recorded graph of operations, from §3) backwards, applying the
calculus chain rule mechanically to each primitive operation — this is
**reverse-mode autodiff**, the same idea as backpropagation in neural nets.

For every primitive operation, JAX needs to know its derivative rule
("if `y = f(x)`, how do I turn a gradient-with-respect-to-`y` into a
gradient-with-respect-to-`x`?"). For `sparse_dense_matmul(hamiltonian, state)`
in [`sparse_expectation`](/home/s1931382/dpqc_venv/lib/python3.12/site-packages/tensorcircuit/templates/measurements.py#L177),
the backward pass needs something shaped like "Hamiltonian-transposed times
a cotangent vector" — which means either materializing the transpose of
that huge sparse matrix, or otherwise doing extra sparse bookkeeping. That's
already a second big object in play, on top of the original.

Now compound that with `vmap`: because there isn't a clean, native
batching rule for "differentiate through a sparse matmul against a
closed-over sparse constant," JAX's fallback strategy for operations it
can't batch cleanly is, roughly, to **trace the operation separately for
each element of the batch** rather than genuinely fusing them into one
batched op. Each of those 5 separate per-trial traces independently closes
over `self.fullham` and embeds its own literal copy — because remember,
from §4, closed-over data gets written into the program as a constant at
the point it's used, and if the same function body gets traced 5 times
instead of once, that's 5 separate embeddings, not one shared one.

5 trials × (a full Hamiltonian embed, roughly doubled for the forward pass
plus whatever the backward pass materializes) is exactly the multiplicative
blow-up that produced ~601GB.

---

## 8. XLA and what those cryptic log lines actually mean

**XLA** ("Accelerated Linear Algebra") is the compiler JAX hands the jaxpr
to. It's XLA's job to take the recorded graph of operations and turn it into
one fused, optimized block of GPU machine code — deciding how to lay
tensors out in memory, which operations to fuse together, when to allocate
and free buffers, and so on. This compiled form is called **HLO**
("High-Level Optimizer" representation) — that's what `gpu_hlo_schedule.cc`
in your error log refers to: XLA's scheduler, trying to plan out memory
allocation for the compiled program *before it ever runs*.

This matters for diagnosis: the crash happened at **compile time**, not
during actual training. Your progress bar was frozen at `0/1500` — no
training step had executed yet. This is a big clue in general:

- A memory error that happens **immediately, before any iterations run** →
  almost always a *compilation/tracing* problem (like this one) — the
  *program itself* is too big to even build, independent of how much
  compute you're asking it to do.
- A memory error that happens **gradually, or scales with how far training
  got** → a genuine *runtime* problem — the computation itself needs more
  memory than you have, iteration by iteration (this was closer to what
  happened in your *first* crash, the Hamiltonian-construction OOM).

The "`jit_reshape`" name in the final traceback is literally XLA telling you
which sub-program it was trying to run when it failed to allocate — a
reshape operation, almost certainly the one trying to lay out all those
embedded Hamiltonian copies contiguously in memory.

---

## 9. The full chain of events, in order

1. `ansatz.optimize()` calls `self._cost_vvag(params)` for the first time
   ([find_gs.py:317](../src/find_gs.py#L317)).
2. First call → JAX has never compiled this function before, so it must
   **trace** it: run it once with tracers standing in for `params`.
3. Tracing hits `energy_from_params`, which reads `self.fullham` — not a
   traced argument, so it becomes a **closed-over constant**.
4. The whole function is wrapped in `vmap(..., vectorized_argnums=0)` over
   5 trials, and in `value_and_grad` for the backward pass. Sparse `BCOO`
   ops don't have solid combined vmap+autodiff support, so JAX's fallback
   ends up tracing/embedding the Hamiltonian **once per trial** instead of
   sharing one copy.
5. XLA receives a jaxpr containing ~601GB of literal constant data (per the
   warning) and, while scheduling a reshape step needed to lay this out,
   tries to allocate a 280GB contiguous buffer.
6. Even the H200's 141GB isn't close to enough → `RESOURCE_EXHAUSTED`,
   crash, `0/1500` — nothing had actually started training yet.

---

## 10. General lessons / rules of thumb worth keeping

- **Large arrays that feed into `jit`/`vmap` should be explicit arguments,
  not reached via `self.something`, a closure, or a global.** Small
  closed-over constants (flags, scalars, small config) are fine. Anything
  GB-scale is a liability the moment it's inside a `vmap`.
- **`jax.experimental.*` is a warning label.** If an operation lives under
  `experimental`, assume its `vmap` and autodiff support may be incomplete
  or inefficient, and be suspicious of using it inside `vmap`/`grad` at
  scale until you've checked.
- **Where in the run a memory error happens tells you what kind of bug it
  is.** Immediate failure at iteration 0, during what looks like a "hang,"
  usually means the *compiled program* is pathologically large — a tracing
  problem. Failure partway through, or one that gets worse over time, means
  the actual *computation* needs more memory than you have.
- **Bigger hardware fixes runtime problems, not compile-time ones.** Your
  first crash (building the Hamiltonian) was legitimately a "the computation
  needs more memory than this GPU has" problem, and a bigger GPU fixed it.
  This second crash is a "the *compiled program* is accidentally quadratic
  in a way it shouldn't be" problem — no GPU you can get access to will
  reliably fix it, because it scales with `trials`.

---

## 11. Where to go next if you want to build intuition further

Search terms rather than links (JAX's docs move around, so these are safer
to look up fresh than a URL that might be stale):

- **"JAX sharp bits"** — JAX's own list of exactly this category of gotcha:
  what `jit` requires of your function, why closures over concrete values
  behave the way they do, pure-function requirements.
- **"JAX autodiff cookbook"** — the mental model for how `grad`/
  `value_and_grad` actually walk the jaxpr backward.
- **"JAX vmap batching rules"** — how `vmap` decides how to rewrite each
  primitive, and what happens when a primitive doesn't have one.
- **`jax.make_jaxpr(fn)(*args)`** — a genuinely useful debugging tool: it
  prints the traced graph *without* compiling or running it. If you ever
  suspect something is being closed over as a giant constant, this will
  show it as a literal in the printed jaxpr — much cheaper than waiting for
  an OOM to find out.
- **`jax.experimental.sparse` docs** — worth reading the "supported
  operations" / limitations section directly, since that's the boundary
  you're up against here.
