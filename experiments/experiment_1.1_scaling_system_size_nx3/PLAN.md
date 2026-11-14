# Experiment 1.1: scaling with system size on Lx × 3 lattices

Status: plan only, written 2026-11-13. Nothing is implemented or submitted yet. The numbers in this plan come from local CPU measurements made with a scratch copy of the proposed ordering and reset rule (no repo changes). Phase 1 repeats them with the committed code.

All decisions made 2026-11-14:

1. Run both cap policies: A (at the need, 192/256) and B (below the need, 64/96).
2. Verticals-first ordering.
3. Alternating-row reset rule.
4. Qubit-matched lattice list.
5. Run A both ways: the literal failure points and a 4x3 calibration.

Next step: Phase 0 (code, tests, local smoke test). No jobs are submitted until you approve Phase 1.

## Summary

- Same study as Experiment 1: how the cost and accuracy of the four reset methods scale with system size. The difference is that it runs on Lx × 3 lattices (5·Lx − 3 qubits) instead of Lx × 2 ladders.
- Everything from Experiment 1 is kept unless it is listed under "What changes".
- **Main finding so far:** three rows need a much larger bond dimension than ladders.
  - At random init, the trajectory arm needs 128–192, against 16 for ladders.
  - Purified needs 192–256, against 24.
  - 256 is exact at every length.
- At these bond dimensions the cost per step grows roughly as χ³. With caps at the need, every method stops at 3x3 or 4x3.
- So the experiment runs two cap policies:
  - A, at the need (192/256): mostly records where the wall is, plus a completed 4x3 calibration;
  - B, below the need (64/96): the actual scaling curve, with its truncation checked on the trained states.

## What stays the same (from Experiment 1 v2)

- Toric code at h = 0, `nlayers=2`, 4 probabilistic resets, all in the final layer (`reset_layers=[1]`), one shared reset angle.
- Four arms, one Slurm job each per lattice:
  - `mc_n1`, `mc_n8`: trajectory resets with 1 or 8 trajectories.
  - `purified`: two ancillas per reset.
  - `enumerated`: all 3^4 = 81 branches.
- Enumerated keeps the coin weight p^(a) outside the amplitudes (422246e).
- 10 trials, 3000 Adam steps, learning rate 1e-2, seed 1234 shared by all arms, `howoften_tosave=10`.
- `track_params` and `track_grads` on, `track_bond_dim` off.
- Training only on A40s, with `DPQC_SVD_FWD_ALG=qr` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Accuracy is the exact σ[Ê_n] from the enumerated twin over all 81 trajectories (`exp1_analyse_run.py`). It is 0 for purified and enumerated.
- Converged means the exact energy is within 0.1% of E0 and stays there to the end. E0 = −(#stars + #plaquettes) = −(5·Lx − 2) for Lx × 3.
- Means over trials. Training curves have a grid line at E0.
- Cost metrics:
  - mean step time (excluding step 0)
  - compile time
  - total time
  - peak GPU memory
  - bond dimension needed, from `bond_dim_requirement` on the production checkpoints
- Stopping rule per method: out of GPU memory on an A40, or longer than 48 h.
- Same folder layout and plots. Plots look human-made: no explanatory text in figures, no conclusions as titles.

## What changes

### 1. Reset-qubit rule (Decision 3: alternating rows, decided 2026-11-14)

`spread_rung_reset_qubits()` only handles Ly = 2. An Lx × 3 lattice has two rows of vertical edges ("rungs"):

- row 0 joins vertex rows 0 and 1;
- row 1 joins vertex rows 1 and 2.

Every option below keeps the Experiment 1 x positions, x = round(linspace(0, Lx−1, 4)), so the resets sit at the same places along the length as on the ladders. The options differ only in which row each reset uses.

Picture of 4x3, with qubit numbers on the edges. Under the alternating rule the resets are [9], [14], [11] and [16]:

```
  o---0---o---1---o---2---o
  |       |       |       |
 [9]     10     [11]     12      rung row 0 (between vertex rows 0 and 1)
  |       |       |       |
  o---3---o---4---o---5---o
  |       |       |       |
  13    [14]     15     [16]     rung row 1 (between vertex rows 1 and 2)
  |       |       |       |
  o---6---o---7---o---8---o
```

Only the rungs, at 10x3 (R = reset). The x positions are the same 0, 3, 6, 9 as Experiment 1's 10x2:

```
                 x:  0 1 2 3 4 5 6 7 8 9
Exp 1, 10x2 rungs:   R . . R . . R . . R

A alternating, row 0: R . . . . . R . . .
               row 1: . . . R . . . . . R

B row 1,       row 0: . . . . . . . . . .
               row 1: R . . R . . R . . R
```

At 3x3 the x positions are 0, 1, 1, 2. Rounding puts two resets at x = 1:

- A puts them in different rows, so all four qubits are distinct.
- B and C have only one row, so they get a duplicate and need an ad-hoc fourth qubit.

| Option | Rule | Comment |
|---|---|---|
| **A. Alternating (recommended)** | Reset k uses row k mod 2, so the rows go 0, 1, 0, 1 | One rule from 3x3 upwards with no special cases: at 3x3 the x positions are 0, 1, 1, 2 and the rows make them distinct. Uses both rows, which is the natural 2D version of "spread". |
| B. Bottom row | All four in row 1 | Closest to Experiment 1 in circuit terms. Each claw is centred on a plaquette's bottom edge. Row-1 rungs are legs of plaquettes whose centre edge is the lattice boundary, as the ladder rungs were. Needs a special case at 3x3, which has only 3 rungs per row. |
| C. Top row | All four in row 0 | Same as B, but these plaquettes' centre edge is the middle row. Needs the same 3x3 special case. |

Qubit indices picked by each option (vertical edge (x, y) is qubit y·Lx + x + 3(Lx−1)):

| Lattice | x positions | A (alternating) | B (row 1) | C (row 0) |
|---|---|---|---|---|
| 3x3 | 0, 1, 1, 2 | 6, 7, 10, 11 | 9, 10, 11 + one more (e.g. 7) | 6, 7, 8 + one more (e.g. 10) |
| 4x3 | 0, 1, 2, 3 | 9, 11, 14, 16 | 13, 14, 15, 16 | 9, 10, 11, 12 |
| 5x3 | 0, 1, 3, 4 | 12, 15, 18, 21 | 17, 18, 20, 21 | 12, 13, 15, 16 |
| 6x3 | 0, 2, 3, 5 | 15, 18, 23, 26 | 21, 23, 24, 26 | 15, 17, 18, 20 |
| 7x3 | 0, 2, 4, 6 | 18, 22, 27, 31 | 25, 27, 29, 31 | 18, 20, 22, 24 |
| 8x3 | 0, 2, 5, 7 | 21, 26, 31, 36 | 29, 31, 34, 36 | 21, 23, 26, 28 |
| 9x3 | 0, 3, 5, 8 | 24, 29, 36, 41 | 33, 36, 38, 41 | 24, 27, 29, 32 |
| 10x3 | 0, 3, 6, 9 | 27, 33, 40, 46 | 37, 40, 43, 46 | 27, 30, 33, 36 |

All three options give the same maximum claw distance for purified (+2 over the ancilla-free chain; see section 2). They are all equally cheap, so this choice is physics, not cost.

Code: add the Ly = 3 branch to `spread_rung_reset_qubits(Lx, Ly, n_resets)`. Ly = 2 stays byte-for-byte the same, and the existing tests keep passing. Ly ≥ 4 still raises. `n_spread_resets` in `run_config.py`, the runner and `bond_dim_requirement --n-spread-resets` all go through this one function, so nothing else changes.

### 2. MPS ordering (Decision 2: verticals first, decided 2026-11-14)

The ladder ordering puts the qubits into the chain column by column. Within each column the order is the top horizontal edge, then the rung, then the bottom horizontal edge. For Lx × 3 there are two natural ways to lay out a column (5 qubits: h0, h1, h2 horizontal and v0, v1 vertical):

- **rows in order**: h0 v0 h1 v1 h2. This is the generalisation you described, and it reduces to the ladder ordering at Ly = 2.
- **verticals first**: v0 v1 h0 h1 h2. I found it with a brute-force search over all 120 in-column orders, looking for the smallest maximum claw distance.

Measured on the lattice alone:

| | rows in order | verticals first | ladder (Ly = 2, for reference) |
|---|---|---|---|
| max claw distance, ancilla-free | 4 | 3 | 2 |
| max claw distance, purified (rule A) | 6 | 5 (7 at 3x3) | 6 |
| SVD-bearing applications per qubit per layer (48x3 / 80x2) | 4.34 | 3.57 | 2.32 |

- The claw distances are the same for every length from 4x3 to 48x3, so they are bounded.
- Purified with verticals first reaches 7 only at 3x3. There, two of the alternating resets share column 1, so two ancilla pairs sit inside one claw.
- "Applications" counts how many SVD-bearing two-qubit gates the MPS applies. TensorCircuit applies a gate whose two sites are d apart as 2(d − 1) SWAPs plus the gate, and each one is an SVD. So this number sets both the step time and the size of the compiled program.

**Why verticals first:**

- It needs less bond dimension at every size measured (table in section 3).
- It does about 18% less SVD work per step.
- Ly = 2 keeps the existing ladder order exactly. The rule is: Ly = 2 uses the ladder pattern, Ly ≥ 3 uses verticals first. This is documented in the docstring, and a test pins the Ly = 2 order.

Code:

- Generalise `ToricCodeAnsatz._ladder_order()` to any Ly ≥ 2 and keep the flag name `use_ladder_ordering`. Experiment 1's `run_config.json` files and the `DPQC_USE_LADDER_ORDERING` variable already use that name. The docstring will say it is a column-by-column ordering that, for Ly = 2, is the ladder ordering.
- `use_ladder_ordering` is already part of the ansatz's hash, and Lx × 3 with the flag on raised an error until now. So no existing cached result or HDF5 key changes meaning.

Tests, in the same style as `tests/test_exp1_tools.py`, in a new `tests/test_exp1_1_tools.py`:

- For Lx in {3, 4, 5, 12, 30, 48} × 3: the order is a permutation of all qubits, and the maximum claw distance is ≤ 3 ancilla-free and ≤ 5 purified (7 at 3x3), independent of Lx.
- The chain length is nq (ancilla-free) or nq + 8 (purified).
- At 3x3, the column ordering and the random-search ordering give the same energy at the same parameters, to 1e-10:
  - enumerated at bond dimension 64, which is exact for 12 qubits;
  - purified at a bond dimension where the truncation is checked to be zero.
- Ly = 2: `_ladder_order()` returns exactly the current order for 2x2 to 30x2. The existing ladder tests stay unchanged except `test_ladder_order_rejects_non_ladder`, which moves to Ly = 1 or 4.

### 3. Bond dimension (Decision 1: run both A and B, decided 2026-11-14)

#### Why three rows need so much more

The bond dimension at a cut between two columns is bounded by 4^(number of two-qubit gates crossing that cut).

- On a ladder, one claw per layer crosses each column boundary. Two layers give 4² = **16**, which is exactly the need Experiment 1 measured, flat with length.
- On Lx × 3, two claws per layer cross each boundary, giving 4⁴ = **256**.

The measurements agree: 256 is exact (discarded weight about 1e-14) at every Lx × 3 size measured.

Below that bound the Schmidt spectrum has a tail. The discarded weight is a sum over every cut in the chain, so the 1e-5 need creeps up slowly with length, from 128 at 8x3 to 192 at 10x3. On ladders it was flat because the bound itself was hit.

#### Measured at random init

The measurement used `bond_dim_requirement` at seed 2026 with normalisation off and alternating resets. The table shows the first cap with discarded weight ≤ 1e-5. In brackets is the discarded weight one rung below that cap.

| Lattice | traj, verticals first | traj, rows in order | purified, verticals first | purified, rows in order |
|---|---|---|---|---|
| 4x3 | 128 (2.4e-5 at 96) | 192 (1.9e-5 at 128) | 192 (2.8e-5 at 128) | 192 (1.5e-4 at 128) |
| 5x3 | 128 (1.1e-5 at 96) | 128 (4.2e-5 at 96) | 128 (9.2e-5 at 96) | 192 (2.0e-5 at 128) |
| 6x3 | 128 (4.8e-5 at 96) | 192 (2.7e-5 at 128) | 192 (3.9e-5 at 128) | 256 (1.0e-5 at 192) |
| 8x3 | 128 (1.0e-4 at 96) | 192 (8.3e-5 at 128) | — | — |
| 10x3 | 192 (1.2e-4 at 128) | 256 (1.3e-5 at 192) | 256 (4.0e-5 at 192) | — |
| 20x3 | 192 (5.4e-5 at 128) | 192 (2.8e-4 at 128) | 192 (1.2e-4 at 128) | — |

So from 10x3 to 20x3 the need is roughly flat: 192 for traj and enumerated, 192–256 for purified. The 1e-5 gate passes at most one rung later for longer chains.

- Trajectory samples all give the same discarded weight. All truncation happens in the claw layers, which come before the resets. A reset is a single-site operation at the orthogonality centre, so it costs no SVD.
- For the same reason, enumerated has exactly the same need as the trajectory arm.

Discarded weight at 10x3 (verticals first) at lower caps:

| Cap | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|
| traj | 6.8e-2 | 1.8e-2 | 5.7e-3 | 7.6e-4 | 1.2e-4 |
| purified | 9.1e-2 | 2.8e-2 | 1.1e-2 | 2.2e-3 | 5.4e-4 |

At 20x3 (verticals first), traj discards 7.7e-3 at 64 and 6.1e-4 at 96. Purified discards 9.7e-4 at 96.

#### Options

A and B were chosen. C and D are kept here for the record.

| | Caps (MC/enumerated, purified) | What it gives | What it costs |
|---|---|---|---|
| **A. At the need** | 192, 256 | Truncation below 1e-5 at init, as in Experiment 1 | Every method stops by about 4x3 (see the cost table). It shows where the wall is and little else. |
| **B. Below the need** | 64, 96 | MC reaches about 17x3–24x3, purified about 8x3, enumerated about 5x3 | At init, 0.6% (traj) and 0.2% (purified at 96) of the weight is discarded at 10x3, and 0.8% (traj) at 20x3. |
| C. Experiment 1 caps | 32, 48 | Cheapest. Same χ as Experiment 1, so the cost difference is pure geometry. | Heavy truncation at init: 6.8% (traj at 32) and 2.8% (purified at 48) at 10x3. |
| D. Small-angle init plus low caps | 32–64 | The init state is close to a product state. If training stays low-entanglement, the caps are at or above the need throughout. | Changes the initialisation relative to Experiment 1, so the training curves are no longer like-for-like. |

Why B is defensible:

- `normalize_state=True` (the production setting) renormalises the truncated state. Every logged energy is therefore the energy of a valid state, so it is an upper bound on E0. "Converged within 0.1% of E0" stays a true statement about the trained simulation.
- The trained endpoint is a toric-code ground state. Its rank across a column cut is tiny (2–4), so the final states should fit well within 64.
- Phase 3 checks this with `bond_dim_requirement --from-checkpoint` on the production checkpoints at a higher cap. If a final state needs more than its cap, the summary says so.
- What B changes: early in training, every arm is optimising a truncated version of the circuit, not the exact one. The comparison between arms is still at an equal cap for the three ancilla-free arms. Purified keeps Experiment 1's 1.5× ratio.

#### Running both (Decision 1) and how to run A (Decision 5)

- A and B are two separate sets of runs, each with its own folder level (section 4).
- **What A will produce, by the estimates:**
  - At 3x3 the ancilla-free arms are identical under A and B: 64 is already exact for 12 qubits, and a bond can never exceed its matrix size. They run once, under B, and A's 3x3 folders link to them.
  - Beyond that, A is predicted to complete nothing:
    - MC at 4x3 is about 126 h (the model is pessimistic at small sizes, but not by 2.6×);
    - purified at 256 is over 48 h even at 3x3;
    - enumerated is out of GPU memory from 4x3.
  - So A's results are its failure points, plus the partial training saved in the checkpoints of runs that hit walltime. Those partial curves are a direct check on B: trial k starts from the same parameters in both, so comparing their first ~1000 steps shows what B's truncation does to training.
- **Decision 5 (decided 2026-11-14): run A both ways.**
  - (a) **Literal.** 10 trials, 3000 steps, 48 h, in `caps_192_256/<lattice>/<method>/`.
    - Submit only where Phase 1's measured step time says the run fits, plus the first size that does not fit, once, to record the wall.
    - Expected: 4x3 MC n1/n8 and 3x3 purified time out (about 3 × 48 A40-hours), and 4x3 enumerated is a quick OOM.
  - (b) **Calibration.** A at 4x3 for MC n1 and purified, with fewer trials and/or steps chosen from Phase 1 so the runs finish inside 48 h.
    - Folders: `caps_192_256/4x3/mc_n1_calib/` and `purified_calib/`. `exp1_compare` ignores folder names that are not method names, so these never enter the stopping-rule plots.
    - They are compared trial by trial with the same trials under B. With this JAX version, the first k rows of a `(10, nparams)` uniform draw from seed 1234 equal a `(k, nparams)` draw from the same seed (checked for one shape), so trial k starts from the same parameters. A test asserts this for the real parameter count.
    - A small extra plot, `comparisons/calibration_4x3.png`, shows exact energy against step for A and B on the shared trials.

### 4. Folder and scripts

New folder `experiments/experiment_1.1_scaling_system_size_nx3/` with the same layout as Experiment 1:

```
PLAN.md  submit_all.sh  analyse_all.sh
bd_requirement/                     Phase 1 measurements (CSV)
timing/                             Phase 1 A40 bench (CSV)
caps_64_96/                         policy B
  config.json
  <lattice>/<method>/               stdout.out, stderr.err, memlog.txt, run.h5, checkpoint.npz,
                                    run_config.json, analysis.h5 (MC), metrics.csv, summary.json, plots/
  comparisons/                      summary.csv, the Experiment 1 plot set, plus vs_exp1_*.png
caps_192_256/                       policy A, same layout
                                    (3x3/{mc_n1,mc_n8,enumerated} link to caps_64_96/3x3/)
comparisons/                        A vs B vs Experiment 1 overlays
```

- Each `caps_*` folder is a complete experiment folder in the Experiment 1 sense, so `exp1_compare --exp-dir` works on it unchanged.
- **Experiment 1's files are not edited.**
- The new `submit_all.sh` and `analyse_all.sh` are copies of Experiment 1's. Each starts with a comment listing exactly what differs:
  - `EXP` path, plus a `POLICY=caps_64_96 | caps_192_256` argument that picks the subfolder and the caps
  - lattice list
  - `walltime_for()` table
  - job names `exp11_<policy>_<kind>_<lattice>_<method>`, so the two policies and Experiment 1 can never collide in `squeue` matching
  - `MEM=160G` set automatically from 24x3 upwards
  - `KIND=calib`, for the 4x3 calibration runs (`<method>_calib/` folders, trial and step counts from config.json)
- I chose copies over making Experiment 1's scripts generic. Experiment 1's 80x2 jobs and analyses are still running, and copying means its scripts never change under them.
- On the Python side, only `exp1_compare.py` changes. It gains an option `--reference-exp-dir experiments/experiment_1_scaling_system_size` (repeatable, with a label per folder, so A, B and Experiment 1 can share one plot), which overlays the reference on the cost-vs-size and convergence plots:
  - same colour per method;
  - solid lines for × 3, dashed for × 2;
  - legend entries such as "MC, n=1 (L×3)".
- `exp1_analyse_run.py` already works on any run folder (it reads `run_config.json`). `exp1_common.EXP_DIR` stays as Experiment 1's default.
- Tests:
  - `exp1_compare` runs on a scratch folder holding one Lx × 3 and one Lx × 2 summary;
  - the reference overlay is drawn;
  - a summary from another experiment's folder is never counted twice.

### 5. Lattice sizes (Decision 4: qubit-matched, decided 2026-11-14)

Production list: 3x3, 4x3, 5x3, 6x3, 8x3, 12x3, 17x3, 24x3, 34x3, 48x3. Each arm stops at its first failure.

Lx × 3 has 5·Lx − 3 qubits; Lx × 2 had 3·Lx − 2. A lattice with the same length therefore has many more qubits with 3 rows: 10x2 is 28 qubits, 10x3 is 47. "Compare with Experiment 1" can mean two different things:

- **at the same number of qubits**: is a 3-row lattice dearer than a 2-row one of the same size? The ×3 lattice is then shorter.
- **at the same length**: what does adding a third row cost? The ×3 lattice then has more qubits.

The two lists below pick sizes for one reading or the other.

Qubit-matched: each ×3 size has an Experiment 1 run with the same, or nearly the same, qubit count.

```
Lx×3  qubits   Exp 1 partner
 5x3    22      8x2 (22)
 6x3    27     10x2 (28)
 8x3    37     14x2 (40)
12x3    57     20x2 (58)
17x3    82     28x2 (82)
24x3   117     40x2 (118)
34x3   167     56x2 (166)
48x3   237     80x2 (238)
```

Lx-matched: each ×3 size has the Experiment 1 run of the same length.

```
Lx×3  qubits   Exp 1 partner
 8x3    37      8x2 (22)
10x3    47     10x2 (28)
14x3    67     14x2 (40)
20x3    97     20x2 (58)
28x3   137     28x2 (82)
40x3   197     40x2 (118)
```

Both lists start with 3x3, 4x3, 5x3 and 6x3. The qubit-matched list was chosen: on the log qubit axis, each Lx × 3 point then sits at the same x as its Experiment 1 partner. The larger sizes only get submitted for the arms that have not yet hit the stopping rule.

## Cost estimates

### Model

Calibrated on Experiment 1 v2, an A40 with 10 trials and fused Cartan blocks:

- **Step time** = (SVD-bearing applications per step) × (time per batched SVD at the cap).
  - The applications per step come from the claw distances above.
  - Time per batched SVD: 9.6 ms at χ=32 (10x2), 49 ms at χ=64 (3x3 bench), 0.41 s at χ=128 (4x3 bench).
  - The χ=64 and χ=128 points are from August benches that used the old Jacobi SVD and chains whose bonds were only partly at the cap. Treat the estimates as ±2×. Phase 1 replaces them with measurements.
- **Compile time** = 370 s × (applications/123)^2.18. This fits Experiment 1's 10x2–56x2 within 30%, and the 3x3/4x3 benches within 30%. It does not depend much on χ.
- **Host RAM** ≈ 2 GB + 0.06 GB per application. Experiment 1 measured 19 GB at 82 qubits, 33 GB at 118, 49 GB at 166, and more than 64 GB at 238.
- **GPU memory** grows with applications × χ². For enumerated, it is about 55× MC n1, calibrated to the 3x3 enumerated bench (16.6 GiB at χ=64).
- **Other arms:**
  - MC n8 was 1.1× MC n1 and enumerated 1.9× in Experiment 1, when the GPU was latency-bound at χ=32.
  - At χ ≥ 64 the SVDs become compute-bound. They are complex128 on an A40, whose FP64 rate is 1/64 of its FP32 rate. In that regime n8 and enumerated could approach 8× and 81× MC n1.
  - The 3x3 bench at χ=64 gave enumerated 2.3×. Phase 1 measures this properly.

### Estimates

Verticals-first ordering, 3000 steps, 10 trials. **Bold** means over 48 h or over 44 GiB.

| Lattice | Qubits | Experiment 1 counterpart (MC n1 total) | Compile | Host RAM | B: MC n1 @64 | B: Purified @96 | B: Enumerated GPU @64 | A: MC n1 @192 | A: Purified @256 | A: Enumerated GPU @192 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3x3 | 12 | — | <0.1 h | 9 GB | 3 h | 16 h | 16 GiB | 3 h (64 is exact) | **308 h** | 16 GiB (64 is exact) |
| 4x3 | 17 | — | <0.1 h | 11 GB | 4 h | 21 h | 24 GiB | **126 h** | **407 h** | **219 GiB** |
| 5x3 | 22 | 8x2: 0.8 h | 0.1 h | 13 GB | 6 h | 26 h | 32 GiB | **168 h** | **506 h** | **291 GiB** |
| 6x3 | 27 | 10x2: 1.1 h | 0.2 h | 15 GB | 8 h | 32 h | 40 GiB | **210 h** | **605 h** | **364 GiB** |
| 8x3 | 37 | 14x2: 1.7 h | 0.5 h | 20 GB | 11 h | 42 h | **57 GiB** | **294 h** | **804 h** | **510 GiB** |
| 12x3 | 57 | 20x2: 2.7 h | 1.3 h | 28 GB | 17 h | **63 h** | **89 GiB** | **463 h** | **1201 h** | **802 GiB** |
| 17x3 | 82 | 28x2: 4.2 h | 3.0 h | 39 GB | 26 h | **91 h** | **130 GiB** | **675 h** | **1697 h** | **1166 GiB** |
| 24x3 | 117 | 40x2: 7.7 h | 6.6 h | 54 GB | 40 h | **130 h** | **186 GiB** | **973 h** | **2394 h** | **1676 GiB** |
| 34x3 | 167 | 56x2: 12.4 h | 14.4 h | 76 GB | **63 h** | **189 h** | **267 GiB** | **1400 h** | **3393 h** | **2405 GiB** |
| 48x3 | 237 | 80x2: running | 31.2 h | 106 GB | **100 h** | **278 h** | **381 GiB** | **2005 h** | **4796 h** | **3425 GiB** |

At the small sizes the model is pessimistic, because bonds near the chain ends sit below the cap (3x3's 16 h for purified is probably a few hours). At large sizes it may be optimistic.

### Where each method is expected to stop

Under option B, caps 64/96:

- **MC n1**: about 24x3 (40 h, borderline). 34x3 is over 48 h.
- **MC n8**: between 17x3 and 24x3 if it stays near 1–2× MC n1. Much earlier if it becomes compute-bound.
- **Purified**: about 8x3. It is 1.5× the cap, so about 3.4× the cost per SVD, on a slightly longer chain.
- **Enumerated**: out of GPU memory at about 6x3–8x3 (Experiment 1: 40x2, 118 qubits).
- **Host RAM**: MEM=160G is needed from about 24x3 (54 GB estimated; Experiment 1 needed more than 64 GB at 238 qubits). Above 34x3, compile alone takes 14–31 h, so 48x3 is out for every method on time grounds.

Under option A, caps 192/256:

- MC n1 and n8: only 3x3, which is shared with B. 4x3 times out.
- Purified: times out at 3x3.
- Enumerated: only 3x3 (shared), then out of GPU memory at 4x3.

See Decision 5.

## Phases

### Phase 0: code, tests, local smoke test (no jobs)

1. Code changes from sections 1, 2 and 4. Small commits, each with its tests.
2. `reset_impl_bench.py` gains `--ladder-ordering` and `--n-spread-resets` (`bond_dim_requirement` already has both), so Phase 1 can time the real configuration.
3. Run `python -m pytest tests`.
4. Smoke test on CPU at 3x3:
   - all four arms, `maxiter=20`, 2 trials, cap 64, writing into a scratch copy of the folder layout;
   - then `exp1_analyse_run` and `exp1_compare --reference-exp-dir ...`;
   - check every file is produced and the plots render.

### Phase 1: measurements (decision gate; needs your approval to submit)

1. **Bond dimension needed** at 10x3 and 20x3 with the committed code, for traj and purified, saved to `bd_requirement/`.
   - Runs locally on CPU: each rung takes under a minute even at χ=512.
   - It repeats the scratch numbers above. Enumerated has the same need as traj (same pre-reset state), so one enumerated check at 10x3 is enough.
2. **A40 timing bench**: one diagnostic job through `submit_diag.sh`.
   - `reset_impl_bench`, 10 trials, all four arms:
     - B (64/96) at 6x3 and 12x3;
     - A (192/256) at 3x3 (purified only) and 4x3.
   - It records warm step time, compile time, peak GPU memory and host RSS.
   - The A configurations at 4x3 are expected to be slow or out of memory. The bench records that as a row and moves on.
   - Estimate: 4–6 h, 08:00:00 requested.
3. Replace the estimate table with measured numbers. Fix the caps, lattice list, walltimes and MEM. Fill `config.json`, then show you the final table before any production submission.

### Phase 2: production

- Run `/code-review high` on the branch before submitting.
- Then submit, smallest lattices first:
  - `submit_all.sh POLICY=caps_64_96 KIND=production`
  - `submit_all.sh POLICY=caps_192_256 KIND=production LATTICES=...` (only the sizes Phase 1 says fit, plus the first that does not)
  - `submit_all.sh POLICY=caps_192_256 KIND=calib`
- Each arm stops independently at the first size that hits the stopping rule. Failures are recorded as data points (`oom`, `oom_host`, `timeout`), as in Experiment 1.

### Phase 3: analysis and comparison

1. `analyse_all.sh`: purified, enumerated and failed runs locally; MC analyses on any GPU.
   - The MC analysis re-evaluates every saved parameter vector over all 81 branches at the run's cap.
   - At χ=64 on three rows this costs much more than in Experiment 1: roughly one enumerated forward pass per 10 vectors, about 2–20 h per MC run at 10x3–24x3.
   - If Phase 1 timing shows that is too slow, there are two fallbacks:
     - (a) Evaluate every 5th snapshot (every 50 steps) plus the last. The convergence-step resolution becomes 50.
     - (b) A twin that builds the shared pre-reset state once and applies the 81 Kraus patterns to it. That is exact here because the resets come after every SVD. It would be checked against `trajectory_table` to 1e-10 before use.
     I'd ask you before using either.
2. `bond_dim_requirement --from-checkpoint` on the production checkpoints (start, middle and end of trial 0, plus the final state of every trial) at a cap above production. This reports the need along the real training path, and whether the final energies are free of truncation.
3. `exp1_compare` on each policy folder, with Experiment 1 as the reference.
   - A top-level overlay of B, A and Experiment 1.
   - The 4x3 calibration plot.
4. Write a summary in the Experiment 1 style:
   - cost, accuracy, training quality, and where each method stopped;
   - each compared directly with the Lx × 2 results at matched qubit count;
   - plus what the larger bond dimension did to each method.

## Risks and things to watch

- **The cap sets the cost, and here the need is above the cap** (option B). The early-training truncation is reported, not hidden. Phase 3's checkpoint check is what makes the final energies trustworthy.
- **A40 FP64.** At χ ≥ 64 the per-step cost is dominated by complex128 SVDs, and the A40's FP64 rate is about 0.6 TFLOPS. The cost curves will partly measure the card. That is still comparable with Experiment 1, which also ran on A40s, but worth a sentence in the write-up.
- **Enumerated's memory is partly an implementation artifact.** All 81 branches share one pre-reset MPS (the resets come after every SVD), but the implementation rebuilds each branch from scratch. Experiment 1 measured the same implementation, so it stays as is here for comparability. A shared-prefix enumerated method would be far cheaper; it is a separate question.
- **Discarded weight grows with length at a fixed cap.** It is summed over cuts. At 64 it is about 0.6% at 10x3 at init and will be roughly proportional to Lx beyond that. The Phase 3 checkpoint check covers this.
- **Host RAM and compile time** grow with the program size, about 1.5× Experiment 1's per qubit. MEM=160G from 24x3 upwards, and walltimes are set from Phase 1's measured compile times.
- **Small-lattice estimates are pessimistic**, large-lattice ones possibly optimistic. Nothing big is submitted before Phase 1 replaces them.

## Decisions needed

1. ~~Bond-dimension policy~~: decided 2026-11-14, run both A (192/256) and B (64/96).
2. ~~In-column ordering~~: decided 2026-11-14, verticals first.
3. ~~Reset-qubit rule~~: decided 2026-11-14, alternating rows.
4. ~~Lattice list~~: decided 2026-11-14, qubit-matched.
5. ~~How to run policy A~~: decided 2026-11-14, both (a) literal and (b) the 4x3 calibration.

Still to settle after Phase 1, with your approval: exact caps (if the A40 bench changes the picture), walltimes, MEM, and the calibration's trial and step counts.

## Recommended effort for the implementation

**High.** The code is moderate in size: one ordering function, one reset-rule branch, two copied scripts, an overlay option, two bench flags and tests. But a mistake in the ordering or the reset rule silently changes every expensive run, so it is worth careful tests and the `/code-review high` pass before Phase 2. xhigh is not needed.

## Your uncommitted files (left alone)

- `experiments/experiment_1_scaling_system_size/80x2/{mc_n1,mc_n8,purified}/memlog.txt`: modified by the running 80x2 jobs.
- `pyrightconfig.json`: untracked.
