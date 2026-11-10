# Experiment 1: scaling with system size

Status: Phase 0 (tooling + smoke test) done 2026-11-10; Phase 1 pilots next. Written 2026-11-10.

## Question

How do the classical simulation cost and the accuracy of each reset method scale with system size on n×2 toric-code ladders?

Resets are held fixed: 4 resets, all in the final layer. Accuracy is measured in practice, so it includes how well each method trains, not only how noisy its energy estimate is.

## Fixed settings (every run)

| Setting | Value | Why |
|---|---|---|
| Lattices | 2×2, 3×2, 4×2, 5×2, then 6×2, 7×2, 8×2, 10×2, 12×2, ... | Ladders first, then push as far as the stopping rule allows |
| Layers | `nlayers=2` | Current default |
| Reset layer | `reset_layers=[1]` | Final layer |
| Number of resets | R = 4 | Fixed across sizes, so 3^R = 81 branches everywhere |
| Reset qubits | 4 rungs spread along the ladder (rule below) | Same physical placement idea at every size |
| Trials | 5 | Fixes memory per run, so "fits in 48 GB" has one meaning |
| Seed | one fixed seed, e.g. 1234, for all arms of a lattice | Shared parameter layout, so all four arms start from identical parameters |
| Field | h = 0 | Exact E0 = -(#stars + #plaquettes) is known for every size |
| GPU | A40 (48 GB), `--gres=gpu:a40:1` | Already the job-script default; memory limit is the stopping rule |
| Walltime | 48 h | Also the ICF-Free partition maximum |
| Forward SVD | `DPQC_SVD_FWD_ALG=qr` | Job-script default; avoids the batched-Jacobi NaNs |
| Bond dimension | per lattice and arm, from Phase 1 | Never `bond_dim=None` (its gradients are wrong) |
| Ordering | `use_optimal_ordering=True` | Bond dimension depends heavily on this |
| Learning rate | 1e-2 (Adam, default) | Unchanged |
| `track_params` | on | Needed for every post-hoc analysis below |
| `track_grads` | on | Gradient norms for the convergence analysis; host memory only |
| `track_bond_dim` | off | Its extra circuit builds inflate the GPU peak; bond dimension comes from Phase 1 instead |

## The four arms

| Folder name | Method | Flags |
|---|---|---|
| `mc_n1` | Monte Carlo, 1 trajectory per step | `use_trajectory_resets=True, n_trajectories=1` |
| `mc_n8` | Monte Carlo, 8 trajectories per step | `use_trajectory_resets=True, n_trajectories=8` |
| `purified` | Two ancillas per reset | `use_trajectory_resets=False, use_enumerated_resets=False` |
| `enumerated` | All 81 trajectories, exact ("parallel trajectories") | `use_trajectory_resets=False, use_enumerated_resets=True` |

**A "run" is one method on one lattice**, e.g. `2x2/mc_n8`. Every measurement below is made separately for each run, so each method gets its own cost and accuracy numbers at each lattice size. Within a run, the 5 trials are 5 independent trainings of that method; per-trial numbers are kept and also summarised as a mean over trials.

## Reset-qubit rule

Use the ladder's rungs (vertical edges, direction 1, the same type the code resets by default). A ladder Lx×2 has Lx rungs, at x = 0 ... Lx-1.

- **Lx ≥ 4:** the 4 rungs at x = round(linspace(0, Lx-1, 4)). For example 4×2 gives x = 0, 1, 2, 3 and 5×2 gives x = 0, 1, 3, 4.
- **Lx = 3:** all 3 rungs, plus one horizontal edge. Default choice: the top edge at x = 0 (qubit 0). **To confirm.**
- **Lx = 2:** all 4 qubits (2 rungs and 2 horizontal edges).

The chosen qubit indices are printed at start-up and saved with each run.

## Measurements

### Cost (per run)

- **Bond dimension needed**: the smallest cap at which the discarded weight stays below 1e-5. Taken as the worst case over random initial parameters and several points along the trained run. Comes from `src/diagnostics/bond_dim_requirement.py` (Phase 1, then re-checked in Phase 3).
- **Peak GPU memory**: `peak_gpu_bytes` from `optimize()`. Steady-state `gpu_live_bytes` is saved alongside it.
- **Time per step**: mean of `step_times[1:]`. Step 0 includes JIT compilation, which is reported separately.
- **Total time**: the sum of `step_times`, and the cumulative time at every logged step (for time-to-converge).

### Accuracy (per method and lattice, at every logged step, for each trial)

Each method's energy curve comes from **its own** trained parameters.

- **Purified and enumerated**: the energy logged during training is already exact, so no extra evaluation is needed.
- **mc_n1 and mc_n8**: the logged energy is a noisy estimate. After training, every saved parameter vector is re-evaluated exactly by enumerating all 81 trajectories (forward pass only, about 81× the cost of one trajectory, no gradients). This gives a noise-free training curve on the same footing as the exact methods. It runs as a separate job, so training timings are untouched.
- **Reset probability** p_reset = sin^2(σ/2) per trial, read from the parameter history. Shows whether and when each method turns its resets off.

Training curves are plotted against the exact ground-state energy E0 as a grid line; no error-to-E0 number is computed.

**Accuracy** is the exact standard deviation of the method's own energy estimate:

σ²[Ê_n] = (1/n) Σ_i p^(i) (E^(i) - E)², with E = Σ_i p^(i) E^(i)

- It uses all 81 trajectories, taken from the same enumeration at each saved parameter vector. `trajectory_table` returns p^(i) E^(i) and p^(i) for every trajectory; E^(i) = ratio where p^(i) > 0, and zero-probability trajectories drop out.
- n is the method's own: 1 for mc_n1, 8 for mc_n8. It is 0 by construction for purified and enumerated.
- Computed at every logged step, for each trial, from the method's own parameters.
- The marker colour uses σ at the final parameters, averaged over the 5 trials.
- Caveat: if training drives p_reset → 0, σ → 0 for both MC methods at the end. The per-step σ curve shows whether this happens; if it does, consider colouring by σ averaged over training instead.

### Convergence (per run)

- **Converged**: exact energy within 0.1% of E0, i.e. |E - E0| / |E0| ≤ 1e-3, where E0 = -(#stars + #plaquettes). Uses the exact energy for every method, so the Monte Carlo methods aren't judged on their noisy logged values.
- **Steps to converge** and **wall-clock time to converge**: the first logged step at which a trial meets the criterion and stays there for the rest of the run. Reported per trial, with the mean over trials.
- **Final energy**: final exact energy per trial and its mean, and the fraction of the 5 trials that converged.

## Folder layout

```
experiments/experiment_1_scaling_system_size/
  PLAN.md                     this file
  README.md                   what was run, exact settings, how to regenerate every plot
  config.json                 fixed settings + reset qubits and bond dimension per lattice/arm
  2x2/
    bond_dim_requirement.csv  Phase 1 output for this lattice (all arms)
    mc_n1/
      stdout.out, stderr.err, memlog.txt
      run.h5                  copy of the per-run HDF5 (full history, params, step_times, memory)
      checkpoint.npz          last checkpoint (survives a walltime kill)
      analysis.h5             post-hoc (MC methods only): exact energy and σ per step/trial
      metrics.csv             one row per (trial, logged step): step, cumulative seconds,
                              logged energy, exact energy (= logged energy for purified/enumerated),
                              σ (0 for purified/enumerated), p_reset, grad norm
      summary.json            one record: all cost numbers + convergence numbers
      plots/                  training history plots for this run
    mc_n8/  purified/  enumerated/   (same contents)
  3x2/  4x2/  5x2/  6x2/  ...
  comparisons/
    summary.csv               one row per (lattice, arm), from every summary.json
    cost_vs_size_bond_dim.png
    cost_vs_size_peak_gpu.png
    cost_vs_size_step_time.png
    cost_vs_size_total_time.png
    cost_vs_accuracy.png      one line per lattice, one marker per arm
    convergence_steps_vs_size.png
    convergence_time_vs_size.png
    final_error_vs_size.png
    training_curves_<lattice>.png   all four arms overlaid, exact energy vs step
```

The raw run files (`run.h5`, `checkpoint.npz`, `analysis.h5`) can be large. Add them to `.gitignore`, and keep `PLAN.md`, `README.md`, `config.json`, the CSVs, the JSONs and the plots in git.

## Plots

Style follows your usual rules: no explanations inside figures and no conclusions as titles.

1. **Cost vs system size**: one plot per cost metric (bond dimension, peak GPU memory, time per step, total time), with the same x-axis on each: lattice size with the qubit count in brackets, e.g. "3×2 (7)". One line per method, connecting the same method across sizes. Log y-axis. Marker colour = mean accuracy over the 5 trials (metric pending), with one shared colour scale and colour bar across all four plots. Runs that ran out of memory or time are drawn as hollow markers at the last size that fit.
2. **Cost vs accuracy**: x = accuracy, y = time per step (and a second version with peak memory). One line per lattice, one marker shape per arm.
3. **Convergence vs system size**: steps to converge, and wall-clock time to converge, per method. Mean with the min-max range over trials. Same x-axis labels as plot 1.
4. **Training curves per lattice**: exact energy vs step for all four methods, with a horizontal grid line at E0. Thin lines are trials, thick lines are the mean.

## Code changes

Each change is small and leaves existing runs unchanged when the new settings are off.

1. **Reset-qubit rule.**
   - Add a helper `spread_rung_reset_qubits(Lx, Ly, n_resets)` next to the toric-code lattice code. It returns the qubit indices from the rule above.
   - Add `n_spread_resets` to `run_config.py` (default `None`, meaning the current behaviour). The training runner passes the helper's output to `ToricCodeAnsatz(which_qubits_for_prob_reset=...)`. That argument already exists, and the purified, trajectory and enumerated paths all honour it.
   - Test: the helper returns 4 distinct, valid qubits for every lattice from 2×2 to 20×2, and matches the examples above.
2. **Send runs to the experiment folder.**
   - `run_plot_training_curve_tc.sh` currently always writes to `logs/<date>_<name>_<id>/`. Add an optional `DPQC_RUN_DIR` environment variable: if it is set, the script uses that folder instead.
   - `submit.sh` already turns `KEY=VALUE` into `DPQC_KEY`, so `./submit.sh RUN_DIR=experiments/.../2x2/mc_n1` works with no change there.
   - At the end of the run, copy the per-run HDF5 from `tmp/` into the run folder as `run.h5`.
3. **Post-hoc analysis script** (`src/experiments/exp1_analyse_run.py`). Given a run folder, it:
   - for mc_n1 and mc_n8 only: rebuilds an enumerated twin of the ansatz with the same lattice, reset qubits and bond dimension, and evaluates every saved parameter vector to get the exact energy and σ;
   - for purified and enumerated: takes the logged energy as the exact energy;
   - writes `analysis.h5` (MC only), `metrics.csv`, `summary.json` and the per-run `plots/`.

   It runs as its own Slurm job through `submit_diag.sh`, so its cost never mixes with training. The work is about (logged steps) × 5 trials × 81 forward passes. That is roughly what the enumerated method does every training step, minus the gradient, for 1/10 as many steps, so it is cheap relative to the training runs.
4. **Comparison script** (`src/experiments/exp1_compare.py`). Reads every `summary.json` and `metrics.csv` and writes everything in `comparisons/`.
5. **Submission script** (`experiments/experiment_1_scaling_system_size/submit_all.sh`). Loops over lattices and arms and calls `./submit.sh` with the right settings and `RUN_DIR`. Settings travel with each job, so editing code later doesn't change queued runs' parameters.
6. **Tests.**
   - The reset-qubit rule (above).
   - At 2×2, the enumerated twin's energy matches the purified energy at the same parameters (to 1e-10).
   - At 2×2, the mean and variance of 2000 sampled trajectory energies match the twin's exact E and Σ_i p^(i) (E^(i) - E)² within sampling error.

`optimize()` itself does not change.

## Phases

### Phase 0: build and smoke test
- Make code changes 1–6 and run `python -m pytest tests`.
- Smoke test: 2×2, all four arms, `maxiter=20`, writing into a scratch copy of the folder layout. Check every file is produced and the comparison plots render.

### Phase 1: bond dimension per lattice and arm
For each lattice:
1. Short pilot run per arm, 200 steps at a generous bond dimension (64), with `track_params` on.
2. Run `bond_dim_requirement` per arm on random initial parameters and on the pilot checkpoints (`--from-checkpoint`).
3. The production bond dimension for that lattice and arm is the worst case, rounded up to the next value in {8, 16, 24, 32, 48, 64, 96, 128}.

This matters because the cap, not the true rank, sets the cost: with a non-empty split config the MPS keeps min(cap, matrix size) singular values even when some are zero. Training every arm at 64 would make the cost curves reflect the cap instead of the method.

Record the results in `<lattice>/bond_dim_requirement.csv` and `config.json`.

### Phase 2: production, 2×2 to 5×2
- 4 lattices × 4 arms = 16 jobs, each with trials = 5, seed 1234, `maxiter` = 2000 (**to confirm**), `howoften_tosave=10`, and 48 h.
- Afterwards, run the post-hoc analysis on every run folder.
- Re-run `bond_dim_requirement --from-checkpoint` on the production checkpoints. If any trained point needs more than the production bond dimension, that run was truncated: redo it at the higher value.

### Phase 3: comparisons
Run `exp1_compare.py`, look at the plots, and decide whether `maxiter` was enough. A run that has not converged by the end has no convergence time to report.

### Phase 4: push n×2
- Continue with Lx = 6, 7, 8, 10, 12, 16, ... using the same pipeline (Phase 1 pilot → production → analysis).
- Each arm stops independently. An arm stops at the first lattice where it either runs out of memory on the A40 (48 GB) or doesn't finish `maxiter` steps within 48 h.
- Record the failure as a data point (`summary.json` with `status: "oom"` or `"timeout"`), so it appears as a hollow marker on the plots.
- Purified is expected to stop first; Monte Carlo and enumeration probably hit the time limit before the memory limit.

## Risks and things to watch

- **Resets in the final layer only mix in less-entangled states**, so the optimum is p_reset → 0 for every arm. The comparison is valid, but the arms will partly differ in how fast they learn to switch resets off. The p_reset history makes this visible.
- **The ordering search is random and gets weaker at large Lx.** The number of random permutations is capped at 2 million, so at large n it may not find the natural ladder order, and the bond dimension would grow because of the ordering rather than the physics. Phase 1 would show this as a jump in the needed bond dimension. If it appears, switch to a fixed ladder ordering for all arms.
- **The purified chain is longer** (N + 8 sites), and its ordering search reserves ancilla slots, so its ordering differs from the ancilla-free arms. That is part of the method's cost and is fine to compare.
- **Enumeration memory**: about 34 GB at 3×3, bond dimension 64, trials = 10. At n×2 the bond dimension is smaller and trials = 5, so it should fit for a long way, but Phase 1 will show it.
- **NaN steps**: frozen trials still count as steps. `n_nonfinite_steps` is saved per run and should be reported in `summary.json`.
- **GPU sharing**: other jobs on the same node don't affect device memory, but can affect timing. Use A40 only, and look at the spread of `step_times` for outliers.

## Open decisions

1. **4th reset qubit at 3×2**: top edge at x = 0 (default) or another edge?
2. **`maxiter`**: 2000 for everything, or chosen per lattice from the Phase 1 pilots?
3. ~~Accuracy metric and convergence criterion~~: decided (exact σ[Ê_n]; within 0.1% of E0).
4. **Folder location**: `experiments/experiment_1_scaling_system_size/` at the repo root, with raw data gitignored?
