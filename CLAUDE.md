# CLAUDE.md

## Project overview

Code for the paper [arXiv:2411.05760](https://arxiv.org/abs/2411.05760) — Dynamic Parameterized Quantum Circuits (DPQCs): variational quantum circuits with intermediate measurements, resets, and feedforward. Used for ground state finding and Gibbs state preparation of quantum Hamiltonians (toric code, transverse-field Ising, XY model).

## Tech stack

- **Simulation**: TensorCircuit (JAX backend) for circuit construction and differentiation
- **Tensor contraction**: cotengra for optimized contraction ordering
- **Optimization**: optax (Adam), JAX autodiff (vmap + JIT)
- **Hamiltonians**: Qiskit (SparsePauliOp, Pauli algebra)
- **Storage**: HDF5 (h5py) for results, CSV for training logs
- **HPC**: Slurm job scripts (`#SBATCH`, submitted via `submit.sh` / `submit_diag.sh`); legacy LSF/Dask support in `lsf_job_runner.py`
- **Python 3.10+**, with CUDA/GPU support via `environment.yml`

## Repository structure

```
src/
  find_gs.py                  # VariationalAnsatz base class + optimization loop
  find_gibbs_fidelity.py      # Gibbs state prep via infidelity minimization
  find_gibbs_varqite.py       # Gibbs state prep via VarQITE
  lsf_job_runner.py           # HPC job submission (LSF/Dask)
  utilities/
    ansatz_classes.py          # ToricCodeAnsatz, OneDBrickwork (concrete subclasses)
    generate_ansatz.py         # All circuit construction functions
    generate_toric_code_hamiltonian.py  # ToricCode lattice + Hamiltonian
    generate_ising_hamiltonian.py       # OneDimTFIM + helpers
    generate_XY_hamiltonian.py          # XY model Hamiltonian
    generate_ghz_hamiltonian.py         # GHZ parent Hamiltonian
    gibbs_varqite.py           # VarQTE (variational imaginary time evolution)
    tc_grads.py                # Gradient and QGT functions via TensorCircuit autodiff
    result_saver.py            # HDF5 result saving/loading
  examples/
    find_gs_tc_example.py      # Ground state demo (toric code)
    find_gibbs_fidelity_examples.py   # Gibbs state via fidelity (TFIM)
    find_gibbs_varqite_example.py     # Gibbs state via VarQITE (XY)
    plot_training_curve_tc.py  # Main experiment runner + plotting
    run_config.py              # Run parameters for plot_training_curve_tc (overridable via DPQC_* env vars)
    plot_bond_dim_by_reset_config.py, plot_reset_branch_evolution.py  # Reset/bond-dim plots
    test.py                    # Old NaN/SVD playground; kept because gate_mode_training_ablation imports from it
  diagnostics/                 # One-off investigation scripts (NOT imported by production code)
    mps_complexity.py          # Shared library for the complexity/cost study
    bond_dim_requirement.py, reset_impl_bench.py   # Bond dim needed / per-step cost per reset mode (traj, pur, enum, none)
    plot_three_resets.py, collect_runs.py          # Figures + run collection for the reset comparison
    svd_call_counting.py       # tap_backends() helper used by the benchmarks
    ...                        # Concluded NaN/SVD + normalisation studies live at git tag archive/nan-svd-normalisation
tests/                         # pytest suite: reset implementations, MPS gate paths, numerics
submit.sh -> run_plot_training_curve_tc.sh     # Training runs (parameters travel with the job)
submit_diag.sh -> run_diag.sh                  # Any diagnostic command as a Slurm job
job_memlog.sh                  # Per-job memory sampler sourced by the run_*.sh scripts
```

## Key architecture

```
VariationalAnsatz (abstract base, find_gs.py)
  ├── ToricCodeAnsatz    (ansatz_classes.py)  — uses ToricCode + construct_*_toriccodelattice*
  └── OneDBrickwork      (ansatz_classes.py)  — uses OneDimTFIM + construct_*_brickwork
```

Subclasses implement `_circuit(params)`, `_hamiltonian_terms()`, and `get_full_hamiltonian()`. The base class handles optimization, JIT compilation, vmapping, and result saving.

## How to run

```bash
# Activate the venv (same one every run_*.sh job script uses)
source /home/s1931382/dpqc_venv/bin/activate

# Run examples directly
python -m src.examples.find_gs_tc_example
python -m src.examples.plot_training_curve_tc

# Tests
python -m pytest tests

# Submit to Slurm (from the login node)
./submit.sh LX=3 LY=3 BOND_DIM=64              # training run; see header of submit.sh
./submit_diag.sh NAME=bdreq TIME=08:00:00 -- python -m src.diagnostics.bond_dim_requirement --lattices 3x3
```

Each job writes to `logs/<date>_<time>_<jobname>_<jobid>/` (stdout.out, stderr.err, memlog.txt, plots/). A run finished cleanly iff stdout.out ends with `Job finished with exit code: 0`.

## Common patterns

- Circuits are built with `tc.Circuit` (state vector) or `tc.MPSCircuit` (MPS approximation), controlled by `split_conf` dict and `use_mps` flags.
- Two-qubit gates use the Cartan decomposition block: 2x single-qubit Ry-Rz-Ry + Rxx/Ryy/Rzz.
- Resets have three implementations, chosen by flags on `ToricCodeAnsatz`: trajectory (`use_trajectory_resets=True`, the dataclass default; ancilla-free, samples one Kraus branch), enumerated (`use_enumerated_resets=True`; sums all 3^R branches exactly), and purified (both False; two ancillas per reset). `VariationalAnsatz.reset_mode` reports which.
- Purified resets use ancilla qubits with parametrized Ry + Toffoli (CCX) gates; on MPS circuits, Toffoli is decomposed into 1- and 2-qubit gates via `decompose_ccx()`.
- Energy is computed via `sparse_expectation(qc, fullham)` (sparse mode) or term-by-term `qc.expectation()`.
- All optimization uses JAX vmap over trials + JIT + Adam (optax).

## Development notes

- `jax_enable_x64` is set globally in multiple files — this is required for numerical precision.
- `split_conf = {}` in `generate_ansatz.py` controls tensor splitting; set it to `{"max_singular_values": N}` for MPS truncation.
- Results go to `results/` (master HDF5) and `tmp/` (per-run HDF5; each is duplicated in the master, so `tmp/` is safe to clear). Plots go to `outputs/` or `$DPQC_OUTDIR`.
- `jax_cache/` and `ctg_cache/` live at the repo root (paths are anchored in code); both are safe to delete.
- Thread counts are pinned (4, via OMP/OPENBLAS/MKL_NUM_THREADS) in job scripts to avoid contention on shared nodes.
