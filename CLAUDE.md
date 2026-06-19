# CLAUDE.md

## Project overview

Code for the paper [arXiv:2411.05760](https://arxiv.org/abs/2411.05760) — Dynamic Parameterized Quantum Circuits (DPQCs): variational quantum circuits with intermediate measurements, resets, and feedforward. Used for ground state finding and Gibbs state preparation of quantum Hamiltonians (toric code, transverse-field Ising, XY model).

## Tech stack

- **Simulation**: TensorCircuit (JAX backend) for circuit construction and differentiation
- **Tensor contraction**: cotengra for optimized contraction ordering
- **Optimization**: optax (Adam), JAX autodiff (vmap + JIT)
- **Hamiltonians**: Qiskit (SparsePauliOp, Pauli algebra)
- **Storage**: HDF5 (h5py) for results, CSV for training logs
- **HPC**: Grid Engine (SGE) job scripts, LSF/Dask support via `lsf_job_runner.py`
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
    chat_gpt_test.py           # MPS vs full state-vector diagnostic
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
# Activate the venv
source eddie_py312_venv/bin/activate
# or on scratch:
source /exports/eddie/scratch/s1931382/scratch_dpqc_venv/bin/activate

# Run examples directly
python -m src.examples.find_gs_tc_example
python -m src.examples.plot_training_curve_tc

# Submit to Grid Engine
qsub run_plot_training_curve_tc.sh
qsub run_mps_diagnostic.sh
```

## Common patterns

- Circuits are built with `tc.Circuit` (state vector) or `tc.MPSCircuit` (MPS approximation), controlled by `split_conf` dict and `use_mps` flags.
- Two-qubit gates use the Cartan decomposition block: 2x single-qubit Ry-Rz-Ry + Rxx/Ryy/Rzz.
- Probabilistic resets use ancilla qubits with parametrized Ry + Toffoli (CCX) gates; on MPS circuits, Toffoli is decomposed into 1- and 2-qubit gates via `decompose_ccx()`.
- Energy is computed via `sparse_expectation(qc, fullham)` (sparse mode) or term-by-term `qc.expectation()`.
- All optimization uses JAX vmap over trials + JIT + Adam (optax).

## Development notes

- `jax_enable_x64` is set globally in multiple files — this is required for numerical precision.
- `split_conf = {}` in `generate_ansatz.py` controls tensor splitting; set it to `{"max_singular_values": N}` for MPS truncation.
- Results go to `results/` (master HDF5) and `tmp/` (per-run HDF5). Plots go to `outputs/` or `$DPQC_OUTDIR`.
- Thread counts are pinned to 1 in job scripts to avoid contention on shared nodes.
