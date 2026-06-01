#!/bin/bash
#$ -cwd
#$ -l h_rt=01:00:00
#$ -l h_vmem=64G

cd /home/s1931382/DynParQCircLearning || exit 1

JOB_NAME="${JOB_NAME:-default-job}"

# Create timestamped output folder
RUN_TAG=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR=logs/${RUN_TAG}_${JOB_NAME}_${JOB_ID}
mkdir -p "$RUN_DIR"

export DPQC_OUTDIR="${RUN_DIR}"

# Redirect SGE stdout/stderr
#$ -o /dev/null
#$ -e /dev/null

# Avoid thread oversubscription on Eddie
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false

# Activate environment
source /exports/eddie/scratch/s1931382/scratch_dpqc_venv/bin/activate

# Save all output
python -m src.examples.compute_qfi_tc \
    > "${RUN_DIR}/stdout.out" \
    2> "${RUN_DIR}/stderr.err"

echo "Saved outputs to:"
echo "${RUN_DIR}"



