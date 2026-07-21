#!/bin/bash
#SBATCH --job-name=lightcone_diag
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=ICF-Free
#SBATCH --cpus-per-task=2
# This is a cheap diagnostic (small toy lattice, no full Hamiltonian build,
# no training loop) - a GPU isn't required, but is requested by default since
# the venv's jaxlib build may expect one. Drop this line (or override at
# submit time) if you'd rather run it on a CPU-only node:
#   sbatch --gres=gpu:0 run_lightcone_diagnostic.sh
#SBATCH --gres=gpu:a40:1

# --- Move into the repository root explicitly ---
mkdir -p /home/s1931382/DynParQCircLearning/logs
cd /home/s1931382/DynParQCircLearning || exit 1

# --- Create a per-run log + output directory ---
RUN_TAG=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR=logs/${RUN_TAG}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

# Redirect stdout/stderr into the per-run directory
exec > "$RUN_DIR/stdout.out"
exec 2> "$RUN_DIR/stderr.err"

# --- Thread settings ---
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=2
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1

# --- Activate environment ---
source /home/s1931382/dpqc_venv/bin/activate
cd /home/s1931382/DynParQCircLearning || exit 1

# --- Log metadata ---
echo "========== RUN METADATA =========="
echo "Job name:   ${SLURM_JOB_NAME}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "GPU info:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"
if command -v git >/dev/null 2>&1; then
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
  echo "Git status: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') files changed"
fi
echo "Run dir:    ${RUN_DIR}"
echo "=================================="

# --- Run code ---
python -m src.examples.lightcone_diagnostic_example
EXIT_CODE=$?

echo "Job finished with exit code: ${EXIT_CODE}"
