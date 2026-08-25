#!/bin/bash
# Generic diagnostics job runner.
#
# Runs whatever python command is passed in DIAG_CMD, with the same environment,
# logging and per-run directory conventions as run_plot_training_curve_tc.sh.
#
# Usage:
#   sbatch --job-name=bdreq --time=02:00:00 \
#     --export=ALL,DIAG_CMD="python -m src.diagnostics.bond_dim_requirement --lattices 3x3" \
#     run_diag.sh
#
# COMMAS: sbatch --export parses commas as separators BETWEEN variables, so any
# comma inside DIAG_CMD silently truncates it -- a command ending in
# "--stages 0,50,200" arrives as "--stages 0" and the job runs, exits 0, and
# quietly produces the wrong thing. For any command containing a comma, pass it
# base64-encoded instead:
#
#   sbatch --export=ALL,DIAG_CMD_B64="$(echo -n 'python -m mod --stages 0,50' | base64 -w0)" run_diag.sh
#
# Diagnostics that write CSVs should point --out at $DPQC_OUTDIR so results land
# next to the log of the job that produced them.
#SBATCH --job-name=dpqc_diag
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=ICF-Free
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1

mkdir -p /home/s1931382/DynParQCircLearning/logs
cd /home/s1931382/DynParQCircLearning || exit 1

RUN_TAG=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR=logs/${RUN_TAG}_${SLURM_JOB_NAME}_${SLURM_JOB_ID}
mkdir -p "$RUN_DIR"

exec > "$RUN_DIR/stdout.out"
exec 2> "$RUN_DIR/stderr.err"

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=2
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1

source /home/s1931382/dpqc_venv/bin/activate
cd /home/s1931382/DynParQCircLearning || exit 1

export DPQC_OUTDIR="$RUN_DIR"

if [ -n "${DIAG_CMD_B64:-}" ]; then
  DIAG_CMD=$(echo "$DIAG_CMD_B64" | base64 -d)
fi

echo "========== RUN METADATA =========="
echo "Job name:   ${SLURM_JOB_NAME}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "GPU info:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "Run dir:    ${RUN_DIR}"
echo "Command:    ${DIAG_CMD}"
echo "=================================="

# Fail fast on a CPU-fallback node -- see run_plot_training_curve_tc.sh for why.
if [ "${DPQC_REQUIRE_GPU:-1}" = "1" ]; then
  if ! python -c "import jax,sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 1)" 2>/dev/null; then
    echo "FATAL: no usable GPU on $(hostname) -- JAX fell back to CPU. Aborting."
    exit 75
  fi
fi

if [ -z "${DIAG_CMD:-}" ]; then
  echo "error: DIAG_CMD is unset; pass it via --export=ALL,DIAG_CMD=\"python -m ...\"" >&2
  exit 1
fi

eval "${DIAG_CMD}"
EXIT_CODE=$?

echo "Job finished with exit code: ${EXIT_CODE}"
exit $EXIT_CODE
