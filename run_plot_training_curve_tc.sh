#!/bin/bash
#$ -N dpqc_plot
#$ -cwd
#$ -l h_rt=48:00:00
#$ -l h_vmem=128G
#$ -o /dev/null
#$ -e /dev/null


# ----------------------------------------------------------------------------------------

JOB_NAME="3x3-resets-off-profiling-and-grads"

# ----------------------------------------------------------------------------------------



# --- Move into the repository root explicitly ---
cd /home/s1931382/DynParQCircLearning || exit 1

# --- Create a per-run log + output directory ---
RUN_TAG=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR=logs/${RUN_TAG}_${JOB_NAME}_${JOB_ID}
mkdir -p "$RUN_DIR"

# Redirect stdout/stderr into the per-run directory
exec > "$RUN_DIR/stdout.out"
exec 2> "$RUN_DIR/stderr.err"

# --- Safety: limit threads ---
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false

# --- Activate environment ---
source /exports/eddie/scratch/s1931382/scratch_dpqc_venv/bin/activate
cd /home/s1931382/DynParQCircLearning || exit 1

# --- Log metadata (so you can always identify a run) ---
echo "========== RUN METADATA =========="
echo "Job name:   ${JOB_NAME}"
echo "Job ID:     ${JOB_ID}"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
if command -v git >/dev/null 2>&1; then
  echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
  echo "Git status: $(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') files changed"
fi
echo "Run dir:    ${RUN_DIR}"
echo "=================================="

# Tell Python where to write plots for this run
export DPQC_OUTDIR="$RUN_DIR/plots"

# --- Run code ---
python -m src.examples.plot_training_curve_tc

echo "End time:   $(date)"