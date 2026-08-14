#!/bin/bash
#SBATCH --job-name=gate-mode-ablation
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=ICF-Free
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1

# Matched training-dynamics comparison across cartan_mode/toffoli_mode/
# bond_dim, isolating whether the SVD-reduction gate-path feature flags
# and/or bond_dim contribute to the "energy descends then oscillates,
# grad_norm spikes" behaviour seen in job 3612091 (seed=4876, trials=3).
# Submit once per config:
#   sbatch --job-name=ablation_fused_direct_bd64 run_gate_mode_ablation.sh fused direct 64 4876 3 500

CARTAN_MODE="${1:?Usage: sbatch run_gate_mode_ablation.sh <cartan_mode> <toffoli_mode> <bond_dim> <seed> <trials> <maxiter>}"
TOFFOLI_MODE="${2:?Usage: sbatch run_gate_mode_ablation.sh <cartan_mode> <toffoli_mode> <bond_dim> <seed> <trials> <maxiter>}"
BOND_DIM="${3:?Usage: sbatch run_gate_mode_ablation.sh <cartan_mode> <toffoli_mode> <bond_dim> <seed> <trials> <maxiter>}"
SEED="${4:?Usage: sbatch run_gate_mode_ablation.sh <cartan_mode> <toffoli_mode> <bond_dim> <seed> <trials> <maxiter>}"
TRIALS="${5:?Usage: sbatch run_gate_mode_ablation.sh <cartan_mode> <toffoli_mode> <bond_dim> <seed> <trials> <maxiter>}"
MAXITER="${6:?Usage: sbatch run_gate_mode_ablation.sh <cartan_mode> <toffoli_mode> <bond_dim> <seed> <trials> <maxiter>}"

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
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=4
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
echo "Run dir:      ${RUN_DIR}"
echo "cartan_mode:  ${CARTAN_MODE}"
echo "toffoli_mode: ${TOFFOLI_MODE}"
echo "bond_dim:     ${BOND_DIM}"
echo "seed:         ${SEED}"
echo "trials:       ${TRIALS}"
echo "maxiter:      ${MAXITER}"
echo "=================================="

# --- Memory + GPU monitor (logs every 30s in the background) ---
(
  while true; do
    echo "[memlog $(date +%H:%M:%S)] RAM: $(free -h | awk '/^Mem:/{print "used="$3" avail="$7}') | GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null)"
    sleep 30
  done
) >> "$RUN_DIR/memlog.txt" &
MEMLOG_PID=$!

# --- Run code ---
python -m src.diagnostics.gate_mode_training_ablation \
    --cartan-mode "$CARTAN_MODE" --toffoli-mode "$TOFFOLI_MODE" --bond-dim "$BOND_DIM" \
    --seed "$SEED" --trials "$TRIALS" --maxiter "$MAXITER"
EXIT_CODE=$?

# --- Cleanup and final snapshot ---
kill $MEMLOG_PID 2>/dev/null
echo "[memlog final] exit_code=${EXIT_CODE} RAM: $(free -h | awk '/^Mem:/{print "used="$3" avail="$7}') | GPU: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null)" >> "$RUN_DIR/memlog.txt"
echo "Job finished with exit code: ${EXIT_CODE}"
