#!/bin/bash
#SBATCH --job-name=traj_reset_gpu_probe
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=ICF-Free
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1

# Validation step 0 for ancilla-free trajectory resets: the one check that could
# not be made on a login node.
#
# A reset leaves its site tensor exactly rank-1 in the physical index, so every
# subsequent MPSCircuit.position() sweep decomposes an exactly singular matrix.
# JAX's QR derivative divides by diag(R), and an isolated jnp.linalg.qr grad
# probe on such a matrix does produce NaN -- but inside the real energy+gradient
# pipeline the cotangents never excite the singular directions and stock QR is
# finite AND correct (verified on CPU to 2.4e-9 against the purified path). The
# residual unknown is that the GPU uses a different QR kernel.
#
# Runs the full equivalence suite on an A40. If the QR concern is real it will
# show up as non-finite gradients in the enumeration tests and the end-to-end
# smoke test. Nothing longer should be launched until this is green.

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

echo "========== RUN METADATA =========="
echo "Job name:   ${SLURM_JOB_NAME}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "GPU info:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "Run dir:    ${RUN_DIR}"
echo "=================================="

export DPQC_OUTDIR="$RUN_DIR"

# Projection also feeds more rank-deficient matrices into the SVD splits, which
# is the already-diagnosed cuSOLVER batched-Jacobi failure. Keep the QR forward
# kernel on for every trajectory-mode run.
export DPQC_SVD_FWD_ALG=qr
python -m src.diagnostics.test_trajectory_resets
echo "Job finished with exit code: $?"
