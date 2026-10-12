#!/bin/bash
# =====================================================================================
# A/B: what does APPLYING RESETS cost, per optimisation step, in single-trajectory mode?
# =====================================================================================
#
# THE QUESTION
# ------------
# Single-trajectory resets are ancilla-free: the chain stays at nq sites and one branch
# of the reset channel is sampled in place. So the resets are not obviously expensive the
# way the purified path is (which spends 2 ancilla sites per reset event). The question
# this job answers is narrower and has never been measured here: holding everything else
# fixed, does firing the resets make a step FASTER or SLOWER than not firing them?
#
# Two effects pull in opposite directions:
#   costlier  each reset is extra gate work and extra MPS gate-splitting (SVD/QR) calls
#   cheaper   the reset Kraus operators K1 = sin(t/2)|0><0| and K2 = sin(t/2)|0><1| are
#             projectors, so a reset can only ever REDUCE entanglement, which means less
#             of the bond-dimension budget is actually used downstream
# This job measures their sum at a FIXED bond dimension. It does not measure the second
# effect's real payoff -- see the caveat at the bottom.
#
# THE TWO ARMS
# ------------
#   none   reset_layers=[]   the reset-free control: total_resets = 0
#   traj   reset_layers=[1]  production single-trajectory resets: total_resets = R
#
# `none` is NOT the plain unitary ansatz. It is the same ansatz with the reset events
# deleted and nothing else changed: same Cartan blocks (3 per plaquette per layer, not
# the unitary ansatz's 4), same optimal-ordering search, same ancilla-free nq-site chain,
# same seed, same trajectory estimator. With zero reset events the DiCE magic box is
# exp(0 - 0) = 1 and the score term vanishes identically, so the `none` energy is
# deterministic and plain backprop through it is exact. That leaves exactly one
# difference between the arms: whether the R resets fire.
#
# Each arm runs in its OWN PROCESS (reset_impl_bench does this by default). That matters:
# peak_bytes_in_use is a monotone high-water mark over a process, so benchmarking two arms
# in one process would report the larger arm's peak for both.
#
# HOW TO READ THE OUTPUT
# ----------------------
# Look at $RUN_DIR/stdout.out. The SUMMARY table at the end has one row per arm:
#
#     config   trials  chain   bd  branch  warm vg   /trial  peak GiB  SVD/step
#
#   warm vg    median wall-clock of one value_and_grad over all trials -- THE TIME ANSWER.
#              traj / none > 1 means resets cost time; < 1 means they save it.
#   peak GiB   peak device bytes after value_and_grad -- THE MEMORY ANSWER at this bond
#              dimension. Expect these to come out NEARLY EQUAL: at a fixed bond dimension
#              the tensors are the same shape whatever the entanglement actually is, so
#              this column mostly measures the chain length and bd, which both arms share.
#              A real memory difference lives in the bond dimension each arm NEEDS, which
#              is the companion job run_reset_bdreq_ab.sh.
#   SVD/step   real SVD calls through TensorNetwork's MPS gate-splitting path. This is the
#              mechanism behind whatever the time column shows -- if traj is slower, this
#              column says by how much extra splitting work.
#   chain      sites in the MPS. Both arms must read the same number (nq). If they do not,
#              something is wrong and the comparison is invalid.
#
# Also printed per arm, above the summary:
#   compile forward / value_and_grad   one-off JIT cost, not part of the per-step answer
#   backward tape adds N GiB           how much of peak is the reverse-mode tape
#   energy, |grad|, all finite         sanity only. The two arms are DIFFERENT ansaetze
#                                      (traj has R more parameters), so their energies are
#                                      not meant to match and a difference means nothing.
#
# The same numbers land in a CSV in $RUN_DIR for plotting.
#
# THE CAVEAT THIS JOB CANNOT COVER
# --------------------------------
# Fixed bond dimension is a fair comparison of per-step cost and an UNFAIR comparison of
# total cost, because it silently assumes both arms need the same bond dimension to be
# accurate. Since resets only reduce entanglement, the reset-free arm should need at least
# as much, possibly more. If it needs more, the cheaper step measured here is bought at a
# larger bd, and the honest verdict has to combine the two. run_reset_bdreq_ab.sh measures
# that half. Neither job says anything about how many STEPS each arm needs to converge --
# that is an optimisation question and would take full training runs.
#
# USAGE
# -----
#   sbatch --job-name=cost_ab_3x3 --time=03:00:00 \
#          --export=ALL,DPQC_AB_LATTICE=3x3,DPQC_AB_BOND_DIM=64 run_reset_cost_ab.sh
#   sbatch --job-name=cost_ab_4x3 --time=04:00:00 \
#          --export=ALL,DPQC_AB_LATTICE=4x3,DPQC_AB_BOND_DIM=128 run_reset_cost_ab.sh
#
# Values are single tokens with no commas on purpose: sbatch --export treats commas as
# separators BETWEEN variables, so a comma inside a value silently truncates it and the
# job then runs, exits 0, and measures the wrong thing. The arm list is built below,
# inside the job, for exactly that reason.
#
#SBATCH --job-name=reset_cost_ab
#SBATCH --time=03:00:00
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

# Without this JAX grabs 75% of the card up front and both nvidia-smi and the sampler
# report a flat ~34.5 GiB for every job, which would make the memory column of this
# comparison meaningless.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source /home/s1931382/dpqc_venv/bin/activate
cd /home/s1931382/DynParQCircLearning || exit 1

export DPQC_OUTDIR="$RUN_DIR"

LATTICE=${DPQC_AB_LATTICE:-3x3}
BOND_DIM=${DPQC_AB_BOND_DIM:-64}
TRIALS=${DPQC_AB_TRIALS:-10}
OUT_CSV="$RUN_DIR/reset_cost_ab_${LATTICE}_bd${BOND_DIM}.csv"

echo "========== RUN METADATA =========="
echo "Job name:   ${SLURM_JOB_NAME}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "GPU info:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "Run dir:    ${RUN_DIR}"
echo "Question:   does firing single-trajectory resets cost or save per-step time/memory?"
echo "Arms:       none (reset_layers=[], R=0) vs traj (reset_layers=[1], R>0)"
echo "Held fixed: lattice=${LATTICE} bond_dim=${BOND_DIM} trials=${TRIALS} nlayers=2 seed=2026"
echo "Out CSV:    ${OUT_CSV}"
echo "=================================="

# Fail fast on a node whose A40 is present but whose cuInit fails: JAX warns, falls back
# to CPU, and produces timings that look like real measurements. For a benchmark that is
# the worst possible failure mode.
if [ "${DPQC_REQUIRE_GPU:-1}" = "1" ]; then
  if ! python -c "import jax,sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 1)" 2>/dev/null; then
    echo "FATAL: no usable GPU on $(hostname) -- JAX fell back to CPU. Aborting rather than"
    echo "       reporting CPU timings as if they were GPU ones."
    exit 75
  fi
fi

source /home/s1931382/DynParQCircLearning/job_memlog.sh
start_memlog "$RUN_DIR"

# --configs is built here rather than passed in: it contains a comma, and a comma in an
# sbatch --export value is a variable separator. Each config runs in its own process, so
# the per-arm peak memory figures do not contaminate each other.
python -m src.diagnostics.reset_impl_bench \
    --configs "${LATTICE}:none:${BOND_DIM},${LATTICE}:traj:${BOND_DIM}" \
    --trials "${TRIALS}" \
    --out "${OUT_CSV}" &
PY_PID=$!
echo "$PY_PID" > "$RUN_DIR/py.pid"
wait $PY_PID
EXIT_CODE=$?

stop_memlog "$RUN_DIR" "$EXIT_CODE"
echo "Job finished with exit code: ${EXIT_CODE}"
exit $EXIT_CODE
