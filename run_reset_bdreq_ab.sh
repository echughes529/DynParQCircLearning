#!/bin/bash
# =====================================================================================
# A/B: do single-trajectory RESETS LOWER THE BOND DIMENSION the simulation needs?
# =====================================================================================
#
# THE QUESTION
# ------------
# This is the other half of run_reset_cost_ab.sh, and for the memory question it is the
# half that matters. That job compares a reset arm and a reset-free arm at ONE bond
# dimension, which measures per-step cost but assumes both arms need the same bd to be
# accurate. Memory in an MPS simulation is set by the bond dimension you have to carry,
# so the real memory question is: what bond dimension does each arm actually need?
#
# There is a prediction to test. The reset Kraus operators
#     K0 = cos(t/2) I,  K1 = sin(t/2)|0><0|,  K2 = sin(t/2)|0><1|
# are a scalar multiple of the identity and two projectors. A projector cannot create
# entanglement, so applying a reset can only ever REDUCE the Schmidt rank across any cut.
# The reset-free arm should therefore need a bond dimension AT LEAST as large as the
# trajectory arm's. Whether it needs meaningfully more -- enough for the resets to pay for
# themselves -- is what this job measures.
#
# THE TWO ARMS
# ------------
#   none   reset_layers=[]   reset-free control, total_resets = 0
#   traj   reset_layers=[1]  production single-trajectory resets, total_resets = R
# Same nq-site ancilla-free chain, same exact cap of 2^floor(nq/2), same seed, same
# ordering search. Only the reset events differ. (See run_reset_cost_ab.sh for why this
# control is used rather than the plain unitary ansatz.)
#
# HOW "NEEDED" IS MEASURED
# ------------------------
# Every MPS truncation discards some weight. With normalisation OFF -- which this
# diagnostic forces, since normalize_state=True hides the loss by rescaling the norm back
# to 1 -- that loss shows up directly as
#
#     discarded weight = 1 - ||psi||^2
#
# and the fractional error in the unnormalised energy equals it exactly. The script walks
# a ladder of bond dimensions and reports the first rung where the discarded weight falls
# to the gate of 1e-5, two orders below the 1e-3 relative-energy criterion used to call a
# training trial converged.
#
# The trajectory arm is read as the WORST of 20 sampled trajectories, not the average: the
# bond dimension has to be enough for every trajectory the optimiser might draw. The
# reset-free arm has no randomness to sample (its status vector is empty, so every draw
# builds the same circuit), so it is measured once per rung.
#
# HOW TO READ THE OUTPUT
# ----------------------
# In $RUN_DIR/stdout.out, each arm prints a ladder and then a verdict line:
#
#     REQUIRED bd: 32  ->  long-run bd = 40
#
# Compare the two REQUIRED numbers -- THAT IS THE MEMORY ANSWER.
#   none > traj   resets pay for themselves: the reset-free circuit has to carry a bigger
#                 bond dimension, and MPS memory grows with it, so the resets buy memory.
#   none = traj   resets are memory-neutral here; the verdict then rests entirely on the
#                 per-step time from run_reset_cost_ab.sh.
#   none < traj   would contradict the projector argument above. Do not accept it without
#                 checking that the two arms really do report the same `chain` -- a
#                 mis-specified control is the likely cause, not new physics.
#
# In the ladder table itself:
#   discarded wt   1 - ||psi||^2, the fraction of the state truncation threw away
#   E (raw)        unnormalised energy; its fractional error IS the discarded weight
#   E (norm)       the same energy rescaled -- what the estimator would actually report
#   max bd         bond dimension the state actually reached. If this sits well below the
#                  rung's cap, that rung was never the binding constraint
#   "(exact: full Hilbert space)"  at bd = 2^floor(nq/2) the MPS is the full state vector,
#                  so truncation is provably zero. A "requirement" only reached there
#                  means the ladder never found a real saving, not that bd is unbounded
#
# Each lattice is measured at two parameter sets where available: `init` (a random start)
# and `converged` (the best trial of the stored 3x3 bd=96 run). Required bond dimension is
# state-dependent, so a random initialisation alone would answer the wrong question. The
# `converged` set is skipped for the `none` arm -- those parameters were trained WITH
# resets and carry R extra thetas the reset-free ansatz has no slot for; the job says so
# when it skips.
#
# The CSV in $RUN_DIR is rewritten after every rung, so a job killed at walltime still
# leaves everything it had measured up to that point.
#
# USAGE
# -----
#   sbatch --job-name=bdreq_ab_3x3 --time=03:00:00 \
#          --export=ALL,DPQC_AB_LATTICE=3x3 run_reset_bdreq_ab.sh
#   sbatch --job-name=bdreq_ab_4x3 --time=06:00:00 \
#          --export=ALL,DPQC_AB_LATTICE=4x3 run_reset_bdreq_ab.sh
#
# 4x3 is the informative one: R = 6 resets against 4 at 3x3, on a 17-site chain whose
# exact cap is 256, so there is room for the arms to separate. It is also the slower one
# -- the previous 4x3 trajectory ladder took about 32 minutes, and the reset-free arm
# climbs further up the ladder before it passes the gate.
#
#SBATCH --job-name=reset_bdreq_ab
#SBATCH --time=06:00:00
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
export XLA_PYTHON_CLIENT_PREALLOCATE=false

source /home/s1931382/dpqc_venv/bin/activate
cd /home/s1931382/DynParQCircLearning || exit 1

export DPQC_OUTDIR="$RUN_DIR"

LATTICE=${DPQC_AB_LATTICE:-3x3}
OUT_CSV="$RUN_DIR/reset_bdreq_ab_${LATTICE}.csv"

echo "========== RUN METADATA =========="
echo "Job name:   ${SLURM_JOB_NAME}"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "GPU info:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "Run dir:    ${RUN_DIR}"
echo "Question:   do single-trajectory resets lower the bond dimension needed?"
echo "Arms:       none (reset_layers=[], R=0) vs traj (reset_layers=[1], R>0)"
echo "Held fixed: lattice=${LATTICE} nlayers=2 seed=2026 normalize_state=False gate=1e-5"
echo "Out CSV:    ${OUT_CSV}"
echo "=================================="

if [ "${DPQC_REQUIRE_GPU:-1}" = "1" ]; then
  if ! python -c "import jax,sys; sys.exit(0 if any(d.platform=='gpu' for d in jax.devices()) else 1)" 2>/dev/null; then
    echo "FATAL: no usable GPU on $(hostname) -- JAX fell back to CPU. Aborting."
    exit 75
  fi
fi

source /home/s1931382/DynParQCircLearning/job_memlog.sh
start_memlog "$RUN_DIR"

# --modes contains a comma, so it is written here rather than passed through --export.
python -m src.diagnostics.bond_dim_requirement \
    --lattices "${LATTICE}" \
    --modes none,traj \
    --out "${OUT_CSV}" &
PY_PID=$!
echo "$PY_PID" > "$RUN_DIR/py.pid"
wait $PY_PID
EXIT_CODE=$?

stop_memlog "$RUN_DIR" "$EXIT_CODE"
echo "Job finished with exit code: ${EXIT_CODE}"
exit $EXIT_CODE
