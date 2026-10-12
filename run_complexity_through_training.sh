#!/bin/bash
# =====================================================================================
# How complex is each arm's MPS at every point along a training run -- and why does
# the single-trajectory arm converge like the exact ones despite approximating them?
# =====================================================================================
#
# THE TWO QUESTIONS
# -----------------
# 1. COMPLEXITY. docs/three_reset_implementations.md section 2 measures "required bond
#    dimension" by laddering hard caps and reporting the first rung that passes a gate,
#    at initialisation and at convergence only. That gives a rung rather than a number,
#    and at 3x3 it reports the ancilla-free arms pinned at their exact cap of 64 -- which
#    section 2 itself has to caveat as "a ceiling, not a measurement". This job replaces
#    that with a continuous curve: rebuild the state at every stored snapshot, truncating
#    to a fixed relative ERROR instead of a fixed bond dimension, and read off the bond
#    dimension the state actually settles at.
#
# 2. WHY THE APPROXIMATION IS TOLERABLE. The trajectory arm samples ONE of the 3^R Kraus
#    branches per energy evaluation, yet converges much like the arms that evaluate all
#    of them. The second script tests whether that is because the optimiser drives the
#    reset angles to zero, collapsing the channel onto the single "nothing fired" branch,
#    at which point sampling one branch and evaluating the channel are the same thing.
#
# WHY THIS RUNS ON CPU AND NEEDS NO GPU
# -------------------------------------
# Error-tolerance truncation cannot run under jit at all: the number of singular values
# kept is data-dependent, so tensor shapes change from call to call. That is precisely
# why the production forward pass only ever uses max_singular_values -- it has to be
# traceable, vmappable and differentiable. Nothing here is any of those, so the rebuild
# runs on TensorCircuit's NumPy backend: plain LAPACK, real numpy arrays, no XLA, no
# autodiff tape, and no exposure to the GPU batched-Jacobi SVD failures that NaN training
# runs. It is also about 10x faster than the same rebuild under JAX-eager.
#
# So, unlike run_reset_cost_ab.sh and run_reset_bdreq_ab.sh, this script requests NO GPU
# and deliberately does NOT carry their DPQC_REQUIRE_GPU guard. A GPU here would be idle.
#
# INPUTS: TRAINING RUNS, NOT NEW TRAINING
# ---------------------------------------
# Both scripts read `all_params` out of a finished run's checkpoint.npz, so nothing is
# retrained. That requires TRACK_PARAMS=true on the source run -- without it all_params is
# all zeros and both scripts refuse rather than silently measuring nothing.
#
# The two checkpoints MUST use the same seed. Matched seeds give the arms bitwise
# identical initial parameters (nparams does not depend on the reset implementation), so
# the comparison starts from the same point.
#
# HOW TO READ THE OUTPUT
# ----------------------
# complexity_through_training_<lattice>.csv, one row per
# (comparison, param source, rebuild mode, snapshot, trial, tolerance):
#
#   max_bond_dim   widest bond. Compare WITHIN an arm across training. Comparing it
#                  BETWEEN arms is unsound -- the trajectory chain is 12 sites at 3x3 and
#                  the purified chain is 20, so it is not the same cut.
#   mps_entries    total stored complex entries. THIS is the cross-arm number: the real
#                  memory footprint, and it does not care how long the chain is.
#   max_entropy_vn physical entanglement rather than representational cost.
#   discarded_wt   1 - ||psi||^2. If this is not near the tolerance, the cap bound instead.
#   cap_bound      true means the safety cap bound: that row is a CEILING, not a
#                  measurement, and its bond dimension must not be quoted as a requirement.
#   comparison     as_trained  = each arm's own parameters in its own mode. The literal
#                               question, but the two runs have drifted apart, so a
#                               difference mixes representation cost with optimisation path.
#                  same_params = one parameter history rebuilt in BOTH modes. The
#                               controlled version, and the only one where a difference is
#                               attributable to the reset implementation alone.
#
# traj_approximation_<lattice>.csv, one row per (snapshot, trial):
#
#   p_no_reset_anywhere   prod_r (1 - sin^2(theta_r/2)). Climbing towards 1 means the
#                         channel is collapsing onto the single all-K0 branch.
#   branch_participation  effective number of the 3^R branches carrying weight. 1 means
#                         fully dominated.
#   energy_exact          exact channel energy, by walking every Kraus branch.
#   energy_traj_std       spread across draws: the noise one draw feeds the optimiser.
#   bias                  sample mean minus exact. NOT an estimator bug when the spread
#                         is 0 -- see the note below.
#   energy_top_branch     the dominant branch's own energy. When the spread reads 0 every
#                         draw took that branch, so energy_traj_mean equals this exactly
#                         and the gap to energy_exact is the UNSAMPLED TAIL, not bias.
#                         The estimator is unbiased in expectation; a finite sample of it
#                         cannot resolve a 6e-4 probability tail.
#
# Both CSVs are rewritten after every measurement, so a job killed at walltime keeps
# everything it had measured.
#
# USAGE
# -----
#   sbatch --job-name=cplx_3x3 --export=ALL,\
# DPQC_TRAJ_CKPT=logs/<traj run>/plots/checkpoint.npz,\
# DPQC_PUR_CKPT=logs/<pur run>/plots/checkpoint.npz \
#          run_complexity_through_training.sh
#
# Defaults point at the matched-seed 3x3 three-arm runs from 2026-08-26 (seed 4242).
# Those trained with the reset angles initialised near pi/2; the working tree now
# initialises them in [0.05, 0.1], which is a different experiment. Every row records
# reset_theta_min/max so the two are distinguishable rather than silently mixed.
#
#SBATCH --job-name=complexity_through_training
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=ICF-Free
#SBATCH --cpus-per-task=8

mkdir -p /home/s1931382/DynParQCircLearning/logs
cd /home/s1931382/DynParQCircLearning || exit 1

RUN_TAG=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR=logs/${RUN_TAG}_${SLURM_JOB_NAME:-complexity}_${SLURM_JOB_ID:-local}
mkdir -p "$RUN_DIR"

exec > "$RUN_DIR/stdout.out"
exec 2> "$RUN_DIR/stderr.err"

export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=2
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1

# The rebuild is NumPy on CPU by construction. Pinning JAX to CPU stops it grabbing a
# card it will never use -- the ansatz objects are still constructed under JAX, because
# __post_init__ builds vmapped value_and_grad helpers the NumPy backend refuses outright.
export JAX_PLATFORMS=cpu

source /home/s1931382/dpqc_venv/bin/activate
cd /home/s1931382/DynParQCircLearning || exit 1

export DPQC_OUTDIR="$RUN_DIR"

LATTICE=${DPQC_LATTICE:-3x3}
TRAJ_CKPT=${DPQC_TRAJ_CKPT:-logs/2026-08-26_14-02-49_3x3_bd64_r1_traj_3arm_3632190/plots/checkpoint.npz}
PUR_CKPT=${DPQC_PUR_CKPT:-logs/2026-08-26_14-02-49_3x3_bd64_r1_pur_3armctl_3632193/plots/checkpoint.npz}
TRIALS=${DPQC_TRIALS:-3}
STRIDE=${DPQC_SNAPSHOT_STRIDE:-1}
CAP=${DPQC_CAP:-512}

echo "========== RUN METADATA =========="
echo "Job name:   ${SLURM_JOB_NAME:-local}"
echo "Job ID:     ${SLURM_JOB_ID:-none}"
echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
echo "Run dir:    ${RUN_DIR}"
echo "Question:   MPS complexity through training, and why one sampled branch suffices"
echo "Lattice:    ${LATTICE}   trials=${TRIALS} stride=${STRIDE} safety cap=${CAP}"
echo "traj ckpt:  ${TRAJ_CKPT}"
echo "pur  ckpt:  ${PUR_CKPT}"
echo "=================================="

for ckpt in "$TRAJ_CKPT" "$PUR_CKPT"; do
  if [ ! -f "$ckpt" ]; then
    echo "FATAL: checkpoint not found: $ckpt"
    exit 66
  fi
done

source /home/s1931382/DynParQCircLearning/job_memlog.sh
start_memlog "$RUN_DIR"

# --tolerances contains commas, so it is written here rather than passed through
# --export: sbatch treats a comma inside an --export value as a variable separator,
# which would silently truncate it and leave the job measuring one tolerance.
python -m src.diagnostics.complexity_through_training \
    --lattice "${LATTICE}" \
    --traj-checkpoint "${TRAJ_CKPT}" \
    --pur-checkpoint "${PUR_CKPT}" \
    --tolerances 1e-4,1e-6,1e-8 \
    --trials "${TRIALS}" --snapshot-stride "${STRIDE}" --traj-samples 5 --cap "${CAP}" \
    --out "$RUN_DIR/complexity_through_training_${LATTICE}.csv"
EXIT_A=$?

python -m src.diagnostics.traj_approximation_quality \
    --lattice "${LATTICE}" \
    --checkpoint "${TRAJ_CKPT}" \
    --trials "${TRIALS}" --snapshot-stride 2 --traj-samples 16 --cap "${CAP}" \
    --out "$RUN_DIR/traj_approximation_${LATTICE}.csv"
EXIT_B=$?

# Both CSVs also go to outputs/ so plot_three_resets picks them up without a copy step.
mkdir -p outputs/three_resets
cp "$RUN_DIR"/complexity_through_training_*.csv "$RUN_DIR"/traj_approximation_*.csv \
   outputs/three_resets/ 2>/dev/null

EXIT_CODE=$(( EXIT_A != 0 ? EXIT_A : EXIT_B ))
stop_memlog "$RUN_DIR" "$EXIT_CODE"
echo "complexity exit ${EXIT_A}, approximation exit ${EXIT_B}"
echo "Job finished with exit code: ${EXIT_CODE}"
exit $EXIT_CODE
