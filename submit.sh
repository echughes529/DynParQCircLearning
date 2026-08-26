#!/bin/bash
# Submit a training run with parameters that travel WITH the job.
#
# sbatch does not snapshot your Python source -- it queues a job that opens the .py files
# when the job STARTS. Editing parameters between submissions is therefore unsafe: jobs
# that queue for a while all read whatever the file says at start time, not at submit
# time. Slurm does capture --export values at submit time, so passing parameters that way
# pins them to the job. See src/examples/run_config.py for the full story.
#
# Usage:
#   ./submit.sh                                 # job name: 3x3_bd64_r1_traj
#   ./submit.sh LX=4 LY=4                       # job name: 4x4_bd64_r1_traj
#   ./submit.sh LX=4 LY=4 BOND_DIM=32           # job name: 4x4_bd32_r1_traj
#   ./submit.sh USE_TRAJECTORY_RESETS=false BOND_DIM=104   # job name: 3x3_bd104_r1_pur
#   ./submit.sh LX=4 LY=4 TAG=traj_pls_wrk      # job name: 4x4_bd64_r1_traj_traj_pls_wrk
#   ./submit.sh NAME=whatever_i_like            # job name: whatever_i_like
#
# Default job name is {Lx}x{Ly}_bd{bond_dim}_r{reset_layers}_{traj|pur|enum}, where the
# last field records which reset implementation ran -- "traj" for the ancilla-free sampled
# path, "pur" for the purified two-ancilla path, "enum" for the ancilla-free exact
# enumeration over all 3^R Kraus branches. TAG= appends your own descriptive text
# to it; NAME= replaces it entirely. Any parameter in run_config.py can be set as
# KEY=VALUE, e.g. MAXITER=2000 TRIALS=20 RESET_LAYERS="[0,1]" TRACK_GRADS=true SEED=1234.
#
# TIME= sets the Slurm walltime, overriding the 24h default in the batch script:
#   ./submit.sh LX=2 LY=2 TIME=00:30:00      # small run, backfills onto a free GPU fast
#   ./submit.sh LX=3 LY=3 TIME=04:00:00
# SHORTER WALLTIMES QUEUE FASTER. Slurm backfill only starts a lower-priority job early if
# it would finish before the reserved high-priority job is due to start, so a 24h request
# rarely fits any gap. Rough guide from past runs: 2x2/3x2 finish in minutes, 3x3 bd64 in
# 1-4h, bd96-128 ablations in 6-12h, with the longest ever at 23h. Overrun = job killed,
# so leave headroom -- but do not ask for 24h to run a 5-minute job.
#
# Sweep example -- five jobs from a file you never touch:
#   for cfg in "2 2" "3 2" "3 3" "4 3" "4 4"; do
#     set -- $cfg
#     ./submit.sh LX=$1 LY=$2 TAG=traj_pls_wrk
#   done

set -euo pipefail
cd /home/s1931382/DynParQCircLearning

TAG=""
NAME=""
TIME=""

for arg in "$@"; do
  if [[ "$arg" != *=* ]]; then
    echo "error: arguments must be KEY=VALUE, got '$arg'" >&2
    exit 1
  fi
  key=${arg%%=*}
  val=${arg#*=}
  case "$key" in
    TAG)  TAG=$val ;;
    NAME) NAME=$val ;;
    TIME) TIME=$val ;;
    *)    export "DPQC_${key}=${val}" ;;
  esac
done

# Ask run_config for the RESOLVED values so the name reflects defaults we did not
# override. run_config imports only os+ast, so this costs ~0.05s -- importing
# plot_training_curve_tc instead would pull in TensorCircuit/JAX/TF and take ~20s.
source /home/s1931382/dpqc_venv/bin/activate
read -r R_LX R_LY R_BD R_RL R_MODE <<<"$(python -c '
from src.examples.run_config import (Lx, Ly, bond_dim, reset_layers,
                                     use_trajectory_resets, use_enumerated_resets)
rl = "all" if reset_layers is None else "-".join(str(x) for x in reset_layers)
mode = "enum" if use_enumerated_resets else ("traj" if use_trajectory_resets else "pur")
print(Lx, Ly, bond_dim, rl, mode)
')"

if [ -z "$NAME" ]; then
  NAME="${R_LX}x${R_LY}_bd${R_BD}_r${R_RL}_${R_MODE}"
  [ -n "$TAG" ] && NAME="${NAME}_${TAG}"
fi

# A --time on the sbatch command line overrides the #SBATCH --time=24:00:00 directive in
# run_plot_training_curve_tc.sh. Shorter walltimes queue much faster: Slurm's backfill
# scheduler only starts a lower-priority job early if it would finish before the
# high-priority job holding that node is due to start, and a 24h request almost never
# fits in such a gap. Right-size this and small runs start in minutes instead of hours.
TIME_ARG=()
if [ -n "$TIME" ]; then
  TIME_ARG=(--time="$TIME")
fi

# crannog05 has an A40 but cuInit fails there, so JAX silently falls back to CPU
# and the job runs ~70x slower while still exiting 0. Keep jobs off it. Override
# with EXCLUDE= to target a specific node set.
EXCLUDE=${EXCLUDE:-crannog05}
EXCLUDE_ARG=()
[ -n "$EXCLUDE" ] && EXCLUDE_ARG=(--exclude="$EXCLUDE")

# --export=ALL forwards this shell's environment (including the DPQC_* we just exported)
# to the job, captured now at submit time.
echo "Submitting job '${NAME}'  (Lx=${R_LX} Ly=${R_LY} bond_dim=${R_BD} reset_layers=${R_RL} resets=${R_MODE}${TIME:+ time=$TIME})"
sbatch --job-name="$NAME" "${TIME_ARG[@]}" "${EXCLUDE_ARG[@]}" --export=ALL run_plot_training_curve_tc.sh
