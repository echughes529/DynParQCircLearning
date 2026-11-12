#!/bin/bash
# Submit the post-training analysis for every finished Experiment 1 run.
#
# Usage:
#   experiments/experiment_1_scaling_system_size/analyse_all.sh            # runs without a summary.json
#   experiments/experiment_1_scaling_system_size/analyse_all.sh FORCE=1    # redo all
#   experiments/experiment_1_scaling_system_size/analyse_all.sh DRY_RUN=1
#
# One diag job per run folder (exp1_analyse_run is cheap for purified and
# enumerated, but re-evaluates every saved parameter vector over all 81
# trajectories for the MC methods, so it gets the GPU). Then, once they are
# all done, make the comparison plots on the login node:
#   python -m src.experiments.exp1_compare

set -euo pipefail
REPO=/home/s1931382/DynParQCircLearning
EXP=experiments/experiment_1_scaling_system_size
cd "$REPO"

FORCE=0
DRY_RUN=0
TIME=02:00:00
for arg in "$@"; do
  [[ "$arg" == *=* ]] || { echo "error: arguments must be KEY=VALUE, got '$arg'" >&2; exit 1; }
  declare "${arg%%=*}=${arg#*=}"
done

for d in "$EXP"/*x*/*/; do
  d=${d%/}
  m=$(basename "$d")
  case "$m" in mc_n1|mc_n8|purified|enumerated) ;; *) continue ;; esac   # skips pilot/
  [ -f "$d/stdout.out" ] || continue
  lat=$(basename "$(dirname "$d")")
  # A queued or running job (first run or a resubmission) means the folder's
  # current files are stale or incomplete, whatever stdout.out says -- a
  # pending resubmission still has the previous run's stdout.out. Exact name
  # match: 2x2 is a substring of 12x2, and pilots use a different prefix.
  if squeue -h -u "$USER" -o %j | grep -qx "exp1_prod_${lat}_${m}"; then
    echo "skip $d: job queued or running"; continue
  fi
  # Re-analyse whenever the run's output is newer than its summary: a run that
  # failed, was analysed, then succeeded on resubmission must not keep its old
  # "timeout"/"oom" summary.
  if [ "$FORCE" != 1 ] && [ -f "$d/summary.json" ] && \
     [ "$d/summary.json" -nt "$d/stdout.out" ] && \
     { [ ! -f "$d/run.h5" ] || [ "$d/summary.json" -nt "$d/run.h5" ]; }; then
    echo "skip $d: already analysed"; continue
  fi
  if squeue -h -u "$USER" -o %j | grep -qx "exp1_an_${lat}_${m}"; then
    echo "skip $d: analysis job already queued or running"; continue
  fi
  # Purified/enumerated (logged energy is already exact) and failed runs
  # (status only) need no GPU and take seconds: run them here. Only the MC
  # methods re-evaluate every saved parameter vector over all 3^R branches, and
  # timing is irrelevant there, so they go to ANY free GPU, not just A40s.
  if [ "$m" = purified ] || [ "$m" = enumerated ] || \
     ! grep -q "Job finished with exit code: 0" "$d/stdout.out"; then
    CMD=(python -m src.experiments.exp1_analyse_run "$d")
  else
    CMD=(./submit_diag.sh NAME="exp1_an_${lat}_${m}" TIME="$TIME" GRES=gpu:1 --
         python -m src.experiments.exp1_analyse_run "$d")
  fi
  if [ "$DRY_RUN" = 1 ]; then echo "${CMD[*]}"; else "${CMD[@]}"; fi
done
