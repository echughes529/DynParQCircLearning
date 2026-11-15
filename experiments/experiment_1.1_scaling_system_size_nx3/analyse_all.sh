#!/bin/bash
# Submit the post-training analysis for every finished Experiment 1.1 run.
#
# Usage:
#   experiments/experiment_1.1_scaling_system_size_nx3/analyse_all.sh POLICY=caps_64_96
#   experiments/experiment_1.1_scaling_system_size_nx3/analyse_all.sh POLICY=caps_192_256 FORCE=1
#   ... DRY_RUN=1    print the commands, run nothing
#   ... TIME=08:00:00 BATCH=1 MEM=128G   for the MC analysis jobs
#
# Purified, enumerated and failed runs are analysed here (no GPU needed, they
# take seconds). The MC methods re-evaluate every saved parameter vector over
# all 3^R branches, so they go to ANY free GPU (timing is irrelevant there).
# Then make the comparison plots on the login node:
#   python -m src.experiments.exp1_compare --exp-dir <EXP>/<POLICY> \
#       --reference-exp-dir experiments/experiment_1_scaling_system_size
#
# Copied from experiments/experiment_1_scaling_system_size/analyse_all.sh. What
# differs: EXP path and the POLICY level; job names exp11_<policy>_*; <method>_calib
# folders are analysed too; symlinked (shared) run folders are skipped, since
# they are analysed in the policy that owns them; TIME/BATCH/MEM options for the
# MC analyses. Experiment 1's script is untouched.

set -euo pipefail
REPO=/home/s1931382/DynParQCircLearning
EXP=experiments/experiment_1.1_scaling_system_size_nx3
cd "$REPO"

POLICY=""
FORCE=0
DRY_RUN=0
TIME=04:00:00
BATCH=""
MEM=""
for arg in "$@"; do
  [[ "$arg" == *=* ]] || { echo "error: arguments must be KEY=VALUE, got '$arg'" >&2; exit 1; }
  declare "${arg%%=*}=${arg#*=}"
done
case "$POLICY" in
  caps_64_96|caps_192_256) ;;
  *) echo "error: POLICY must be caps_64_96 or caps_192_256" >&2; exit 1 ;;
esac
PDIR="$EXP/$POLICY"
P=${POLICY#caps_}

for d in "$PDIR"/*x*/*/; do
  d=${d%/}
  m=$(basename "$d")
  case "$m" in
    mc_n1|mc_n8|purified|enumerated) kind=prod ;;
    mc_n1_calib|mc_n8_calib|purified_calib|enumerated_calib) kind=cali ;;
    *) continue ;;
  esac
  [ -L "$d" ] && { echo "skip $d: shared run, analysed where it lives"; continue; }
  [ -f "$d/stdout.out" ] || continue
  lat=$(basename "$(dirname "$d")")
  base=${m%_calib}
  # A queued or running job means the folder's current files are stale or
  # incomplete, whatever stdout.out says. Exact name match.
  if squeue -h -u "$USER" -o %j | grep -qx "exp11_${P}_${kind}_${lat}_${base}"; then
    echo "skip $d: job queued or running"; continue
  fi
  # Re-analyse whenever the run's output is newer than its summary.
  if [ "$FORCE" != 1 ] && [ -f "$d/summary.json" ] && \
     [ "$d/summary.json" -nt "$d/stdout.out" ] && \
     { [ ! -f "$d/run.h5" ] || [ "$d/summary.json" -nt "$d/run.h5" ]; }; then
    echo "skip $d: already analysed"; continue
  fi
  AN_NAME="exp11_${P}_an_${lat}_${m}"
  if squeue -h -u "$USER" -o %j | grep -qx "$AN_NAME"; then
    echo "skip $d: analysis job already queued or running"; continue
  fi
  if [ "$base" = purified ] || [ "$base" = enumerated ] || \
     ! grep -q "Job finished with exit code: 0" "$d/stdout.out"; then
    CMD=(python -m src.experiments.exp1_analyse_run "$d")
  else
    CMD=(./submit_diag.sh NAME="$AN_NAME" TIME="$TIME" GRES=gpu:1 ${MEM:+MEM="$MEM"} --
         python -m src.experiments.exp1_analyse_run "$d" ${BATCH:+--batch "$BATCH"})
  fi
  if [ "$DRY_RUN" = 1 ]; then echo "${CMD[*]}"; else "${CMD[@]}"; fi
done
