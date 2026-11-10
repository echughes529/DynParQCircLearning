#!/bin/bash
# Submit Experiment 1 training runs: one Slurm job per (lattice, method).
#
# Usage (from anywhere):
#   experiments/experiment_1_scaling_system_size/submit_all.sh KIND=pilot
#   experiments/experiment_1_scaling_system_size/submit_all.sh KIND=production
#   experiments/experiment_1_scaling_system_size/submit_all.sh KIND=production LATTICES="6x2 7x2" METHODS="mc_n1 mc_n8"
#   ... DRY_RUN=1      print the submit commands, submit nothing
#   ... FORCE=1        resubmit runs that already finished cleanly
#
# KIND=pilot       -> <lattice>/pilot/<method>, 200 steps, bond dimension 64 (Phase 1)
# KIND=production  -> <lattice>/<method>, MAXITER steps, bond dimension from
#                     config.json bond_dim[lattice][method] (Phase 2/4).
#                     BOND_DIM=... overrides it for every run in this call.
#
# Every setting is passed through ./submit.sh, so it travels with the job and
# later code edits cannot change a queued run's parameters (see run_config.py).

set -euo pipefail
REPO=/home/s1931382/DynParQCircLearning
EXP=experiments/experiment_1_scaling_system_size
cd "$REPO"

KIND=production
LATTICES="2x2 3x2 4x2 5x2"
METHODS="mc_n1 mc_n8 purified enumerated"
MAXITER=2000
TIME=""            # default: 06:00:00 for pilots (queue faster), 48:00:00 for production
BOND_DIM=""
DRY_RUN=0
FORCE=0
for arg in "$@"; do
  [[ "$arg" == *=* ]] || { echo "error: arguments must be KEY=VALUE, got '$arg'" >&2; exit 1; }
  declare "${arg%%=*}=${arg#*=}"
done

# Fixed settings for every run of this experiment (PLAN.md, "Fixed settings").
TRIALS=5
SEED=1234
N_SPREAD_RESETS=4
RESET_LAYERS="[1]"
HOWOFTEN_TOSAVE=10

if [ "$KIND" = pilot ]; then
  MAXITER=200
  BOND_DIM=${BOND_DIM:-64}
  TIME=${TIME:-06:00:00}
elif [ "$KIND" != production ]; then
  echo "error: KIND must be pilot or production" >&2; exit 1
fi
TIME=${TIME:-48:00:00}

# JAX otherwise caps itself at 75% of the card (36 GB on an A40), so an
# "out of memory" would mean 36 GB, not the 48 GB the stopping rule is about.
# It also makes the allocator's live/peak numbers reflect real use.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for lat in $LATTICES; do
  LX=${lat%x*}; LY=${lat#*x}
  for m in $METHODS; do
    case "$m" in
      mc_n1)      FLAGS=(USE_TRAJECTORY_RESETS=true  USE_ENUMERATED_RESETS=false N_TRAJECTORIES=1) ;;
      mc_n8)      FLAGS=(USE_TRAJECTORY_RESETS=true  USE_ENUMERATED_RESETS=false N_TRAJECTORIES=8) ;;
      purified)   FLAGS=(USE_TRAJECTORY_RESETS=false USE_ENUMERATED_RESETS=false N_TRAJECTORIES=1) ;;
      enumerated) FLAGS=(USE_TRAJECTORY_RESETS=false USE_ENUMERATED_RESETS=true  N_TRAJECTORIES=1) ;;
      *) echo "error: unknown method '$m'" >&2; exit 1 ;;
    esac

    if [ "$KIND" = pilot ]; then
      RUN_DIR="$EXP/$lat/pilot/$m"
      BD=$BOND_DIM
    else
      RUN_DIR="$EXP/$lat/$m"
      if [ -n "$BOND_DIM" ]; then
        BD=$BOND_DIM
      else
        BD=$(python3 -c "
import json, sys
c = json.load(open('$EXP/config.json'))
print(c.get('bond_dim', {}).get('$lat', {}).get('$m', ''))" 2>/dev/null || true)
        if [ -z "$BD" ]; then
          echo "skip $lat/$m: no bond_dim for it in $EXP/config.json (run Phase 1, or pass BOND_DIM=)" >&2
          continue
        fi
      fi
    fi

    JOB_NAME="exp1_${KIND:0:4}_${lat}_${m}"
    # Never two jobs in one folder: they would interleave stdout.out, run.h5
    # and checkpoint.npz. Exact name match (-x), not substring -- 2x2 is a
    # substring of 12x2. FORCE does not override this.
    if squeue -h -u "$USER" -o %j | grep -qx "$JOB_NAME"; then
      echo "skip $RUN_DIR: job $JOB_NAME is already queued or running"
      continue
    fi
    if [ "$FORCE" != 1 ] && [ -f "$RUN_DIR/stdout.out" ] && \
       grep -q "Job finished with exit code: 0" "$RUN_DIR/stdout.out"; then
      echo "skip $RUN_DIR: already finished cleanly (FORCE=1 to resubmit)"
      continue
    fi

    CMD=(./submit.sh
      NAME="$JOB_NAME" TIME="$TIME"
      LX="$LX" LY="$LY" TRIALS="$TRIALS" SEED="$SEED" MAXITER="$MAXITER"
      HOWOFTEN_TOSAVE="$HOWOFTEN_TOSAVE" RESET_LAYERS="$RESET_LAYERS"
      N_SPREAD_RESETS="$N_SPREAD_RESETS" BOND_DIM="$BD" USE_OPTIMAL_ORDERING=true
      "${FLAGS[@]}"
      TRACK_PARAMS=true TRACK_GRADS=true TRACK_BOND_DIM=false
      RUN_DIR="$RUN_DIR" CHECKPOINT="$RUN_DIR/checkpoint.npz")
    if [ "$DRY_RUN" = 1 ]; then
      echo "${CMD[*]}"
    else
      mkdir -p "$RUN_DIR"
      "${CMD[@]}"
    fi
  done
done
