#!/bin/bash
# Submit Experiment 1.1 training runs (Lx x 3): one Slurm job per (lattice, method).
#
# Usage (from anywhere):
#   experiments/experiment_1.1_scaling_system_size_nx3/submit_all.sh LATTICES="3x3 4x3"
#
# 2026-11-16: only policy A (caps_192_256, the default) is run; policy B
# (caps_64_96) and the KIND=calib runs were dropped (PLAN.md update note). The
# POLICY/KIND machinery below is kept as written.
#   ... METHODS="mc_n1 mc_n8"   only these arms
#   ... DRY_RUN=1               print the submit commands, submit nothing
#   ... FORCE=1                 resubmit runs that already finished cleanly
#   ... TIME=12:00:00 MEM=160G BOND_DIM=...   override config.json for this call
#
# POLICY (required) picks the cap policy and its folder (PLAN.md section 3):
#   caps_64_96    B: below the need   -> <POLICY>/<lattice>/<method>/
#   caps_192_256  A: at the need      -> <POLICY>/<lattice>/<method>/
# KIND:
#   production    10 trials, 3000 steps (the Experiment 1 settings)
#   calib         <POLICY>/<lattice>/<method>_calib/, with the trial and step
#                 counts from config.json "calibration" (lattice and methods too)
#
# Everything per policy comes from <POLICY>/config.json:
#   bond_dim[lattice][method] or bond_dim_default[method]
#   walltime[lattice][method] or walltime_default
#   mem[lattice] (else 160G from Lx >= 24, else the batch script's 64G)
#   shared_runs[lattice][method] = other policy: no job; the folder becomes a
#     symlink to that policy's run (3x3 ancilla-free arms, identical under A and B)
#
# Copied from experiments/experiment_1_scaling_system_size/submit_all.sh. What differs:
#   EXP path and the POLICY level; lattice list (qubit-matched, PLAN.md section 5);
#   walltime/MEM/caps read from config.json instead of walltime_for(); job names
#   exp11_<policy>_<kind>_<lattice>_<method>; KIND=calib instead of KIND=pilot;
#   shared_runs symlinks; Lx x 3 reset rule and column ordering via the same
#   N_SPREAD_RESETS / USE_LADDER_ORDERING settings.
# Experiment 1's script is untouched.
#
# Every setting is passed through ./submit.sh, so it travels with the job and
# later code edits cannot change a queued run's parameters (see run_config.py).

set -euo pipefail
REPO=/home/s1931382/DynParQCircLearning
EXP=experiments/experiment_1.1_scaling_system_size_nx3
cd "$REPO"

POLICY=caps_192_256
KIND=production
LATTICES=""
METHODS=""
TIME=""
BOND_DIM=""
MEM=""
DRY_RUN=0
FORCE=0
for arg in "$@"; do
  [[ "$arg" == *=* ]] || { echo "error: arguments must be KEY=VALUE, got '$arg'" >&2; exit 1; }
  declare "${arg%%=*}=${arg#*=}"
done

case "$POLICY" in
  caps_192_256) ;;
  *) echo "error: POLICY must be caps_192_256 (policy B was dropped 2026-11-16)" >&2; exit 1 ;;
esac
PDIR="$EXP/$POLICY"
CFG="$PDIR/config.json"
[ -f "$CFG" ] || { echo "error: $CFG not found" >&2; exit 1; }

# Read one value from config.json; prints nothing if it is missing or null.
cfg() {
  python3 - "$CFG" "$@" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    c = c.get(k) if isinstance(c, dict) else None
    if c is None:
        break
print("" if c is None else (" ".join(map(str, c)) if isinstance(c, list) else c))
PY
}

# Fixed settings for every run of this experiment (PLAN.md, "What stays the same").
TRIALS=10
MAXITER=3000
SEED=1234
N_SPREAD_RESETS=4
RESET_LAYERS="[1]"
HOWOFTEN_TOSAVE=10
SUFFIX=""

if [ "$KIND" = calib ]; then
  TRIALS=$(cfg calibration trials)
  MAXITER=$(cfg calibration maxiter)
  if [ -z "$TRIALS" ] || [ -z "$MAXITER" ]; then
    echo "error: set calibration.trials and calibration.maxiter in $CFG (from Phase 1)" >&2; exit 1
  fi
  LATTICES=${LATTICES:-$(cfg calibration lattice)}
  METHODS=${METHODS:-$(cfg calibration methods)}
  SUFFIX=_calib
elif [ "$KIND" != production ]; then
  echo "error: KIND must be production or calib" >&2; exit 1
fi
LATTICES=${LATTICES:-"3x3 4x3 5x3 6x3 8x3 12x3 17x3 24x3 34x3 48x3"}
METHODS=${METHODS:-"mc_n1 mc_n8 purified enumerated"}

# JAX otherwise caps itself at 75% of the card (36 GB on an A40), so an
# "out of memory" would mean 36 GB, not the 48 GB the stopping rule is about.
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
    RUN_DIR="$PDIR/$lat/$m$SUFFIX"

    # Runs identical under both policies (PLAN.md section 3) are made once and
    # linked, so the same run is never trained twice.
    SHARED=$( [ "$KIND" = production ] && cfg shared_runs "$lat" "$m" || true )
    if [ -n "$SHARED" ]; then
      TARGET="../../$SHARED/$lat/$m"
      if [ "$DRY_RUN" = 1 ]; then
        echo "link $RUN_DIR -> $TARGET (shared with $SHARED)"
      elif [ -L "$RUN_DIR" ]; then
        echo "skip $RUN_DIR: already linked to $(readlink "$RUN_DIR")"
      elif [ -e "$RUN_DIR" ]; then
        echo "error: $RUN_DIR exists and is not a link; shared_runs says it should be one" >&2; exit 1
      else
        mkdir -p "$PDIR/$lat"
        ln -s "$TARGET" "$RUN_DIR"
        echo "linked $RUN_DIR -> $TARGET"
      fi
      continue
    fi

    BD=${BOND_DIM:-$(cfg bond_dim "$lat" "$m")}
    BD=${BD:-$(cfg bond_dim_default "$m")}
    [ -n "$BD" ] || { echo "skip $lat/$m: no bond_dim in $CFG" >&2; continue; }

    RUN_TIME=${TIME:-$(cfg walltime "$lat" "$m")}
    RUN_TIME=${RUN_TIME:-$(cfg walltime_default)}
    [ -n "$RUN_TIME" ] || { echo "skip $lat/$m: no walltime in $CFG" >&2; continue; }

    RUN_MEM=${MEM:-$(cfg mem "$lat")}
    if [ -z "$RUN_MEM" ] && [ "$LX" -ge 24 ]; then RUN_MEM=160G; fi

    JOB_NAME="exp11_${POLICY#caps_}_${KIND:0:4}_${lat}_${m}"
    # Never two jobs in one folder: they would interleave stdout.out, run.h5
    # and checkpoint.npz. Exact name match (-x), not substring. FORCE does not
    # override this.
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
      NAME="$JOB_NAME" TIME="$RUN_TIME" ${RUN_MEM:+MEM="$RUN_MEM"}
      LX="$LX" LY="$LY" TRIALS="$TRIALS" SEED="$SEED" MAXITER="$MAXITER"
      HOWOFTEN_TOSAVE="$HOWOFTEN_TOSAVE" RESET_LAYERS="$RESET_LAYERS"
      N_SPREAD_RESETS="$N_SPREAD_RESETS" BOND_DIM="$BD" USE_OPTIMAL_ORDERING=true USE_LADDER_ORDERING=true
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
