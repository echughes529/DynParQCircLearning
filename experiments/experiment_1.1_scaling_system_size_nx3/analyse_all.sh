#!/bin/bash
# Submit the post-training analysis for every finished Experiment 1.1 run.
#
# Usage:
#   experiments/experiment_1.1_scaling_system_size_nx3/analyse_all.sh
#   experiments/experiment_1.1_scaling_system_size_nx3/analyse_all.sh FORCE=1
# (2026-11-16: only policy A, POLICY=caps_192_256, the default, is run.)
#   ... DRY_RUN=1    print the commands, run nothing
#   ... TIME=08:00:00 BATCH=1 MEM=128G   for the MC analysis jobs
#   ... EXACT=0      MC: skip the exact re-evaluation (exact view left missing)
#
# Both energy views are produced (2026-11-16 decision): each method's logged
# training energy, and the exact energy of its trained parameters (exact reset
# channel = purified/enumerated). They coincide for purified/enumerated, which
# are analysed here (no GPU, seconds), as are failed runs. For the MC methods
# the exact view means re-evaluating every saved parameter vector over all 3^R
# branches (--sigma exact, which also gives the exact sigma), so they go to ANY
# free GPU (timing is irrelevant there). With EXACT=0 they run with --sigma none
# here instead: logged view only, exact energy and sigma recorded as missing.
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
# The exact-method analyses below run locally with `python`; without the venv
# that is the system Python (system matplotlib + ~/.local NumPy 2 -> ImportError).
# activate references unset variables, so relax `set -u` around it.
set +u
source /home/s1931382/dpqc_venv/bin/activate
set -u

POLICY=caps_192_256
EXACT=1
FORCE=0
DRY_RUN=0
TIME=04:00:00
# A40, as for training: at chi=192 the exact trajectory table needs 9-25 GiB,
# which does not fit the 11 GiB cards that a bare gpu:1 can land on.
GRES=gpu:a40:1
BATCH=""
MEM=""
for arg in "$@"; do
  [[ "$arg" == *=* ]] || { echo "error: arguments must be KEY=VALUE, got '$arg'" >&2; exit 1; }
  declare "${arg%%=*}=${arg#*=}"
done
case "$POLICY" in
  caps_192_256) ;;
  *) echo "error: POLICY must be caps_192_256 (policy B was dropped 2026-11-16)" >&2; exit 1 ;;
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
  elif [ "$EXACT" != 1 ]; then
    CMD=(python -m src.experiments.exp1_analyse_run "$d" --sigma none)
  else
    CMD=(./submit_diag.sh NAME="$AN_NAME" TIME="$TIME" GRES="$GRES" ${MEM:+MEM="$MEM"} --
         python -m src.experiments.exp1_analyse_run "$d" --sigma exact ${BATCH:+--batch "$BATCH"})
  fi
  if [ "$DRY_RUN" = 1 ]; then echo "${CMD[*]}"; else "${CMD[@]}"; fi
done
