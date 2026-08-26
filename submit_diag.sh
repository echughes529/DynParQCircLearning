#!/bin/bash
# Submit a diagnostics job, base64-encoding the command so commas survive.
#
# Usage:
#   ./submit_diag.sh NAME=bd_req TIME=08:00:00 -- python -m src.diagnostics.bond_dim_requirement --lattices 3x3,4x3
#
# Everything after `--` is the command. NAME= sets the job name, TIME= the
# walltime, MEM= the memory request; any other KEY=VALUE is exported as DPQC_KEY
# so diagnostics can pick up the same configuration knobs training runs use.
#
# WHY BASE64: sbatch --export treats commas as separators BETWEEN variables, so
# a command containing "--lattices 3x3,4x3" arrives truncated to "--lattices
# 3x3". The job then runs, exits 0, and silently measures the wrong thing.
# run_diag.sh accepts DIAG_CMD_B64 precisely to avoid that, and this wrapper
# always uses it rather than leaving the choice to whoever is submitting.

set -euo pipefail
cd /home/s1931382/DynParQCircLearning

NAME=""
TIME=""
MEM=""
GRES=""
PARTITION=""

while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; break; fi
  if [[ "$1" != *=* ]]; then
    echo "error: options before -- must be KEY=VALUE, got '$1'" >&2
    exit 1
  fi
  key=${1%%=*}
  val=${1#*=}
  case "$key" in
    NAME) NAME=$val ;;
    TIME) TIME=$val ;;
    MEM)  MEM=$val ;;
    # GRES= targets a different accelerator, e.g. gpu:nvidia_h200:1 for the
    # 141 GB cards. Timings from a different card are NOT comparable with the
    # A40 measurements -- use it for feasibility questions, not for cost ones.
    GRES) GRES=$val ;;
    PARTITION) PARTITION=$val ;;
    *)    export "DPQC_${key}=${val}" ;;
  esac
  shift
done

if [ $# -eq 0 ]; then
  echo "error: no command given; put it after --" >&2
  exit 1
fi

DIAG_CMD="$*"
NAME=${NAME:-dpqc_diag}

ARGS=(--job-name="$NAME")
[ -n "$TIME" ] && ARGS+=(--time="$TIME")
[ -n "$MEM" ] && ARGS+=(--mem="$MEM")
[ -n "$GRES" ] && ARGS+=(--gres="$GRES")
[ -n "$PARTITION" ] && ARGS+=(--partition="$PARTITION")

EXCLUDE=${EXCLUDE:-crannog05,crannog01}
[ -n "$EXCLUDE" ] && ARGS+=(--exclude="$EXCLUDE")

B64=$(printf '%s' "$DIAG_CMD" | base64 -w0)

echo "Submitting diagnostic '${NAME}'${TIME:+ (time=$TIME)}${MEM:+ (mem=$MEM)}"
echo "  command: ${DIAG_CMD}"
sbatch "${ARGS[@]}" --export=ALL,DIAG_CMD_B64="$B64" run_diag.sh
