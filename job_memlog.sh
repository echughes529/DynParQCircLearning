#!/bin/bash
# Shared per-job memory sampling for the Slurm job scripts.
#
# Source this, then:
#     start_memlog "$RUN_DIR"
#     python -m your.module &
#     PY_PID=$!; echo "$PY_PID" > "$RUN_DIR/py.pid"
#     wait $PY_PID; EXIT_CODE=$?
#     stop_memlog "$RUN_DIR" "$EXIT_CODE"
#
# WHY NOT `free -m`
# -----------------
# `free -m` is node-wide. Two jobs sharing a node report each other's memory:
# the enum and pur arms of the 2026-08-26 three-arm run both landed on
# crannog06 and both reported peak ram_mb=21034, which is the node's figure and
# belongs to neither of them. It is still logged here for continuity, but
# proc_hwm_mb -- VmHWM of the python process, its true peak RSS -- is the number
# to quote.
#
# GPU memory from nvidia-smi is cgroup-isolated under Slurm, so it IS per-job.
# But with XLA_PYTHON_CLIENT_PREALLOCATE at its default of true it reports a
# flat 75% of the card whatever the job actually uses, which is why the job
# scripts turn preallocation off. The authoritative per-arm figure is the
# in-process peak_bytes_in_use that optimize() records into the checkpoint;
# this sampler is the independent cross-check on it.

start_memlog() {
  local run_dir="$1"
  (
    while true; do
      ts=$(date +%H:%M:%S)
      ram=$(free -m | awk '/^Mem:/{print $3}')
      gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      rss="NA"; hwm="NA"
      if [ -f "${run_dir}/py.pid" ]; then
        pid=$(cat "${run_dir}/py.pid" 2>/dev/null)
        if [ -n "$pid" ] && [ -r "/proc/${pid}/status" ]; then
          rss=$(awk '/^VmRSS:/{print int($2/1024)}' "/proc/${pid}/status" 2>/dev/null)
          hwm=$(awk '/^VmHWM:/{print int($2/1024)}' "/proc/${pid}/status" 2>/dev/null)
        fi
      fi
      echo "${ts} ram_mb=${ram} proc_rss_mb=${rss:-NA} proc_hwm_mb=${hwm:-NA} gpu_mb=${gpu:-NA}"
      sleep 30
    done
  ) >> "${run_dir}/memlog.txt" 2>/dev/null &
  MEMLOG_PID=$!
}

stop_memlog() {
  local run_dir="$1"
  local exit_code="$2"
  kill "$MEMLOG_PID" 2>/dev/null

  # cgroup v2 keeps a high-water mark for the whole job step. This is the
  # accounting Slurm itself enforces the --mem limit against, so it is the
  # figure that decides whether a job survives.
  local cg_peak="NA"
  local cg_path
  cg_path=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)
  for candidate in "/sys/fs/cgroup${cg_path}/memory.peak" "/sys/fs/cgroup/memory.peak"; do
    if [ -r "$candidate" ]; then
      cg_peak=$(( $(cat "$candidate") / 1048576 ))
      break
    fi
  done

  {
    echo "[memlog final] exit_code=${exit_code}"
    echo "[memlog final] cgroup_peak_mb=${cg_peak}"
    echo "[memlog final] node_ram: $(free -h | awk '/^Mem:/{print "used="$3" avail="$7}')"
    echo "[memlog final] gpu: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo NA)"
    if command -v sacct >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
      echo "[memlog final] sacct:"
      sacct -j "${SLURM_JOB_ID}" --units=M \
            --format=JobID,JobName%24,MaxRSS,MaxVMSize,Elapsed,State 2>/dev/null
    fi
  } >> "${run_dir}/memlog.txt" 2>&1
}
