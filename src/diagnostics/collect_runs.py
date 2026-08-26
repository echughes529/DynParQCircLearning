"""Scrape completed training runs out of logs/ into one comparison table.

Run:
    python -m src.diagnostics.collect_runs --match bdladder
    python -m src.diagnostics.collect_runs --match abtest --out outputs/traj_vs_purified/runs.csv

Each training job leaves a per-run directory under logs/ holding stdout (which
carries the resolved settings), stderr (which carries tqdm's timing), and -- for
runs made after checkpointing was added -- checkpoint.npz with the per-step wall
clock and the energy history.

Reading those back by hand across a dozen jobs is where transcription errors come
from, so every number quoted in the writeup comes through here instead.
"""

import argparse
import csv
import os
import re

import numpy as np

LOGS = "logs"

# The settings dict is printed with plain print(self.__dict__), which contains a
# lattice object repr and so cannot be eval'd; pull the fields out individually.
SETTING_RE = {
    "Lx": r"'Lx': (\d+)",
    "Ly": r"'Ly': (\d+)",
    "nlayers": r"'nlayers': (\d+)",
    "bond_dim": r"'bond_dim': (\d+)",
    "trials": r"'trials': (\d+)",
    "maxiter": r"'maxiter': (\d+)",
    "seed": r"'seed': (\d+)",
    "total_resets": r"'total_resets': (\d+)",
    "n_mps_qubits": r"'n_mps_qubits': (\d+)",
    "nancillas": r"'nancillas': (\d+)",
    "nparams": r"'nparams': (\d+)",
}


def E0(Lx, Ly):
    return -(Lx * Ly + (Lx - 1) * (Ly - 1))


def _peak_gib(npz, key):
    """Max of a sampled byte trace, in GiB, or None if it is empty or all NaN."""
    if key not in npz.files:
        return None
    arr = np.asarray(npz[key], dtype=float)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None
    return float(np.nanmax(arr)) / 2 ** 30


def parse_memlog(run_dir):
    """Peak memory from the job script's sampler, as an external cross-check.

    Three figures with three different meanings, and only two of them are about
    this job:

      memlog_node_ram_peak_mb  node-wide `free -m`. NOT this job's: two arms
                               sharing a node report each other's memory, which
                               is how the enum and pur runs on crannog06 both
                               came to report 21034 MB.
      memlog_proc_hwm_mb       VmHWM of the python process. Per-job, honest.
      memlog_gpu_peak_mb       nvidia-smi, cgroup-isolated so per-job, but only
                               meaningful with preallocation off -- otherwise it
                               is a flat 75% of the card for every arm alike.
    """
    out = {"memlog_node_ram_peak_mb": None, "memlog_proc_hwm_mb": None,
           "memlog_gpu_peak_mb": None, "memlog_cgroup_peak_mb": None,
           "sacct_maxrss": None}
    path = os.path.join(run_dir, "memlog.txt")
    if not os.path.exists(path):
        return out
    with open(path, errors="replace") as fh:
        text = fh.read()

    for field, col in (("ram_mb", "memlog_node_ram_peak_mb"),
                       ("proc_hwm_mb", "memlog_proc_hwm_mb"),
                       ("gpu_mb", "memlog_gpu_peak_mb")):
        vals = [int(v) for v in re.findall(rf"\b{field}=(\d+)\b", text)]
        if vals:
            out[col] = max(vals)

    m = re.search(r"cgroup_peak_mb=(\d+)", text)
    if m:
        out["memlog_cgroup_peak_mb"] = int(m.group(1))
    # sacct prints MaxRSS as e.g. "12345.67M"; take the largest step's.
    rss = re.findall(r"\b(\d+(?:\.\d+)?)M\b", text)
    if rss:
        out["sacct_maxrss"] = max(float(v) for v in rss)
    return out


def parse_run(run_dir):
    out_path = os.path.join(run_dir, "stdout.out")
    if not os.path.exists(out_path):
        return None
    with open(out_path, errors="replace") as fh:
        stdout = fh.read()

    row = {"run": os.path.basename(run_dir)}
    for key, pattern in SETTING_RE.items():
        m = re.search(pattern, stdout)
        row[key] = int(m.group(1)) if m else None
    if row["Lx"] is None:
        return None

    # Three arms, not two. Reading only use_trajectory_resets labelled every
    # enumerated run "pur", because enumerated runs have it False as well. The
    # checkpoint's own reset_mode overrides this below where it exists; this
    # regex is the fallback for runs made before checkpoints carried it.
    traj = re.search(r"'use_trajectory_resets': (True|False)", stdout)
    enum = re.search(r"'use_enumerated_resets': (True|False)", stdout)
    if enum and enum.group(1) == "True":
        row["mode"] = "enum"
    elif traj:
        row["mode"] = "traj" if traj.group(1) == "True" else "pur"
    else:
        row["mode"] = None
    m = re.search(r"'n_trajectories': (\d+)", stdout)
    row["n_trajectories"] = int(m.group(1)) if m else None
    m = re.search(r"Max claw distance: (\d+)", stdout)
    row["max_claw"] = int(m.group(1)) if m else None
    row["device"] = "gpu" if "CudaDevice" in stdout else ("cpu" if "CpuDevice" in stdout else "?")
    m = re.search(r"GPU info:\s+(.+)", stdout)
    row["gpu"] = m.group(1).strip() if m else ""

    m = re.search(r"(\d+)/(\d+) trials \(([\d.]+)%\) within 0\.1%", stdout)
    if m:
        row["converged"] = int(m.group(1))
        row["converged_frac"] = float(m.group(3)) / 100.0
    else:
        row["converged"] = row["converged_frac"] = None

    row["lattice"] = f"{row['Lx']}x{row['Ly']}"
    row["E0"] = E0(row["Lx"], row["Ly"])

    # tqdm writes to stderr with carriage returns; the last complete bar carries
    # the total elapsed time and the achieved rate.
    err_path = os.path.join(run_dir, "stderr.err")
    row["elapsed"] = row["it_per_s"] = row["s_per_it"] = None
    row["finished"] = False
    if os.path.exists(err_path):
        with open(err_path, errors="replace") as fh:
            err = fh.read().replace("\r", "\n")
        bars = re.findall(r"(\d+)/(\d+) \[([\d:]+)<([^,]+), *([\d.]+)(it/s|s/it)", err)
        if bars:
            done, total, elapsed, _, rate, unit = bars[-1]
            row["finished"] = done == total
            row["steps_done"] = int(done)
            parts = [int(p) for p in elapsed.split(":")]
            secs = 0
            for p in parts:
                secs = secs * 60 + p
            row["elapsed"] = secs
            row["it_per_s"] = float(rate) if unit == "it/s" else 1.0 / float(rate)
            row["s_per_it"] = 1.0 / row["it_per_s"]
        m = re.findall(r"Current value: (-?[\d.]+|nan)", err)
        row["last_reported_value"] = None if not m else (float(m[-1]) if m[-1] != "nan" else float("nan"))

    # The checkpoint carries the honest per-step timing: tqdm's rate includes the
    # first step's JIT compilation, which can be minutes. The job script points
    # DPQC_OUTDIR at the run's plots/ subdirectory, so that is where the
    # checkpoint lands; the run root is checked too for hand-run jobs.
    ckpt = os.path.join(run_dir, "plots", "checkpoint.npz")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "checkpoint.npz")
    row["checkpoint"] = ckpt if os.path.exists(ckpt) else None
    row["compile_s"] = row["warm_step_s"] = None
    row["best_energy"] = None
    row["peak_gpu_gib"] = row["peak_host_rss_gib"] = None
    row["n_nonfinite_steps"] = None
    if os.path.exists(ckpt):
        try:
            d = np.load(ckpt, allow_pickle=True)
            st = d["step_times"]
            row["compile_s"] = float(st[0])
            warm = st[1:][np.isfinite(st[1:])]
            row["warm_step_s"] = float(np.median(warm)) if warm.size else None
            energies = d["all_energies"]
            filled = int(d["nsnapshots_filled"])
            if filled:
                # nanmin, not min: optimize() freezes a trial whose gradient goes
                # non-finite but still records its NaN value, and a single NaN
                # would otherwise swallow the whole run's best energy.
                block = energies[:, :filled]
                row["best_energy"] = float(np.nanmin(block)) if np.any(np.isfinite(block)) else None
                row["nonfinite_entries"] = int(np.sum(~np.isfinite(block)))
            row["has_params"] = bool(np.any(d["all_params"]))

            # Checkpoints written since the three-way study are self-describing:
            # they carry the arm, the chain geometry and the memory trace, so
            # these no longer have to be recovered by regexing stdout -- and
            # stdout is not guaranteed to survive next to the checkpoint.
            if "reset_mode" in d.files:
                row["mode"] = str(d["reset_mode"])
            for key in ("n_mps_qubits", "total_resets", "nancillas", "bond_dim",
                        "trials", "seed", "n_trajectories"):
                if key in d.files:
                    v = int(d[key])
                    if v >= 0:
                        row[key] = v
            if "n_nonfinite_steps" in d.files:
                row["n_nonfinite_steps"] = int(d["n_nonfinite_steps"])
            # In-process allocator high-water mark: per-process and immune to
            # both preallocation and node co-tenancy, unlike anything the job
            # script can see from outside.
            row["peak_gpu_gib"] = _peak_gib(d, "gpu_peak_bytes")
            row["peak_host_rss_gib"] = _peak_gib(d, "host_rss_bytes")
        except Exception as exc:
            row["ckpt_error"] = str(exc)

    row.update(parse_memlog(run_dir))

    if row["best_energy"] is not None and row["E0"]:
        row["rel_error"] = abs(row["best_energy"] - row["E0"]) / abs(row["E0"])
    else:
        row["rel_error"] = None
    if row["warm_step_s"] and row["trials"]:
        row["warm_step_per_trial_s"] = row["warm_step_s"] / row["trials"]
    else:
        row["warm_step_per_trial_s"] = None
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="", help="substring the run directory must contain")
    parser.add_argument("--out", default="outputs/three_resets/runs.csv")
    args = parser.parse_args()

    rows = []
    for name in sorted(os.listdir(LOGS)):
        if args.match and args.match not in name:
            continue
        path = os.path.join(LOGS, name)
        if not os.path.isdir(path):
            continue
        row = parse_run(path)
        if row:
            rows.append(row)

    if not rows:
        print("no matching runs found")
        return

    rows.sort(key=lambda r: (r["lattice"], r["mode"] or "", r["bond_dim"] or 0,
                             r.get("seed") or 0))
    hdr = (f"{'lattice':>8} {'mode':>5} {'bd':>5} {'seed':>6} {'chain':>6} {'trials':>7} "
           f"{'steps':>7} {'warm s/step':>12} {'/trial':>8} {'elapsed':>9} "
           f"{'GPU GiB':>8} {'RSS GiB':>8} {'best E':>11} {'rel err':>10} {'conv':>6} "
           f"{'nan':>4} {'dev':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(key, fmt, scale=1.0):
            v = r.get(key)
            return "-" if v is None else format(v * scale, fmt)
        elapsed = "-" if r["elapsed"] is None else f"{r['elapsed'] // 60:d}m{r['elapsed'] % 60:02d}"
        conv = "-" if r["converged"] is None else f"{r['converged']}/{r['trials']}"
        print(f"{r['lattice']:>8} {str(r['mode']):>5} {str(r['bond_dim']):>5} "
              f"{str(r.get('seed')):>6} {str(r['n_mps_qubits']):>6} {str(r['trials']):>7} "
              f"{str(r.get('steps_done', '-')):>7} "
              f"{f('warm_step_s', '.3f'):>12} {f('warm_step_per_trial_s', '.4f'):>8} {elapsed:>9} "
              f"{f('peak_gpu_gib', '.2f'):>8} {f('peak_host_rss_gib', '.2f'):>8} "
              f"{f('best_energy', '.6f'):>11} {f('rel_error', '.2e'):>10} {conv:>6} "
              f"{str(r.get('n_nonfinite_steps', '-')):>4} {r['device']:>4}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
