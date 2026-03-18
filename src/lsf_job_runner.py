# (C) Copyright IBM 2025.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
LSF Job Runner for DPQC Examples

This module provides utilities to submit and manage LSF jobs for running
the example scripts in src/examples on a cluster. Supports both individual
job submission and parallel execution using Dask.

Usage:
    # Submit a single job
    python -m src.lsf_job_runner --script find_gs_tc_example --queue normal --memory 8GB

    # Submit multiple jobs with parameter sweep
    python -m src.lsf_job_runner --script find_gibbs_fidelity_examples --queue normal --array 1-10

    # Use Dask for distributed execution
    python -m src.lsf_job_runner --script find_gibbs_varqite_example --use-dask --n-workers 10
"""

import os
import sys
import argparse
import subprocess
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
import time


# Default hosts for LSF cluster (customize as needed)
DEFAULT_HOSTS = []


def setup_logging(log_dir: str) -> None:
    """
    Configure logging to write to both console and file.
    
    Args:
        log_dir: Directory for log files
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")


class LSFJobRunner:
    """Helper class to submit and manage LSF jobs for DPQC examples."""
    
    def __init__(self, 
                 queue: str = "normal",
                 memory: str = "8GB",
                 walltime: str = "24:00",
                 cores: int = 1,
                 gpu: bool = False,
                 project_root: Optional[str] = None):
        """
        Initialize LSF job runner.
        
        Args:
            queue: LSF queue name
            memory: Memory requirement (e.g., "8GB", "16GB")
            walltime: Wall time limit (e.g., "24:00", "48:00")
            cores: Number of CPU cores
            gpu: Whether to request GPU resources
            project_root: Root directory of the project (defaults to parent of src/)
        """
        self.queue = queue
        self.memory = memory
        self.walltime = walltime
        self.cores = cores
        self.gpu = gpu
        
        if project_root is None:
            # Assume this file is in src/, so project root is parent
            self.project_root = Path(__file__).parent.parent.absolute()
        else:
            self.project_root = Path(project_root).absolute()
            
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def create_job_script(self, 
                         script_name: str,
                         script_args: Optional[List[str]] = None,
                         job_name: Optional[str] = None,
                         output_file: Optional[str] = None,
                         error_file: Optional[str] = None,
                         array_spec: Optional[str] = None,
                         env_vars: Optional[Dict[str, str]] = None) -> str:
        """
        Create an LSF job submission script.
        
        Args:
            script_name: Name of the Python script to run (without .py extension)
            script_args: Additional arguments to pass to the script
            job_name: Name for the LSF job
            output_file: Path for stdout log
            error_file: Path for stderr log
            array_spec: Array job specification (e.g., "1-10" for 10 jobs)
            env_vars: Environment variables to set
            
        Returns:
            Path to the created job script
        """
        if job_name is None:
            job_name = f"dpqc_{script_name}"
            
        if output_file is None:
            output_file = str(self.logs_dir / f"{job_name}_%J.out")
            
        if error_file is None:
            error_file = str(self.logs_dir / f"{job_name}_%J.err")
            
        script_path = self.project_root / "src" / "examples" / f"{script_name}.py"
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Build LSF directives
        lsf_directives = [
            f"#BSUB -J {job_name}",
            f"#BSUB -q {self.queue}",
            f"#BSUB -n {self.cores}",
            f'#BSUB -R "rusage[mem={self.memory}]"',
            f"#BSUB -M {self.memory}",
            f"#BSUB -W {self.walltime}",
            f"#BSUB -o {output_file}",
            f"#BSUB -e {error_file}",
        ]
        
        if array_spec:
            lsf_directives.append(f"#BSUB -J {job_name}[{array_spec}]")
            
        if self.gpu:
            lsf_directives.append("#BSUB -gpu num=1")
            
        # Build job script content
        job_script_lines = [
            "#!/bin/bash",
            "",
            "# LSF directives",
        ] + lsf_directives + [
            "",
            "# Load required modules (customize as needed)",
            "# module load python/3.9",
            "# module load cuda/11.8",
            "",
            "# Set environment variables",
            f"export PYTHONPATH={self.project_root}:$PYTHONPATH",
        ]
        
        if env_vars:
            for key, value in env_vars.items():
                job_script_lines.append(f"export {key}={value}")
                
        job_script_lines.extend([
            "",
            "# Activate virtual environment if needed",
            "# source /path/to/venv/bin/activate",
            "",
            f"# Change to project directory",
            f"cd {self.project_root}",
            "",
            "# Run the Python script",
            f"python -m src.examples.{script_name}",
        ])
        
        if script_args:
            job_script_lines[-1] += " " + " ".join(script_args)
            
        job_script_lines.append("")
        
        # Write job script
        job_script_path = self.logs_dir / f"job_{job_name}.sh"
        with open(job_script_path, 'w') as f:
            f.write('\n'.join(job_script_lines))
            
        # Make executable
        os.chmod(job_script_path, 0o755)
        
        return str(job_script_path)
    
    def submit_job(self, job_script_path: str) -> Optional[str]:
        """
        Submit a job to LSF.
        
        Args:
            job_script_path: Path to the job script
            
        Returns:
            Job ID if successful, None otherwise
        """
        try:
            # Use bsub with input redirection
            with open(job_script_path, 'r') as f:
                result = subprocess.run(
                    ["bsub"],
                    stdin=f,
                    capture_output=True,
                    text=True,
                    check=True
                )
            
            # Parse job ID from output (typically "Job <12345> is submitted...")
            output = result.stdout
            if "Job <" in output:
                job_id = output.split("Job <")[1].split(">")[0]
                print(f"✓ Job submitted successfully: {job_id}")
                print(f"  Script: {job_script_path}")
                return job_id
            else:
                print(f"✓ Job submitted: {output}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to submit job: {e}")
            print(f"  Error: {e.stderr}")
            return None
        except FileNotFoundError:
            print(f"✗ bsub command not found. Are you on an LSF cluster?")
            return None
    
    def check_job_status(self, job_id: str) -> Optional[str]:
        """
        Check the status of an LSF job.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Job status string or None if job not found
        """
        try:
            result = subprocess.run(
                ["bjobs", job_id],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                # Parse status from second line
                status = lines[1].split()[2]
                return status
            return None
            
        except subprocess.CalledProcessError:
            return None
    
    def wait_for_jobs(self, job_ids: List[str], poll_interval: int = 60):
        """
        Wait for a list of jobs to complete.
        
        Args:
            job_ids: List of job IDs to monitor
            poll_interval: Time in seconds between status checks
        """
        print(f"\nMonitoring {len(job_ids)} jobs...")
        
        pending_jobs = set(job_ids)
        
        while pending_jobs:
            time.sleep(poll_interval)
            
            completed = []
            for job_id in pending_jobs:
                status = self.check_job_status(job_id)
                if status is None or status == "DONE" or status == "EXIT":
                    completed.append(job_id)
                    print(f"  Job {job_id}: {status or 'COMPLETED'}")
                    
            for job_id in completed:
                pending_jobs.remove(job_id)
                
            if pending_jobs:
                print(f"  {len(pending_jobs)} jobs still running...")


class DaskLSFRunner:
    """Helper class to run DPQC examples using Dask with LSF backend."""
    
    def __init__(self, 
                 n_workers: int = 10,
                 cores_per_worker: int = 1,
                 memory_per_worker: str = "8GB",
                 queue: str = "normal",
                 walltime: str = "48:00",
                 project_root: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 hosts: Optional[List[str]] = None):
        """
        Initialize Dask LSF runner.
        
        Args:
            n_workers: Number of Dask workers
            cores_per_worker: CPU cores per worker
            memory_per_worker: Memory per worker
            queue: LSF queue name
            walltime: Wall time limit
            project_root: Root directory of the project
            log_dir: Directory for logs (created if None)
            hosts: List of allowed hostnames for job placement
        """
        self.n_workers = n_workers
        self.cores_per_worker = cores_per_worker
        self.memory_per_worker = memory_per_worker
        self.queue = queue
        self.walltime = walltime
        self.hosts = hosts or DEFAULT_HOSTS
        
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.absolute()
        else:
            self.project_root = Path(project_root).absolute()
        
        if log_dir is None:
            self.log_dir = str(self.project_root / "logs" / f"dask_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.log_dir = log_dir
            
        os.makedirs(self.log_dir, exist_ok=True)
            
    def setup_cluster(self):
        """
        Set up a Dask cluster with LSF backend.
        
        Returns:
            Tuple of (client, cluster)
        """
        try:
            from dask_jobqueue import LSFCluster
            from dask.distributed import Client
        except ImportError:
            raise ImportError(
                "dask-jobqueue is required for Dask support. "
                "Install with: pip install dask-jobqueue dask distributed"
            )
        
        # Build host constraint if hosts are specified
        job_extra_directives = [
            f'-R "rusage[mem={self.memory_per_worker}]"',
            f"-n {self.cores_per_worker}",
        ]
        
        if self.hosts:
            host_constraint = " || ".join(f"hname == {h}" for h in self.hosts)
            job_extra_directives.append(f'-R "{host_constraint}"')
        
        cluster = LSFCluster(
            queue=self.queue,
            cores=self.cores_per_worker,
            processes=1,
            memory=self.memory_per_worker,
            walltime=self.walltime,
            job_extra_directives=job_extra_directives,
            log_directory=self.log_dir,
            scheduler_options={"dashboard_address": ":0"},
        )
        
        cluster.scale(jobs=self.n_workers)
        client = Client(cluster)
        
        logging.info(f"Dask cluster created with {self.n_workers} workers")
        logging.info(f"Dashboard: {client.dashboard_link}")
        print(f"✓ Dask cluster created with {self.n_workers} workers")
        print(f"  Dashboard: {client.dashboard_link}")
        
        return client, cluster
    
    def run_parallel_tasks(self, 
                          task_func: Callable[..., Any],
                          task_args_list: List,
                          batch_size: int = 10_000,
                          description: str = "tasks") -> Optional[List]:
        """
        Run tasks in parallel using Dask with batching support.
        
        Args:
            task_func: Function to execute
            task_args_list: List of argument tuples for each task
            batch_size: Maximum number of tasks per batch
            description: Description of the tasks for logging
            
        Returns:
            List of results or None on failure
        """
        client, cluster = self.setup_cluster()
        
        futures = []
        results = []
        
        try:
            logging.info(f"Submitting tasks for {description} ({len(task_args_list)} total)")
            print(f"Submitting tasks for {description} ({len(task_args_list)} total)")
            
            if len(task_args_list) <= batch_size:
                # Single batch processing
                futures = client.map(task_func, task_args_list)
                logging.info(f"{len(futures)} tasks submitted.")
                print(f"{len(futures)} tasks submitted.")
                
                from dask.distributed import wait
                wait(futures)
                results = client.gather(futures, errors="raise")
                
            else:
                # Multi-batch processing
                total_batches = (len(task_args_list) + batch_size - 1) // batch_size
                logging.info(f"Processing in {total_batches} batches (batch_size={batch_size})")
                print(f"Processing in {total_batches} batches (batch_size={batch_size})")
                
                for i in range(total_batches):
                    start = i * batch_size
                    end = min(start + batch_size, len(task_args_list))
                    batch = task_args_list[start:end]
                    
                    logging.info(f"Batch {i + 1}/{total_batches} ({len(batch)} tasks)")
                    print(f"  Batch {i + 1}/{total_batches} ({len(batch)} tasks)")
                    
                    futures = client.map(task_func, batch)
                    
                    from dask.distributed import wait
                    wait(futures)
                    batch_results = client.gather(futures, errors="raise")
                    results.extend(batch_results)
                    
                    logging.info(f"Batch {i + 1} done — {len(results)} results so far.")
                    print(f"  Batch {i + 1} done — {len(results)} results so far.")
            
            logging.info("All tasks completed.")
            print("All tasks completed.")
            
            # Save summary
            summary = {
                "description": description,
                "n_tasks": len(task_args_list),
                "n_results": len(results),
                "n_workers": self.n_workers,
                "batch_size": batch_size,
                "status": "completed",
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            summary_path = os.path.join(self.log_dir, "run_summary.json")
            with open(summary_path, "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
            logging.info(f"Summary saved → {summary_path}")
            print(f"Summary saved → {summary_path}")
            
            return results
            
        except KeyboardInterrupt:
            logging.warning("Interrupted by user. Cancelling tasks and shutting down...")
            print("\n[INTERRUPTED] Cancelling tasks and shutting down...")
            return results or None
            
        except Exception as exc:
            logging.error(f"Error during parallel execution: {exc}")
            print(f"\n[ERROR] {exc}")
            
            # Save crash log
            err_path = os.path.join(self.log_dir, "crash.log")
            try:
                with open(err_path, "w") as fh:
                    traceback.print_exc(file=fh)
                logging.error(f"Crash traceback → {err_path}")
                print(f"Crash traceback → {err_path}")
            except OSError as e:
                logging.error(f"Could not write crash log: {e}")
                print(f"[WARNING] Could not write crash log: {e}")
                traceback.print_exc()
            
            raise
            
        finally:
            # Cleanup
            try:
                client.cancel(futures, force=True)
            except Exception:
                pass
            
            try:
                cluster.scale(0)
                client.wait_for_workers(0, timeout=10)
            except Exception:
                pass
            
            try:
                client.close()
                cluster.close()
            except (AssertionError, Exception) as e:
                logging.warning(f"Cluster cleanup error (ignoring): {e}")
                print(f"[WARNING] Cluster cleanup error (ignoring): {e}")


def parallel_implementation(
    function: Callable[..., Any],
    info: Dict[str, Any],
    jobs: int = 30,
    batch_size: int = 10_000,
    log_dir: Optional[str] = None,
    run_config: Optional[dict] = None,
    hosts: Optional[List[str]] = None,
) -> Optional[list]:
    """
    Submit *function* over every element of ``info['params']`` on a
    Dask-on-LSF cluster.
    
    Args:
        function: Callable to execute on each parameter set
        info: Dictionary containing 'params', 'memory', and 'description'
        jobs: Number of Dask workers
        batch_size: Maximum tasks per batch
        log_dir: Directory for logs
        run_config: Configuration dict to save in summary
        hosts: List of allowed hostnames
    
    Returns:
        List of results or None on failure
    """
    if log_dir is not None:
        setup_logging(log_dir)
    
    memory = info.get("memory", "8GB")
    description = info.get("description", "parallel tasks")
    params = info["params"]
    
    runner = DaskLSFRunner(
        n_workers=jobs,
        cores_per_worker=1,
        memory_per_worker=memory,
        queue="normal",
        walltime="48:00",
        log_dir=log_dir,
        hosts=hosts
    )
    
    results = runner.run_parallel_tasks(
        task_func=function,
        task_args_list=params,
        batch_size=batch_size,
        description=description
    )
    
    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Submit DPQC example scripts as LSF jobs"
    )
    
    parser.add_argument(
        "--script",
        required=True,
        choices=["find_gs_tc_example", "find_gibbs_fidelity_examples", "find_gibbs_varqite_example"],
        help="Example script to run"
    )
    
    parser.add_argument("--queue", default="normal", help="LSF queue name")
    parser.add_argument("--memory", default="8GB", help="Memory requirement")
    parser.add_argument("--walltime", default="24:00", help="Wall time limit")
    parser.add_argument("--cores", type=int, default=1, help="Number of CPU cores")
    parser.add_argument("--gpu", action="store_true", help="Request GPU resources")
    parser.add_argument("--array", help="Array job specification (e.g., '1-10')")
    parser.add_argument("--job-name", help="Custom job name")
    parser.add_argument("--use-dask", action="store_true", help="Use Dask for parallel execution")
    parser.add_argument("--n-workers", type=int, default=10, help="Number of Dask workers")
    parser.add_argument("--wait", action="store_true", help="Wait for jobs to complete")
    parser.add_argument("--log-dir", help="Directory for logs")
    
    args = parser.parse_args()
    
    if args.use_dask:
        print("Setting up Dask cluster for parallel execution...")
        dask_runner = DaskLSFRunner(
            n_workers=args.n_workers,
            cores_per_worker=args.cores,
            memory_per_worker=args.memory,
            queue=args.queue,
            walltime=args.walltime,
            log_dir=args.log_dir
        )
        
        print(f"Note: Dask cluster is ready. Import and use in your script:")
        print(f"  from src.lsf_job_runner import DaskLSFRunner")
        print(f"  runner = DaskLSFRunner(n_workers={args.n_workers})")
        print(f"  client, cluster = runner.setup_cluster()")
        
    else:
        runner = LSFJobRunner(
            queue=args.queue,
            memory=args.memory,
            walltime=args.walltime,
            cores=args.cores,
            gpu=args.gpu
        )
        
        job_script = runner.create_job_script(
            script_name=args.script,
            job_name=args.job_name,
            array_spec=args.array
        )
        
        job_id = runner.submit_job(job_script)
        
        if job_id and args.wait:
            runner.wait_for_jobs([job_id])


if __name__ == "__main__":
    main()

# Made with Bob
