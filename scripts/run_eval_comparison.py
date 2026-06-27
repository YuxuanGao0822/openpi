#!/usr/bin/env python3
"""Orchestrator script to evaluate and compare multiple JAX Pi0Drift checkpoints
against the baseline 10-NFE pi0_base model in parallel across multiple GPUs.
"""

import argparse
import json
import logging
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_lock = threading.Lock()


def kill_port_owner(port: int):
    """Find and terminate any process listening on the given port."""
    try:
        cmd = ["lsof", "-t", f"-i:{port}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid:
                    logging.info(f"Port {port} in use by PID {pid}. Killing it...")
                    subprocess.run(["kill", "-9", pid], check=False)
            time.sleep(1.0)
    except Exception as e:
        logging.warning(f"Could not check or kill process on port {port}: {e}")


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 120.0) -> bool:
    """Wait for the port to start listening by checking if we fail to bind to it."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            # We succeeded in binding, which means NO server is listening yet
            s.close()
            time.sleep(1.0)
        except OSError:
            # We failed to bind, meaning the port is occupied and the server is listening
            s.close()
            return True
    return False


def load_summary(path: str) -> Optional[dict]:
    """Load evaluation summary from JSON file."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading summary from {path}: {e}")
        return None


def worker(
    gpu_id: int,
    port: int,
    q: queue.Queue,
    results: dict,
    args: argparse.Namespace
):
    """Worker thread that executes jobs on a specific GPU and port."""
    logger = logging.getLogger(f"Worker-GPU{gpu_id}")
    
    while not q.empty():
        try:
            job = q.get_nowait()
        except queue.Empty:
            break

        job_type = job["type"]
        suite = job["suite"]

        # Setup paths
        if job_type == "drift":
            step = job["step"]
            checkpoint_dir = job["checkpoint_dir"]
            summary_path = os.path.join(ROOT_DIR, "data", "libero", f"eval_drift_{step}_{suite}.json")
            logger.info(f"Starting Drift model (Step {step}) on suite {suite}")
        else:
            summary_path = os.path.join(ROOT_DIR, "data", "libero", f"eval_baseline_{suite}.json")
            logger.info(f"Starting Baseline model on suite {suite}")

        # 1. Free up port
        kill_port_owner(port)

        # 2. Build serve command
        cmd = [sys.executable, "scripts/serve_policy.py", "--port", str(port)]
        if job_type == "drift":
            cmd.extend([
                "policy:checkpoint",
                "--policy.config", args.checkpoint_config,
                "--policy.dir", checkpoint_dir
            ])
            log_name = f"server_drift_{step}_gpu{gpu_id}.log"
        else:
            if args.baseline_dir:
                cmd.extend([
                    "policy:checkpoint",
                    "--policy.config", args.baseline_config,
                    "--policy.dir", args.baseline_dir
                ])
            else:
                cmd.extend([
                    "--env", "libero",
                    "policy:default"
                ])
            log_name = f"server_baseline_gpu{gpu_id}.log"

        os.makedirs(os.path.join(ROOT_DIR, "data", "libero"), exist_ok=True)
        log_path = os.path.join(ROOT_DIR, "data", "libero", log_name)
        log_file = open(log_path, "w")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"
        env["MUJOCO_GL"] = "egl"

        # Append third_party/libero to PYTHONPATH to allow finding libero
        libero_path = os.path.join(ROOT_DIR, "third_party", "libero")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{libero_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = libero_path

        # Spawn Policy Server
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            env=env,
            cwd=ROOT_DIR
        )

        # Wait for server port
        if not wait_for_port(port, host=args.host, timeout=120.0):
            logger.error(f"Policy server failed to start on GPU {gpu_id} port {port}. Check log at {log_path}")
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            kill_port_owner(port)
            log_file.close()
            q.task_done()
            continue

        log_file.close()

        # 3. Run client evaluation script
        client_cmd = [
            sys.executable,
            "examples/libero/eval_with_timing.py",
            "--host", args.host,
            "--port", str(port),
            "--task-suite-name", suite,
            "--num-trials-per-task", str(args.num_trials),
            "--summary-out-path", summary_path
        ]

        logger.info(f"Running client for suite {suite} connecting to port {port}...")
        client_res = subprocess.run(client_cmd, cwd=ROOT_DIR, env=env, check=False)

        # 4. Clean up Policy Server
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning(f"Server on GPU {gpu_id} did not terminate. Killing forcefully...")
            proc.kill()
            proc.wait()
        kill_port_owner(port)

        # 5. Load and record outcomes
        if client_res.returncode == 0:
            summary = load_summary(summary_path)
            if summary:
                with results_lock:
                    if job_type == "drift":
                        if step not in results["drift"]:
                            results["drift"][step] = {}
                        results["drift"][step][suite] = summary
                    else:
                        results["baseline"][suite] = summary
                logger.info(f"Finished suite {suite} successfully. Recorded results.")
            else:
                logger.error(f"Summary JSON not found or empty at {summary_path}")
        else:
            logger.error(f"Client failed for suite {suite} with code {client_res.returncode}")

        q.task_done()


def generate_drift_ranking_report(drift_rankings: List[dict]) -> str:
    """Construct a Markdown ranking table for all evaluated checkpoints."""
    lines = []
    lines.append("# Checkpoint Success Rate Ranking (1-NFE Drift VLA)\n")
    lines.append("| Rank | Checkpoint Step | Successes / Episodes | Overall Success Rate | Avg Step Latency | Status |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for rank, r in enumerate(drift_rankings, start=1):
        status = "**BEST (KEPT)**" if rank == 1 else "Archived to .trash/"
        lines.append(
            f"| {rank} | **Step {r['step']}** | {r['successes']}/{r['episodes']} | **{r['success_rate']:.2f}%** | {r['avg_latency']:.2f} ms | {status} |"
        )
    return "\n".join(lines)


def generate_suite_comparison_report(
    step: int,
    suites: List[str],
    drift_suite_data: dict,
    baseline_summaries: dict
) -> str:
    """Construct a Markdown comparison table for a specific step against baseline."""
    lines = []
    lines.append(f"## Comparison Report: Step {step} (1-NFE) vs Baseline (10-NFE)\n")
    lines.append("| Task Suite | Model | Successes / Episodes (Rate) | Avg Step Latency | Speedup |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")

    total_drift_successes = 0
    total_drift_episodes = 0
    total_drift_latency_sum = 0.0
    total_drift_tasks = 0

    total_baseline_successes = 0
    total_baseline_episodes = 0
    total_baseline_latency_sum = 0.0
    total_baseline_tasks = 0

    for suite in suites:
        drift = drift_suite_data.get(suite)
        baseline = baseline_summaries.get(suite)

        drift_rate_str = "N/A"
        drift_latency_str = "N/A"
        if drift:
            drift_success = drift.get("total_successes", 0)
            drift_episodes = drift.get("total_episodes", 0)
            drift_rate = drift.get("success_rate", 0.0) * 100.0
            drift_rate_str = f"{drift_success}/{drift_episodes} ({drift_rate:.1f}%)"

            drift_latency = drift.get("avg_inference_time_ms", 0.0)
            drift_latency_str = f"{drift_latency:.2f} ms"

            total_drift_successes += drift_success
            total_drift_episodes += drift_episodes
            total_drift_latency_sum += drift_latency
            total_drift_tasks += 1

        baseline_rate_str = "N/A"
        baseline_latency_str = "N/A"
        if baseline:
            baseline_success = baseline.get("total_successes", 0)
            baseline_episodes = baseline.get("total_episodes", 0)
            baseline_rate = baseline.get("success_rate", 0.0) * 100.0
            baseline_rate_str = f"{baseline_success}/{baseline_episodes} ({baseline_rate:.1f}%)"

            baseline_latency = baseline.get("avg_inference_time_ms", 0.0)
            baseline_latency_str = f"{baseline_latency:.2f} ms"

            total_baseline_successes += baseline_success
            total_baseline_episodes += baseline_episodes
            total_baseline_latency_sum += baseline_latency
            total_baseline_tasks += 1

        speedup_str = "N/A"
        if drift and baseline:
            drift_latency = drift.get("avg_inference_time_ms", 0.0)
            baseline_latency = baseline.get("avg_inference_time_ms", 0.0)
            if drift_latency > 0:
                speedup = baseline_latency / drift_latency
                speedup_str = f"**{speedup:.2f}x**"

        lines.append(
            f"| **{suite}** | Pi0Drift (1-NFE)<br>pi0_base (10-NFE) | {drift_rate_str}<br>{baseline_rate_str} | {drift_latency_str}<br>{baseline_latency_str} | {speedup_str} |"
        )

    # Overall summary row
    if len(suites) > 1:
        drift_rate_str = "N/A"
        drift_latency_str = "N/A"
        if total_drift_episodes > 0:
            drift_rate = (total_drift_successes / total_drift_episodes) * 100.0
            drift_rate_str = f"{total_drift_successes}/{total_drift_episodes} ({drift_rate:.1f}%)"
        if total_drift_tasks > 0:
            avg_drift_lat = total_drift_latency_sum / total_drift_tasks
            drift_latency_str = f"{avg_drift_lat:.2f} ms"

        baseline_rate_str = "N/A"
        baseline_latency_str = "N/A"
        if total_baseline_episodes > 0:
            baseline_rate = (total_baseline_successes / total_baseline_episodes) * 100.0
            baseline_rate_str = f"{total_baseline_successes}/{total_baseline_episodes} ({baseline_rate:.1f}%)"
        if total_baseline_tasks > 0:
            avg_base_lat = total_baseline_latency_sum / total_baseline_tasks
            baseline_latency_str = f"{avg_base_lat:.2f} ms"

        speedup_str = "N/A"
        if total_drift_tasks > 0 and total_baseline_tasks > 0:
            avg_drift_lat = total_drift_latency_sum / total_drift_tasks
            avg_base_lat = total_baseline_latency_sum / total_baseline_tasks
            if avg_drift_lat > 0:
                speedup = avg_base_lat / avg_drift_lat
                speedup_str = f"**{speedup:.2f}x**"

        lines.append(
            f"| **OVERALL** | Pi0Drift (1-NFE)<br>pi0_base (10-NFE) | {drift_rate_str}<br>{baseline_rate_str} | {drift_latency_str}<br>{baseline_latency_str} | {speedup_str} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-GPU parallel evaluations on multiple JAX Pi0Drift checkpoints and baseline."
    )
    parser.add_argument(
        "--checkpoint-base-dir",
        type=str,
        default="checkpoints/pi0_drift_libero/my_drifting_vla_run",
        help="Path to the directory containing checkpoint step subdirectories"
    )
    parser.add_argument(
        "--checkpoint-config",
        type=str,
        default="pi0_drift_libero",
        help="Model config name for the trained checkpoints"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[],
        help="Specific checkpoint steps to evaluate. If empty, auto-detects all digit subdirectories."
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Optional path to baseline checkpoint directory. If not specified, uses default env config."
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default="pi05_libero",
        help="Model config name for the baseline checkpoint"
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=50,
        help="Number of trials/rollouts per task in each suite"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated list of GPU IDs to use for parallel evaluation"
    )
    parser.add_argument(
        "--port-start",
        type=int,
        default=8000,
        help="Starting port number for parallel workers"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="IP address/host to connect to policy servers"
    )
    parser.add_argument(
        "--suites",
        type=str,
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="List of Libero task suites to evaluate"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip evaluation of the baseline model"
    )
    parser.add_argument(
        "--keep-best-only",
        action="store_true",
        default=True,
        help="Clean up non-best checkpoints by relocating them to .trash/ checkpoints directory"
    )
    parser.add_argument(
        "--trash-dir",
        type=str,
        default=None,
        help="Custom destination path for discarded checkpoints. If not provided, defaults to ROOT_DIR/.trash/checkpoints"
    )
    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        logging.error("No valid GPU IDs specified.")
        sys.exit(1)

    # 1. Discover checkpoint steps
    checkpoint_base = os.path.join(ROOT_DIR, args.checkpoint_base_dir)
    if not args.steps:
        if not os.path.exists(checkpoint_base):
            logging.error(f"Checkpoint base directory not found at: {checkpoint_base}")
            sys.exit(1)
        subdirs = os.listdir(checkpoint_base)
        steps = []
        for d in subdirs:
            if d.isdigit() and os.path.isdir(os.path.join(checkpoint_base, d)):
                steps.append(int(d))
        steps.sort()
        if not steps:
            logging.error(f"No digit step directories found in {checkpoint_base}")
            sys.exit(1)
        logging.info(f"Auto-discovered checkpoint steps: {steps}")
    else:
        steps = sorted(args.steps)

    # 2. Build Job Queue
    job_queue = queue.Queue()
    results = {
        "drift": {},     # step -> suite -> summary
        "baseline": {}   # suite -> summary
    }

    # Add Drift jobs
    for step in steps:
        step_dir = os.path.join(checkpoint_base, str(step))
        for suite in args.suites:
            job_queue.put({
                "type": "drift",
                "step": step,
                "checkpoint_dir": step_dir,
                "suite": suite
            })

    # Add Baseline jobs if requested
    if not args.skip_baseline:
        for suite in args.suites:
            job_queue.put({
                "type": "baseline",
                "suite": suite
            })

    total_jobs = job_queue.qsize()
    logging.info(f"Initialized job queue with {total_jobs} tasks to run in parallel on GPUs: {gpu_list}")

    # 3. Launch worker threads
    threads = []
    for idx, gpu_id in enumerate(gpu_list):
        port = args.port_start + idx
        t = threading.Thread(
            target=worker,
            args=(gpu_id, port, job_queue, results, args),
            name=f"WorkerThread-GPU{gpu_id}"
        )
        t.daemon = True
        threads.append(t)
        t.start()

    # Wait for queue completion
    try:
        job_queue.join()
    except KeyboardInterrupt:
        logging.info("Interrupted during queue execution. Stopping remaining workers...")
        sys.exit(1)

    logging.info("All parallel evaluation workers finished execution. Gathering metrics...")

    # 4. Process metrics and generate reports
    drift_rankings = []
    for step, suite_data in results["drift"].items():
        total_successes = 0
        total_episodes = 0
        latencies = []
        
        for suite, summary in suite_data.items():
            total_successes += summary.get("total_successes", 0)
            total_episodes += summary.get("total_episodes", 0)
            latencies.append(summary.get("avg_inference_time_ms", 0.0))
            
        if total_episodes > 0:
            success_rate = (total_successes / total_episodes) * 100.0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            drift_rankings.append({
                "step": step,
                "successes": total_successes,
                "episodes": total_episodes,
                "success_rate": success_rate,
                "avg_latency": avg_latency
            })

    # Sort rankings (highest success rate first, then lowest latency)
    drift_rankings.sort(key=lambda x: (-x["success_rate"], x["avg_latency"]))

    if not drift_rankings:
        logging.warning("No checkpoints were successfully evaluated.")
        return

    # Master Ranking Report
    ranking_report = generate_drift_ranking_report(drift_rankings)
    
    # Checkpoint Comparison Reports
    comparison_reports = []
    for r in drift_rankings:
        step = r["step"]
        step_report = generate_suite_comparison_report(
            step=step,
            suites=args.suites,
            drift_suite_data=results["drift"][step],
            baseline_summaries=results["baseline"]
        )
        comparison_reports.append(step_report)

    full_report = ranking_report + "\n\n" + "\n\n---\n\n".join(comparison_reports)

    # Save final report to disk
    report_path = os.path.join(ROOT_DIR, "data", "libero", "checkpoint_comparison_report.md")
    with open(report_path, "w") as f:
        f.write(full_report)

    logging.info("\n" + "=" * 60)
    logging.info("PARALLEL EVALUATION COMPLETE - RANKING")
    logging.info("=" * 60)
    print(ranking_report)
    logging.info("=" * 60)
    logging.info(f"Full comparisons and ranking report saved to: {report_path}")

    # 5. Clean up non-best checkpoints (archive to .trash/)
    if args.keep_best_only and len(drift_rankings) > 1:
        best_step = drift_rankings[0]["step"]
        trash_base = args.trash_dir if args.trash_dir else os.path.join(ROOT_DIR, ".trash", "checkpoints")
        os.makedirs(trash_base, exist_ok=True)
        
        logging.info("\n" + "=" * 60)
        logging.info(f"CLEANUP PROCESS: Best step is Step {best_step} ({drift_rankings[0]['success_rate']:.2f}%).")
        logging.info(f"Archiving remaining checkpoints to: {trash_base}")
        logging.info("=" * 60)

        for r in drift_rankings[1:]:
            step_str = str(r["step"])
            step_path = os.path.join(checkpoint_base, step_str)
            if os.path.exists(step_path):
                dest_path = os.path.join(trash_base, step_str)
                if os.path.exists(dest_path):
                    # Collision avoidance: append timestamp
                    dest_path = os.path.join(trash_base, f"{step_str}_{int(time.time())}")
                
                try:
                    logging.info(f"[FILE_MOVE] Relocating checkpoint step {step_str} -> {dest_path}")
                    shutil.move(step_path, dest_path)
                except Exception as e:
                    logging.error(f"Failed to relocate checkpoint step {step_str}: {e}")
        logging.info("Cleanup successfully completed.")


if __name__ == "__main__":
    main()
