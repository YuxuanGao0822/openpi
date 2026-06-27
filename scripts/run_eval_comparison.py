#!/usr/bin/env python3
"""Orchestrator script to evaluate and compare the 1-NFE JAX Pi0Drift model
against the baseline 10-NFE pi0_base model across Libero task suites.
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Keep track of active server process for cleanup
active_server_proc = None


def kill_port_owner(port: int):
    """Find and terminate any process listening on the given port."""
    try:
        # Cross-platform friendly port checking/killing on Linux/macOS
        cmd = ["lsof", "-t", f"-i:{port}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            pids = result.stdout.strip().split()
            for pid in pids:
                if pid:
                    logging.info(f"Port {port} in use by PID {pid}. Killing it...")
                    subprocess.run(["kill", "-9", pid], check=False)
            time.sleep(1.0)  # Wait for socket to free up
    except Exception as e:
        logging.warning(f"Could not check or kill process on port {port}: {e}")


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 120.0) -> bool:
    """Wait for the port to start listening."""
    start_time = time.time()
    logging.info(f"Waiting up to {timeout}s for server to start on {host}:{port}...")
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                logging.info(f"Policy server is up and listening on port {port}!")
                return True
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(1.0)
    return False


def run_server(
    is_drift: bool,
    port: int,
    host: str,
    checkpoint_dir: Optional[str],
    checkpoint_config: str,
    baseline_dir: Optional[str],
    baseline_config: str,
) -> bool:
    """Launch the policy server as a background subprocess."""
    global active_server_proc
    kill_port_owner(port)

    # Construct server command
    cmd = [sys.executable, "scripts/serve_policy.py", "--port", str(port)]

    if is_drift:
        if not checkpoint_dir:
            logging.error("Checkpoint directory is required to serve the drift model.")
            return False
        cmd.extend([
            "--policy.config", checkpoint_config,
            "--policy.dir", checkpoint_dir
        ])
        log_name = "server_drift.log"
    else:
        if baseline_dir:
            cmd.extend([
                "--policy.config", baseline_config,
                "--policy.dir", baseline_dir
            ])
        else:
            cmd.extend([
                "--env", "libero",
                "--policy", "default"
            ])
        log_name = "server_baseline.log"

    os.makedirs(os.path.join(ROOT_DIR, "data", "libero"), exist_ok=True)
    log_path = os.path.join(ROOT_DIR, "data", "libero", log_name)
    
    # Open file for writing logs
    log_file = open(log_path, "w")

    logging.info(f"Starting policy server with command: {' '.join(cmd)}")
    logging.info(f"Server logs will be written to: {log_path}")

    active_server_proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        cwd=ROOT_DIR
    )

    if not wait_for_port(port, host=host):
        logging.error(f"Policy server failed to start on port {port} within timeout. Check logs at {log_path}")
        active_server_proc.terminate()
        try:
            active_server_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            active_server_proc.kill()
        active_server_proc = None
        log_file.close()
        return False

    log_file.close()
    return True


def stop_active_server(port: int):
    """Clean up the active server process and release the port."""
    global active_server_proc
    if active_server_proc:
        logging.info("Stopping active policy server...")
        active_server_proc.terminate()
        try:
            active_server_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logging.warning("Server did not terminate within timeout. Force killing...")
            active_server_proc.kill()
            active_server_proc.wait()
        active_server_proc = None
    kill_port_owner(port)


def run_client(suite: str, port: int, host: str, num_trials: int, summary_path: str) -> bool:
    """Run the timed evaluation client for a specific task suite."""
    cmd = [
        sys.executable,
        "examples/libero/eval_with_timing.py",
        "--host", host,
        "--port", str(port),
        "--task-suite-name", suite,
        "--num-trials-per-task", str(num_trials),
        "--summary-out-path", summary_path
    ]
    logging.info(f"Running evaluation client for suite {suite} with command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=ROOT_DIR, check=False)
    if result.returncode != 0:
        logging.error(f"Evaluation client failed for suite {suite} with return code {result.returncode}")
        return False
    return True


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


def generate_comparison_report(suites: List[str], drift_summaries: dict, baseline_summaries: dict) -> str:
    """Construct a Markdown comparison table of the results."""
    lines = []
    lines.append("# Libero Evaluation & Timing Comparison Report\n")
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
        drift = drift_summaries.get(suite)
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
        description="Run comparison evaluations on JAX Pi0Drift and pi0_base across Libero suites."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Path to trained JAX Pi0Drift checkpoint directory (e.g. checkpoints/pi0_drift_libero/my_drifting_vla_run/1000)"
    )
    parser.add_argument(
        "--checkpoint-config",
        type=str,
        default="pi0_drift_libero",
        help="Model config name for the trained checkpoint"
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
        "--port",
        type=int,
        default=8000,
        help="Port to serve policy on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="IP address/host to use for serving policy"
    )
    parser.add_argument(
        "--suites",
        type=str,
        nargs="+",
        default=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="List of Libero task suites to evaluate"
    )
    parser.add_argument(
        "--skip-drift",
        action="store_true",
        help="Skip evaluation of the drift model"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip evaluation of the baseline model"
    )
    args = parser.parse_args()

    drift_summaries = {}
    baseline_summaries = {}

    try:
        # 1. Evaluate Pi0Drift
        if not args.skip_drift:
            logging.info("========================================")
            logging.info("Starting evaluation for Pi0Drift (1-NFE)")
            logging.info("========================================")
            
            if run_server(
                is_drift=True,
                port=args.port,
                host=args.host,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_config=args.checkpoint_config,
                baseline_dir=args.baseline_dir,
                baseline_config=args.baseline_config,
            ):
                for suite in args.suites:
                    summary_path = os.path.join(ROOT_DIR, "data", "libero", f"eval_drift_{suite}.json")
                    logging.info(f"Evaluating suite: {suite} ...")
                    run_client(
                        suite=suite,
                        port=args.port,
                        host=args.host,
                        num_trials=args.num_trials,
                        summary_path=summary_path
                    )
                    
                    summary = load_summary(summary_path)
                    if summary:
                        drift_summaries[suite] = summary
                        
                stop_active_server(args.port)

        # 2. Evaluate Baseline (pi0_base)
        if not args.skip_baseline:
            logging.info("========================================")
            logging.info("Starting evaluation for pi0_base (10-NFE)")
            logging.info("========================================")
            
            if run_server(
                is_drift=False,
                port=args.port,
                host=args.host,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_config=args.checkpoint_config,
                baseline_dir=args.baseline_dir,
                baseline_config=args.baseline_config,
            ):
                for suite in args.suites:
                    summary_path = os.path.join(ROOT_DIR, "data", "libero", f"eval_baseline_{suite}.json")
                    logging.info(f"Evaluating suite: {suite} ...")
                    run_client(
                        suite=suite,
                        port=args.port,
                        host=args.host,
                        num_trials=args.num_trials,
                        summary_path=summary_path
                    )
                    
                    summary = load_summary(summary_path)
                    if summary:
                        baseline_summaries[suite] = summary
                        
                stop_active_server(args.port)

        # 3. Generate Report
        report_suites = [s for s in args.suites if s in drift_summaries or s in baseline_summaries]
        if not report_suites:
            logging.warning("No evaluations were completed successfully. Cannot generate report.")
            return

        report = generate_comparison_report(report_suites, drift_summaries, baseline_summaries)
        
        # Save report
        report_path = os.path.join(ROOT_DIR, "data", "libero", "comparison_report.md")
        with open(report_path, "w") as f:
            f.write(report)
            
        logging.info("\n" + "=" * 50)
        logging.info("EVALUATION COMPARISON RESULTS")
        logging.info("=" * 50)
        print(report)
        logging.info("=" * 50)
        logging.info(f"Detailed Markdown report saved to {report_path}")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Cleaning up and exiting...")
    finally:
        stop_active_server(args.port)


if __name__ == "__main__":
    main()
