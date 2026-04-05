#!/usr/bin/env python
"""
SNA Twitch Influencer Project - Python CLI Pipeline Runner
==========================================================
Cross-platform Python script to run the analysis pipeline.
Especially useful for Windows users who may not have Make/Bash.

Usage:
    python run_all.py              # Run full pipeline
    python run_all.py --stage 1    # Run specific stage
    python run_all.py --list       # List available stages
"""

import argparse
import subprocess
import sys
import time
import json
from typing import Tuple, Dict
from pathlib import Path
from datetime import datetime


# Stage definitions
STAGES = {
    0: {
        "name": "Data Audit and Preprocessing",
        "scripts": [
            "src/data/load_raw.py",
            "src/data/preprocess_graph.py"
        ]
    },
    1: {
        "name": "Centrality Computation",
        "scripts": [
            "src/graph/centrality.py"
        ]
    },
    2: {
        "name": "Community Detection and K-Shell",
        "scripts": [
            "src/graph/community.py",
            "src/graph/kshell.py"
        ]
    },
    3: {
        "name": "SIS, Typology, and Null Model",
        "scripts": [
            "src/sis/compute_sis.py",
            "src/sis/build_typology.py",
            "src/sis/robustness.py",
            "src/graph/null_model.py"
        ]
    },
    4: {
        "name": "IC Calibration and Single-Seed Validation",
        "scripts": [
            "src/simulation/ic_calibration.py",
            "src/simulation/run_single_seed_ic.py"
        ]
    },
    5: {
        "name": "Multi-Seed IC Benchmark",
        "scripts": [
            "src/simulation/seed_strategies.py",
            "src/simulation/run_multi_seed_ic.py"
        ]
    },
    6: {
        "name": "ML Detectability Analysis",
        "scripts": [
            "src/ml/features_surface.py",
            "src/ml/train_lr.py",
            "src/ml/train_rf.py",
            "src/ml/shap_analysis.py",
            "src/ml/evaluate_metrics.py"
        ]
    },
    7: {
        "name": "Summary and Reporting",
        "scripts": [
            "src/evaluation/summary_tables.py"
        ]
    }
}


def print_header(text: str, char: str = "="):
    """Print formatted header."""
    line = char * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def print_status(text: str, status: str = "INFO"):
    """Print status message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    symbols = {
        "INFO": "[i]",
        "OK": "[OK]",
        "ERROR": "[ERR]",
        "RUN": "[RUN]"
    }
    symbol = symbols.get(status, "•")
    print(f"[{timestamp}] {symbol} {text}")


def run_script(script_path: str) -> Tuple[bool, float]:
    """
    Run a Python script and return success status and duration.

    Returns
    -------
    Tuple[bool, float]
        (success, duration_seconds)
    """
    start_time = time.time()

    try:
        print_status(f"Running {script_path}", "RUN")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        duration = time.time() - start_time
        print_status(f"Completed in {duration:.1f}s", "OK")
        return True, duration

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print_status(f"FAILED: {script_path}", "ERROR")
        print(f"  Error output:\n{e.stderr[:500]}..." if len(e.stderr) > 500 else f"  Error output:\n{e.stderr}")
        return False, duration

    except FileNotFoundError:
        duration = time.time() - start_time
        print_status(f"Script not found: {script_path}", "ERROR")
        return False, duration


def run_stage(stage_num: int) -> Tuple[bool, float]:
    """
    Run all scripts in a stage.

    Returns
    -------
    Tuple[bool, float]
        (all_success, total_duration)
    """
    if stage_num not in STAGES:
        print_status(f"Invalid stage number: {stage_num}", "ERROR")
        return False, 0

    stage = STAGES[stage_num]
    print_header(f"Stage {stage_num}: {stage['name']}")

    total_duration = 0
    all_success = True

    for script in stage["scripts"]:
        if not Path(script).exists():
            print_status(f"Script not found: {script} (skipping)", "ERROR")
            continue

        success, duration = run_script(script)
        total_duration += duration

        if not success:
            all_success = False
            # Continue to next script or stop?
            # For now, continue to see all failures

    return all_success, total_duration


def run_full_pipeline() -> Dict:
    """
    Run the complete pipeline.

    Returns
    -------
    Dict
        Execution summary
    """
    print_header("SNA Twitch Influencer Project - Full Pipeline", "=")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print()

    # Create output directories
    for dir_path in [
        "outputs/stage0_data_quality",
        "outputs/stage1",
        "outputs/stage2",
        "outputs/stage3",
        "outputs/stage3_ic_calibration",
        "outputs/stage4_single_seed",
        "outputs/stage5_multi_seed",
        "outputs/stage6_ml",
        "reports/figures",
        "reports/tables",
        "logs/run_history",
        "logs/timing"
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        "run_id": run_id,
        "stages": {},
        "success": True
    }

    for stage_num in sorted(STAGES.keys()):
        success, duration = run_stage(stage_num)
        results["stages"][stage_num] = {
            "name": STAGES[stage_num]["name"],
            "success": success,
            "duration_seconds": duration
        }
        if not success:
            results["success"] = False
            print_status(f"Stage {stage_num} failed. Continuing...", "ERROR")

    total_duration = time.time() - start_time
    results["total_duration_seconds"] = total_duration

    # Summary
    print_header("Pipeline Complete")
    print(f"Total runtime: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    print(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}")
    print()
    print("Stage Summary:")
    for stage_num, stage_result in results["stages"].items():
        status = "✓" if stage_result["success"] else "✗"
        print(f"  {status} Stage {stage_num}: {stage_result['name']} ({stage_result['duration_seconds']:.1f}s)")

    # Save run log
    log_path = Path(f"logs/run_history/run_{run_id}.json")
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nLog saved to: {log_path}")

    return results


def list_stages():
    """Print list of available stages."""
    print_header("Available Stages")
    for stage_num, stage in STAGES.items():
        print(f"  Stage {stage_num}: {stage['name']}")
        for script in stage["scripts"]:
            exists = "✓" if Path(script).exists() else "✗"
            print(f"    {exists} {script}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="SNA Twitch Influencer Project Pipeline Runner"
    )
    parser.add_argument(
        "--stage", "-s",
        type=int,
        help="Run specific stage (0-7)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available stages"
    )
    parser.add_argument(
        "--from-stage",
        type=int,
        help="Run from this stage onwards"
    )

    args = parser.parse_args()

    if args.list:
        list_stages()
        return

    if args.stage is not None:
        success, duration = run_stage(args.stage)
        sys.exit(0 if success else 1)

    if args.from_stage is not None:
        print_header(f"Running stages {args.from_stage} onwards")
        for stage_num in sorted(STAGES.keys()):
            if stage_num >= args.from_stage:
                success, _ = run_stage(stage_num)
                if not success:
                    print_status(f"Pipeline stopped at stage {stage_num}", "ERROR")
                    sys.exit(1)
        sys.exit(0)

    # Default: run full pipeline
    results = run_full_pipeline()
    sys.exit(0 if results["success"] else 1)

if __name__ == "__main__":
    main()
