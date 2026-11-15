"""
evaluation/cli.py
=================
Command line interface for the CRIP offline evaluation harness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evaluation.suite import BenchmarkSuite
from evaluation.tuning import HyperparameterTuner


def main() -> None:
    parser = argparse.ArgumentParser(description="CRIP Offline Evaluation Harness")
    subparsers = parser.add_subparsers(dest="command", help="Evaluation commands")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run offline evaluation benchmark suite")
    p_bench.add_argument("--suite", choices=["quick", "full"], default="quick", help="Suite to execute (default: quick)")
    p_bench.add_argument("--output", type=str, default=None, help="Output directory path")

    # tune
    p_tune = subparsers.add_parser("tune", help="Run hyperparameter search on TUNE seeds")
    p_tune.add_argument("--quick", action="store_true", help="Quick tuning mode")

    # report
    p_rep = subparsers.add_parser("report", help="Print path or status of latest evaluation report")

    args = parser.parse_args()

    if args.command == "benchmark":
        out_dir = Path(args.output) if args.output else None
        suite = BenchmarkSuite()
        report_path = suite.run_benchmark(suite_name=args.suite, output_dir=out_dir)
        print(f"\n[OK] Evaluation completed successfully! View report at:\n     {report_path}")

    elif args.command == "tune":
        tuner = HyperparameterTuner()
        best_cfg = tuner.run_grid_search(quick=args.quick)
        print("\n[OK] Hyperparameter tuning complete on TUNE seeds:")
        for k, v in best_cfg.items():
            print(f"  {k}: {v}")

    elif args.command == "report":
        latest = Path(__file__).parent / "results" / "latest" / "REPORT.md"
        if latest.exists():
            print(f"Latest report: {latest.resolve()}")
            print("\n" + "=" * 60)
            print(latest.read_text(encoding="utf-8")[:1000] + "\n...")
        else:
            print("No evaluation reports found yet. Run: crip eval-offline")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
