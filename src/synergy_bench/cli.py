"""Phase 1.2 command-line reporting entry point."""

from __future__ import annotations

import argparse

from .evaluation.reporting import build_report
from .experiment import run_phase1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synergy benchmark Phase 1 utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="regenerate table and figure from raw results")
    report.add_argument("--results", required=True)
    report.add_argument("--table", required=True)
    report.add_argument("--figure", required=True)
    run = commands.add_parser("run", help="run a configured Phase 1 experiment matrix")
    run.add_argument("--config", required=True)
    run.add_argument("--artifacts", default="artifacts")
    run.add_argument("--results", required=True)
    arguments = parser.parse_args(argv)

    if arguments.command == "report":
        build_report(arguments.results, arguments.table, arguments.figure)
    elif arguments.command == "run":
        run_phase1(arguments.config, arguments.artifacts, arguments.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
