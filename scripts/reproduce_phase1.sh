#!/bin/sh
set -eu

RESULTS=${1:-artifacts/results/raw.jsonl}
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m synergy_bench.cli report \
  --results "$RESULTS" \
  --table artifacts/tables/summary.csv \
  --figure artifacts/figures/accuracy.svg
