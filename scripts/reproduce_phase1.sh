#!/bin/sh
set -eu

MODE=${1:-smoke}
case "$MODE" in
  smoke|full) ;;
  *) echo "usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

RESULTS=artifacts/results/raw.jsonl
PYTHONPATH=src python3 -m pytest
PYTHONPATH=src python3 -m synergy_bench.cli run \
  --config "configs/phase1/$MODE.yaml" \
  --artifacts artifacts \
  --results "$RESULTS"
PYTHONPATH=src python3 -m synergy_bench.cli report \
  --results "$RESULTS" \
  --table artifacts/tables/summary.csv \
  --figure artifacts/figures/accuracy.svg
