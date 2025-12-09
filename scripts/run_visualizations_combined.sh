#!/bin/bash
set -euo pipefail

# One-off runner to generate visualizations for the combined 2008-2019 corpus.

ROOT=/staging/groups/lis501_fall2025/lis501-frt-analysis/years
CORPUS="${ROOT}/corpus_threads_2008_2019_clean.jsonl"
OUT_DIR="${ROOT}/visuals_2008_2019"

tar xzf python_libs.tar.gz
export PYTHONPATH="$PWD/python_libs:${PYTHONPATH:-}"

if [ ! -f "$CORPUS" ]; then
  echo "Cleaned corpus not found: $CORPUS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python3 scripts/make_visualizations.py \
  "$CORPUS" \
  "$OUT_DIR" \
  --num-topics 50 \
  --min-df 50 \
  --max-features 20000
