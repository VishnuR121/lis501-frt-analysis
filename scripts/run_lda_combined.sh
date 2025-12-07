#!/bin/bash
set -euo pipefail

# One-off runner to clean and run LDA on the combined 2008-2019 corpus.

# Location of the combined corpus and outputs on shared space
ROOT=/staging/groups/lis501_fall2025/lis501-frt-analysis/years
COMBINED="${ROOT}/corpus_threads_2008_2019.jsonl"
CLEANED="${ROOT}/corpus_threads_2008_2019_clean.jsonl"
OUT_DIR="${ROOT}/lda_2008_2019"

# Unpack deps for the job sandbox and set PYTHONPATH
tar xzf python_libs.tar.gz
export PYTHONPATH="$PWD/python_libs:${PYTHONPATH:-}"

if [ ! -f "$COMBINED" ]; then
  echo "Combined corpus not found: $COMBINED" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python3 scripts/clean_corpus.py "$COMBINED" "$CLEANED"

python3 scripts/run_lda.py \
  "$CLEANED" \
  "$OUT_DIR" \
  --num-topics 50 \
  --min-df 50 \
  --max-features 20000 \
  --learning-method online \
  --batch-size 1024 \
  --max-iter 10
