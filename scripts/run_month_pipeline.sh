#!/bin/bash
set -euo pipefail

# Usage: ./scripts/run_month_pipeline.sh YEAR MONTH

YEAR=$1
MONTH=$2

SHARED=/staging/groups/lis501_fall2025
BASE_DIR=$PWD/years/${YEAR}
RAW_DIR=$BASE_DIR/comments/raw
THREADS_DIR=$BASE_DIR/comments/threads
CORPUS_DIR=$BASE_DIR/comments/corpus

mkdir -p "$RAW_DIR" "$THREADS_DIR" "$CORPUS_DIR"

FILE="comments_${YEAR}-${MONTH}"
SHARED_BZ="$SHARED/${FILE}.bz2"
RAW_BZ="$RAW_DIR/${FILE}.bz2"
RAW_JSON="$RAW_DIR/${FILE}.jsonl"

if [ ! -f "$RAW_JSON" ]; then
  echo "Copying $SHARED_BZ"
  cp "$SHARED_BZ" "$RAW_BZ"
  bunzip2 -k "$RAW_BZ"
  mv "$RAW_DIR/${FILE}" "$RAW_JSON"
fi

THREADS_OUT="$THREADS_DIR/threads_${YEAR}-${MONTH}.jsonl"
CORPUS_OUT="$CORPUS_DIR/corpus_threads_${YEAR}-${MONTH}.jsonl"

python3 scripts/reconstruct_threads.py "$RAW_JSON" "$THREADS_OUT" --min-comments 5
python3 scripts/build_thread_corpus.py "$THREADS_OUT" "$CORPUS_OUT" --min-comments 5
