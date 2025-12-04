#!/bin/bash
set -euo pipefail

# Wrapper for Condor: unpack deps, merge monthly corpus (if needed), clean, then run LDA.

YEAR="$1"

# Unpack Python deps (scikit-learn, spaCy, model) into python_libs/
tar xzf python_libs.tar.gz
export PYTHONPATH="$PWD/python_libs:${PYTHONPATH:-}"

# Paths (default to working dir; override with OUTPUT_ROOT)
BASE=${OUTPUT_ROOT:-$PWD}
CORPUS_DIR="${BASE}/years/${YEAR}/comments/corpus"
THREADS_DIR="${BASE}/years/${YEAR}/comments/threads"
OUTPUT_DIR="${BASE}/years/${YEAR}/lda/yearly"

mkdir -p "${CORPUS_DIR}" "${OUTPUT_DIR}"

COMBINED="${CORPUS_DIR}/corpus_threads_${YEAR}.jsonl"
PATH_CHECK="${COMBINED}"
if [ ! -f "${PATH_CHECK}" ]; then
  echo "Combined corpus not found: ${PATH_CHECK}" >&2
  ls -l "${CORPUS_DIR}" >&2 || true
  exit 1
fi

CLEANED="${CORPUS_DIR}/corpus_threads_${YEAR}_clean.jsonl"

# Clean + lemmatize
python3 scripts/clean_corpus.py "${COMBINED}" "${CLEANED}"

# Run LDA on cleaned corpus
python3 scripts/run_lda.py \
  "${CLEANED}" \
  "${OUTPUT_DIR}" \
  --num-topics 50 \
  --min-df 50 \
  --max-features 20000 \
  --learning-method online \
  --batch-size 1024 \
  --max-iter 10
