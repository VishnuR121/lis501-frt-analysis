#!/bin/bash
set -euo pipefail

# Wrapper for Condor: unpack deps, merge monthly corpus (if needed), clean, then run LDA.

YEAR="$1"

# Unpack Python deps (scikit-learn, spaCy, model) into python_libs/
tar xzf python_libs.tar.gz
export PYTHONPATH="$PWD/python_libs:${PYTHONPATH:-}"

# Paths (use OUTPUT_ROOT if set, else current dir)
BASE=${OUTPUT_ROOT:-$PWD}
YEAR_DIR="${BASE}/years/${YEAR}"
OUTPUT_DIR="${YEAR_DIR}/lda/yearly"

mkdir -p "${YEAR_DIR}" "${OUTPUT_DIR}"

COMBINED="${YEAR_DIR}/corpus_threads_${YEAR}.jsonl"
CLEANED="${YEAR_DIR}/corpus_threads_${YEAR}_clean.jsonl"

if [ ! -f "${COMBINED}" ]; then
  echo "Combined corpus not found: ${COMBINED}" >&2
  ls -l "${YEAR_DIR}" >&2 || true
  exit 1
fi

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
