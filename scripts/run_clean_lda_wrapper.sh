#!/bin/bash
set -euo pipefail

# Wrapper for Condor: unpack deps, merge monthly corpus (if needed), clean, then run LDA.

YEAR="$1"

# Unpack Python deps (scikit-learn, spaCy, model) into python_libs/
tar xzf python_libs.tar.gz
export PYTHONPATH="$PWD/python_libs:${PYTHONPATH:-}"

# Paths
CORPUS_DIR="years/${YEAR}/comments/corpus"
THREADS_DIR="years/${YEAR}/comments/threads"
OUTPUT_DIR="years/${YEAR}/lda/yearly"

mkdir -p "${CORPUS_DIR}" "${OUTPUT_DIR}"

COMBINED="${CORPUS_DIR}/corpus_threads_${YEAR}.jsonl"
CLEANED="${CORPUS_DIR}/corpus_threads_${YEAR}_clean.jsonl"

# If Condor dropped the corpus as a flat file in CWD, move it into place
FLAT_COMBINED="corpus_threads_${YEAR}.jsonl"
if [ ! -f "${COMBINED}" ] && [ -f "${FLAT_COMBINED}" ]; then
  mkdir -p "${CORPUS_DIR}"
  mv "${FLAT_COMBINED}" "${COMBINED}"
fi

# Bail out early if the combined corpus is still missing
if [ ! -f "${COMBINED}" ]; then
  echo "Combined corpus not found: ${COMBINED}" >&2
  ls -l . >&2
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
