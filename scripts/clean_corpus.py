#!/usr/bin/env python3
"""
Clean a corpus JSONL (one document per line) for LDA using lightweight,
dependency-friendly preprocessing (regex + scikit-learn stopwords).

For each record, the `text` field is replaced with a cleaned token string:
- lowercase
- alphabetic tokens only
- drop stop words, the literal token "url"
- drop very short tokens (<=2 characters)
- optional stemming if nltk is available (no external downloads required)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from nltk.stem import PorterStemmer
except Exception:
    PorterStemmer = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and lemmatize a corpus JSONL for LDA (NLTK-based)."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to corpus JSONL (e.g., corpus_threads_2008.jsonl).",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Destination JSONL with cleaned `text` field.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit on number of documents (for quick tests).",
    )
    return parser.parse_args()


STOP_WORDS = set(ENGLISH_STOP_WORDS) | {"url"}
STEMMER = PorterStemmer() if PorterStemmer is not None else None
TOKEN_RE = re.compile(r"[A-Za-z]+")


def clean_text(text: str) -> str:
    tokens = TOKEN_RE.findall(text.lower())
    kept = []
    for tok in tokens:
        if tok in STOP_WORDS:
            continue
        if len(tok) <= 2:
            continue
        if STEMMER:
            tok = STEMMER.stem(tok)
        kept.append(tok)
    return " ".join(kept)


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {args.input_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    bad_lines = 0
    bad_examples = 0

    with args.output_path.open("w", encoding="utf-8") as writer:
        with args.input_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    bad_lines += 1
                    if bad_examples < 5:
                        print(f"[warn] Skipping invalid JSON line {idx}: {e}", file=sys.stderr)
                        bad_examples += 1
                    continue

                text = rec.get("text", "") or ""
                cleaned = clean_text(text)
                total_in += 1
                if args.max_docs and total_in > args.max_docs:
                    break
                if not cleaned:
                    continue
                rec["text"] = cleaned
                writer.write(json.dumps(rec))
                writer.write("\n")
                total_out += 1

    print(
        f"Cleaned corpus written to {args.output_path} | "
        f"docs read: {total_in:,} | docs kept: {total_out:,} | bad lines skipped: {bad_lines:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
