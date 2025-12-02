#!/usr/bin/env python3
"""
Clean and lemmatize a corpus JSONL (one document per line) for LDA using NLTK.

For each record, the `text` field is replaced with a cleaned, lemmatized string:
- lowercase
- alphabetic tokens only
- drop stop words, punctuation, the literal token "url"
- drop very short tokens (<=2 characters)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure required NLTK data is available
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}") if "punkt" in resource else nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


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


STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
ALPHA_RE = re.compile(r"[^a-zA-Z]+")


def clean_text(text: str) -> str:
    # Lowercase and remove non-alpha characters
    text = ALPHA_RE.sub(" ", text.lower())
    tokens = word_tokenize(text)
    kept = []
    for tok in tokens:
        if tok == "url":
            continue
        if tok in STOP_WORDS:
            continue
        if len(tok) <= 2:
            continue
        lemma = LEMMATIZER.lemmatize(tok)
        if not lemma.isalpha():
            continue
        kept.append(lemma)
    return " ".join(kept)


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {args.input_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    with args.output_path.open("w", encoding="utf-8") as writer:
        for rec in iter_records(args.input_path):
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
        f"docs read: {total_in:,} | docs kept: {total_out:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
