#!/usr/bin/env python3
"""
Clean and lemmatize a corpus JSONL (one document per line) for LDA using spaCy.

For each record, the `text` field is replaced with a cleaned, lemmatized string:
- lowercase
- alphabetic tokens only
- drop stop words, punctuation, the literal token "url"
- drop very short tokens (<=2 characters)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import spacy

NLP = None


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


def load_nlp():
    return spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])


def clean_text(text: str) -> str:
    doc = NLP(text.lower())
    kept = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or not tok.is_alpha:
            continue
        if tok.text == "url":
            continue
        lemma = tok.lemma_.strip()
        if len(lemma) <= 2:
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

    global NLP
    NLP = load_nlp()

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
