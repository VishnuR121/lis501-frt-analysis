#!/usr/bin/env python3
"""
Clean and lemmatize a corpus JSONL (one document per line) for LDA.

For each record, the `text` field is replaced with a cleaned, lemmatized string:
- lowercase
- alphabetic tokens only
- drop stop words, punctuation, and the literal token "url"
- drop very short tokens (<=2 characters)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import spacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and lemmatize a corpus JSONL for LDA."
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
        "--batch-size",
        type=int,
        default=500,
        help="spaCy pipe batch size (larger is faster but uses more memory).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit on number of documents (for quick tests).",
    )
    return parser.parse_args()


def load_nlp():
    # Expect the small English model to be installed.
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise SystemExit(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Install with: python -m spacy download en_core_web_sm"
        )


def clean_doc(doc: spacy.tokens.Doc) -> str:
    tokens: List[str] = []
    for tok in doc:
        if not tok.is_alpha:
            continue
        lemma = tok.lemma_.lower()
        if lemma == "url":
            continue
        if tok.is_stop or tok.is_punct or len(lemma) <= 2:
            continue
        tokens.append(lemma)
    return " ".join(tokens)


def iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {args.input_path}")

    nlp = load_nlp()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0
    batch_records: List[Tuple[dict, str]] = []

    with args.output_path.open("w", encoding="utf-8") as writer:
        for rec in iter_records(args.input_path):
            text = rec.get("text", "") or ""
            batch_records.append((rec, text))
            total_in += 1

            if args.max_docs and total_in > args.max_docs:
                break

            if len(batch_records) >= args.batch_size:
                docs = nlp.pipe((t for _, t in batch_records), batch_size=args.batch_size)
                for (orig, _), doc in zip(batch_records, docs):
                    cleaned = clean_doc(doc)
                    if not cleaned:
                        continue
                    orig["text"] = cleaned
                    writer.write(json.dumps(orig))
                    writer.write("\n")
                    total_out += 1
                batch_records.clear()

        # process remaining
        if batch_records:
            docs = nlp.pipe((t for _, t in batch_records), batch_size=args.batch_size)
            for (orig, _), doc in zip(batch_records, docs):
                cleaned = clean_doc(doc)
                if not cleaned:
                    continue
                orig["text"] = cleaned
                writer.write(json.dumps(orig))
                writer.write("\n")
                total_out += 1

    print(
        f"Cleaned corpus written to {args.output_path} | "
        f"docs read: {total_in:,} | docs kept: {total_out:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

