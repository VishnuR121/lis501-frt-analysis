#!/usr/bin/env python3
"""
Filter, clean, and lemmatize thread-level documents prior to LDA.

The script expects a JSONL corpus produced by build_thread_corpus.py.
Only documents within the requested temporal window are kept.
Each text is lowercased, stripped of stopwords/URL tokens/non-letter chars,
and lemmatized before being written back out as JSON.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
from pathlib import Path
from typing import Iterable, Tuple

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

TOKEN_PATTERN = re.compile(r"[a-z]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and lemmatize a thread-level corpus JSONL file."
    )
    parser.add_argument("input_path", type=Path, help="Path to raw corpus JSONL.")
    parser.add_argument("output_path", type=Path, help="Destination for cleaned JSONL.")
    parser.add_argument(
        "--year",
        type=int,
        help="Restrict documents to this calendar year (UTC).",
    )
    parser.add_argument(
        "--start-ts",
        type=int,
        default=None,
        help="Earliest created_utc_min timestamp (inclusive).",
    )
    parser.add_argument(
        "--end-ts",
        type=int,
        default=None,
        help="Latest created_utc_min timestamp (exclusive).",
    )
    parser.add_argument(
        "--assume-sorted",
        action="store_true",
        help="Stop processing once timestamps exceed the end bound.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of cleaned documents written (debugging).",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=1_000,
        help="Print progress after this many input records.",
    )
    return parser.parse_args()


def resolve_bounds(args: argparse.Namespace) -> Tuple[int | None, int | None]:
    if args.year and (args.start_ts or args.end_ts):
        raise SystemExit("Specify either --year or explicit timestamps, not both.")
    if args.year:
        start = calendar.timegm((args.year, 1, 1, 0, 0, 0))
        end = calendar.timegm((args.year + 1, 1, 1, 0, 0, 0))
        return start, end
    if args.start_ts and args.end_ts and args.start_ts >= args.end_ts:
        raise SystemExit("--start-ts must be < --end-ts")
    return args.start_ts, args.end_ts


def wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def build_stopwords() -> set[str]:
    stops = set(stopwords.words("english"))
    stops.add("url")
    return stops


def clean_text(
    text: str,
    *,
    stops: set[str],
    lemmatizer: WordNetLemmatizer,
) -> str:
    tokens = TOKEN_PATTERN.findall(text.lower())
    tokens = [tok for tok in tokens if tok not in stops]
    if not tokens:
        return ""

    tagged = pos_tag(tokens)
    cleaned: list[str] = []
    for token, tag in tagged:
        lemma = lemmatizer.lemmatize(token, wordnet_pos(tag))
        if lemma and lemma not in stops:
            cleaned.append(lemma)
    return " ".join(cleaned)


def document_in_range(
    created_utc_min: int,
    start_ts: int | None,
    end_ts: int | None,
) -> bool:
    if start_ts and created_utc_min < start_ts:
        return False
    if end_ts and created_utc_min >= end_ts:
        return False
    return True


def preprocess_corpus(args: argparse.Namespace) -> None:
    start_ts, end_ts = resolve_bounds(args)
    stops = build_stopwords()
    lemmatizer = WordNetLemmatizer()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_seen = 0
    written = 0

    with args.input_path.open("r", encoding="utf-8") as reader, args.output_path.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            total_seen += 1
            record = json.loads(line)

            created = int(record.get("created_utc_min", 0))
            in_range = document_in_range(created, start_ts, end_ts)

            if not in_range:
                if (
                    args.assume_sorted
                    and end_ts is not None
                    and created >= end_ts
                ):
                    break
                continue

            text = record.get("text", "")
            cleaned = clean_text(text, stops=stops, lemmatizer=lemmatizer)
            if not cleaned:
                continue

            record["text"] = cleaned
            writer.write(json.dumps(record))
            writer.write("\n")
            written += 1

            if args.max_docs and written >= args.max_docs:
                break

            if total_seen % args.report_every == 0:
                print(
                    f"Processed {total_seen:,} records | cleaned docs: {written:,}",
                    flush=True,
                )

    print(
        f"Completed preprocessing. Considered {total_seen:,} records, "
        f"wrote {written:,} cleaned documents.",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {args.input_path}")
    preprocess_corpus(args)


if __name__ == "__main__":
    main()
