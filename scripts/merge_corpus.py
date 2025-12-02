#!/usr/bin/env python3
"""
Concatenate monthly corpus files into a single yearly corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge corpus_threads_<YEAR>-MM.jsonl into corpus_threads_<YEAR>.jsonl"
    )
    parser.add_argument("year", type=int, help="Year to merge (e.g., 2008)")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("years"),
        help="Base directory that contains years/<YEAR>/comments/corpus/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dir = args.base / str(args.year) / "comments" / "corpus"
    output = corpus_dir / f"corpus_threads_{args.year}.jsonl"
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    month_files = sorted(corpus_dir.glob(f"corpus_threads_{args.year}-??.jsonl"))
    if not month_files:
        raise FileNotFoundError(
            f"No monthly corpus files found in {corpus_dir} "
            f"(expected corpus_threads_{args.year}-MM.jsonl)"
        )

    corpus_dir.mkdir(parents=True, exist_ok=True)
    count_in = 0
    count_out = 0
    with output.open("w", encoding="utf-8") as w:
        for mf in month_files:
            with mf.open("r", encoding="utf-8") as r:
                for line in r:
                    if not line.strip():
                        continue
                    w.write(line)
                    count_out += 1
            count_in += 1

    print(
        f"Merged {count_in} monthly files into {output} | docs written: {count_out:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()

