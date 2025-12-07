#!/usr/bin/env python3
"""
Concatenate multiple yearly merged corpora into a single JSONL file.

Input files are expected to be named corpus_threads_<YEAR>.jsonl under a
base directory (default: ./years/<YEAR>/). You can override the base path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple yearly corpora (corpus_threads_<YEAR>.jsonl) into one JSONL."
    )
    parser.add_argument(
        "years",
        nargs="+",
        help="Years to include (e.g., 2008 2009 2010 ...).",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("years"),
        help="Base directory that contains per-year corpus files (default: ./years).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the combined output JSONL.",
    )
    return parser.parse_args()


def iter_lines(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield line


def main() -> None:
    args = parse_args()

    input_paths = []
    for year in args.years:
        p = args.base / str(year) / f"corpus_threads_{year}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing corpus for {year}: {p}")
        input_paths.append(p)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    with args.output.open("w", encoding="utf-8") as out:
        for line in iter_lines(input_paths):
            out.write(line)
            total_in += 1

    print(
        f"Wrote combined corpus to {args.output} | lines/documents: {total_in:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
