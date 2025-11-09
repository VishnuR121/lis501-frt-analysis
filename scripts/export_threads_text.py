#!/usr/bin/env python3
"""
Bulk-export every reconstructed Reddit thread into its own text file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.thread_render import render_thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render all threads from a threads_<YYYY-MM>.jsonl file into text files."
    )
    parser.add_argument(
        "threads_path",
        type=Path,
        help="Path to the JSONL produced by reconstruct_threads.py.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that will receive one text file per thread.",
    )
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=140,
        help="Truncate comment bodies to this many characters in the rendered output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of threads to export (useful for smoke tests).",
    )
    return parser.parse_args()


def sanitize_filename(link_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in link_id)
    if not safe:
        safe = "thread"
    return safe


def export_threads(args: argparse.Namespace) -> int:
    if not args.threads_path.exists():
        raise FileNotFoundError(f"Threads file not found: {args.threads_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    with args.threads_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            thread = json.loads(line)
            content = render_thread(thread, max_body_chars=args.max_body_chars)
            filename = f"{line_no:06d}_{sanitize_filename(thread['link_id'])}.txt"
            out_path = args.output_dir / filename
            out_path.write_text(content + "\n", encoding="utf-8")
            exported += 1
            if args.limit and exported >= args.limit:
                break

    return exported


def main() -> None:
    args = parse_args()
    count = export_threads(args)
    print(
        f"Wrote {count} thread files to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
