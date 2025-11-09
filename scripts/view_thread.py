#!/usr/bin/env python3
"""
Helper CLI to visualize reconstructed Reddit threads in a readable tree format.

Usage examples:

    python3 scripts/view_thread.py data/interim/threads_2008-01.jsonl --index 0
    python3 scripts/view_thread.py data/interim/threads_2008-01.jsonl --link-id t3_648iy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.thread_render import render_thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretty-print a reconstructed Reddit thread from a JSONL file."
    )
    parser.add_argument(
        "threads_path",
        type=Path,
        help="Path to the threads_<YYYY-MM>.jsonl output from reconstruct_threads.py.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--link-id",
        type=str,
        help="Specific submission link_id (e.g., t3_648iy) to display.",
    )
    group.add_argument(
        "--index",
        type=int,
        help="Zero-based index of the thread within the JSONL file.",
    )
    parser.add_argument(
        "--max-body-chars",
        type=int,
        default=140,
        help="Truncate comment bodies to this many characters for display.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional text file to send the rendered tree to. Prints to stdout if omitted.",
    )
    return parser.parse_args()


def load_thread(
    path: Path,
    *,
    link_id: Optional[str] = None,
    index: Optional[int] = None,
) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Threads file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        if link_id:
            for line_no, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["link_id"] == link_id:
                    return record
            raise ValueError(f"link_id {link_id} not found in {path}")

        assert index is not None and index >= 0
        for current_index, line in enumerate(handle):
            if not line.strip():
                continue
            if current_index == index:
                return json.loads(line)

    raise IndexError(f"Index {index} is out of range for {path}")


def main() -> None:
    args = parse_args()
    thread = load_thread(
        args.threads_path,
        link_id=args.link_id,
        index=args.index,
    )
    content = render_thread(thread, max_body_chars=args.max_body_chars)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(content)


if __name__ == "__main__":
    main()
