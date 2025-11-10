#!/usr/bin/env python3
"""
Convert reconstructed Reddit threads into LDA-ready documents.
Each document aggregates all comment text from a single submission/thread.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create thread-level documents suitable for LDA/topic modeling."
    )
    parser.add_argument(
        "threads_path",
        type=Path,
        help="Path to threads_<YYYY-MM>.jsonl produced by reconstruct_threads.py",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Destination JSONL file containing one document per thread.",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=5,
        help="Skip threads with fewer comments than this threshold (default: 5).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of documents written (useful for debugging).",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=5_000,
        help="Print progress after processing this many threads.",
    )
    return parser.parse_args()


def iter_comments(node: Dict) -> Iterable[Dict]:
    """Depth-first traversal yielding every comment node in the thread tree."""
    yield node
    for child in node.get("children", []):
        yield from iter_comments(child)


def extract_comment_text(comment: Dict) -> Optional[str]:
    text = comment.get("body_cleaned") or comment.get("body") or ""
    text = text.strip()
    return text or None


def aggregate_thread_text(thread: Dict) -> str:
    parts: List[str] = []
    for root in thread["roots"]:
        for comment in iter_comments(root):
            text = extract_comment_text(comment)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def main() -> None:
    args = parse_args()
    if not args.threads_path.exists():
        raise FileNotFoundError(f"Threads file not found: {args.threads_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_threads = 0
    written_docs = 0
    with args.threads_path.open("r", encoding="utf-8") as reader, args.output_path.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            thread = json.loads(line)
            total_threads += 1

            if thread["comment_count"] < args.min_comments:
                continue

            text = aggregate_thread_text(thread)
            if not text:
                continue

            record = {
                "link_id": thread["link_id"],
                "subreddit": thread["subreddit"],
                "comment_count": thread["comment_count"],
                "root_count": thread["root_count"],
                "created_utc_min": thread["created_utc_min"],
                "created_utc_max": thread["created_utc_max"],
                "text": text,
            }
            writer.write(json.dumps(record))
            writer.write("\n")
            written_docs += 1

            if args.max_docs and written_docs >= args.max_docs:
                break

            if total_threads % args.report_every == 0:
                print(
                    f"Processed {total_threads:,} threads | documents written: {written_docs:,}",
                    flush=True,
                )

    print(
        f"Finished. Considered {total_threads:,} threads, wrote {written_docs:,} documents.",
        flush=True,
    )


if __name__ == "__main__":
    main()

