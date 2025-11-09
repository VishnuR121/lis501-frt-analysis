#!/usr/bin/env python3
"""
Utilities for reconstructing Reddit discussion threads from the
politosphere monthly comment dumps.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Reddit thread trees from monthly JSONL comment dumps."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the raw comments_<YYYY-MM>.jsonl file.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Destination JSONL file where reconstructed threads will be stored.",
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        default=None,
        help="Only keep comments that match this subreddit name.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Limit the number of submissions (link_id) written to the output. "
        "Useful for debugging while working with large dumps.",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=1,
        help="Drop submissions that contain fewer than this many comments.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=250_000,
        help="Print progress every N processed comments.",
    )
    return parser.parse_args()


def normalize_comment_id(comment_id: str) -> str:
    """Ensure comment ids always carry the t1_ prefix."""
    return comment_id if comment_id.startswith("t1_") else f"t1_{comment_id}"


@dataclass
class CommentNode:
    id: str
    parent_id: str
    link_id: str
    subreddit: str
    created_utc: int
    score: int
    controversiality: int
    author: Optional[str]
    body: Optional[str]
    body_cleaned: Optional[str]
    distinguished: Optional[str]
    edited: Optional[bool]
    children: List[str] = field(default_factory=list)


def read_comments(
    input_path: Path,
    report_every: int,
    subreddit_filter: Optional[str] = None,
) -> Dict[str, CommentNode]:
    lookup: Dict[str, CommentNode] = {}

    with input_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            if subreddit_filter and record["subreddit"].lower() != subreddit_filter.lower():
                continue

            comment_id = normalize_comment_id(record["id"])

            lookup[comment_id] = CommentNode(
                id=comment_id,
                parent_id=record["parent_id"],
                link_id=record["link_id"],
                subreddit=record["subreddit"],
                created_utc=int(record["created_utc"]),
                score=int(record.get("score", 0)),
                controversiality=int(record.get("controversiality", 0)),
                author=record.get("author"),
                body=record.get("body"),
                body_cleaned=record.get("body_cleaned"),
                distinguished=record.get("distinguished"),
                edited=record.get("edited"),
            )

            if idx % report_every == 0:
                print(f"Read {idx:,} comments...", flush=True)

    print(f"Loaded {len(lookup):,} comments into memory.", flush=True)
    return lookup


def attach_children(lookup: Dict[str, CommentNode]) -> defaultdict[str, List[str]]:
    children_map: defaultdict[str, List[str]] = defaultdict(list)
    for comment in lookup.values():
        children_map[comment.parent_id].append(comment.id)

    for comment in lookup.values():
        comment.children = sorted(
            children_map.get(comment.id, []),
            key=lambda cid: (lookup[cid].created_utc, lookup[cid].id),
        )

    return children_map


def group_by_submission(comments: Dict[str, CommentNode]) -> defaultdict[str, List[str]]:
    by_link: defaultdict[str, List[str]] = defaultdict(list)
    for comment_id, comment in comments.items():
        by_link[comment.link_id].append(comment_id)
    return by_link


def build_tree(
    comment_id: str,
    lookup: Dict[str, CommentNode],
    depth: int = 0,
) -> Dict:
    comment = lookup[comment_id]
    return {
        "id": comment.id,
        "parent_id": comment.parent_id,
        "author": comment.author,
        "body": comment.body,
        "body_cleaned": comment.body_cleaned,
        "score": comment.score,
        "controversiality": comment.controversiality,
        "created_utc": comment.created_utc,
        "distinguished": comment.distinguished,
        "edited": comment.edited,
        "depth": depth,
        "children": [
            build_tree(child_id, lookup, depth + 1) for child_id in comment.children
        ],
    }


def iter_submissions_in_chron_order(
    submissions: defaultdict[str, List[str]],
    lookup: Dict[str, CommentNode],
) -> Iterable[str]:
    def submission_sort_key(link_id: str):
        ids = submissions[link_id]
        timestamps = [lookup[cid].created_utc for cid in ids]
        return (min(timestamps), link_id)

    return sorted(submissions.keys(), key=submission_sort_key)


def reconstruct_threads(
    lookup: Dict[str, CommentNode],
    max_threads: Optional[int],
    min_comments: int,
) -> List[Dict]:
    submissions = group_by_submission(lookup)
    ordered_links = iter_submissions_in_chron_order(submissions, lookup)

    results: List[Dict] = []
    for link_id in ordered_links:
        comment_ids = submissions[link_id]
        if len(comment_ids) < min_comments:
            continue

        roots: List[str] = []
        orphaned = 0
        for comment_id in comment_ids:
            parent_id = lookup[comment_id].parent_id
            if parent_id == link_id or parent_id not in lookup:
                if parent_id not in lookup and not parent_id.startswith("t3_"):
                    orphaned += 1
                roots.append(comment_id)

        roots = sorted(
            roots,
            key=lambda cid: (lookup[cid].created_utc, lookup[cid].id),
        )

        if not roots:
            continue

        created_times = [lookup[cid].created_utc for cid in comment_ids]
        thread_payload = {
            "link_id": link_id,
            "subreddit": lookup[roots[0]].subreddit,
            "comment_count": len(comment_ids),
            "root_count": len(roots),
            "created_utc_min": min(created_times),
            "created_utc_max": max(created_times),
            "orphan_comments": orphaned,
            "roots": [build_tree(root_id, lookup) for root_id in roots],
        }
        results.append(thread_payload)

        if max_threads and len(results) >= max_threads:
            break

    return results


def write_threads(threads: Iterable[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for thread in threads:
            handle.write(json.dumps(thread))
            handle.write("\n")
    print(f"Wrote {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    comments = read_comments(
        args.input_path,
        report_every=args.report_every,
        subreddit_filter=args.subreddit,
    )
    attach_children(comments)
    threads = reconstruct_threads(
        comments,
        max_threads=args.max_threads,
        min_comments=args.min_comments,
    )
    write_threads(threads, args.output_path)


if __name__ == "__main__":
    main()

