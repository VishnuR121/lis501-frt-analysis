from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Dict, List, Optional


def human_time(epoch_seconds: int) -> str:
    dt = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_body(body: Optional[str], max_chars: int) -> str:
    if not body:
        return ""
    body = body.replace("\n", " ").strip()
    if len(body) <= max_chars:
        return body
    return body[: max_chars - 3].rstrip() + "..."


def render_comment(comment: Dict, *, max_body_chars: int, depth: int = 0) -> List[str]:
    lines: List[str] = []
    indent = "  " * depth
    author = comment.get("author") or "[unknown]"
    timestamp = human_time(comment["created_utc"])
    net_votes = comment.get("net_votes", comment.get("score", 0))
    body = format_body(comment.get("body"), max_body_chars)
    meta = f"{indent}- {author} | net votes={net_votes} | {timestamp}"
    lines.append(meta)
    if body:
        width = max(20, 100 - len(indent))
        wrapped = textwrap.wrap(body, width=width)
        for line in wrapped:
            lines.append(f"{indent}  {line}")
    for child in comment.get("children", []):
        lines.extend(
            render_comment(child, max_body_chars=max_body_chars, depth=depth + 1)
        )
    return lines


def render_thread(thread: Dict, max_body_chars: int) -> str:
    header = (
        f"Thread {thread['link_id']} | subreddit={thread['subreddit']} | "
        f"comments={thread['comment_count']} | roots={thread['root_count']} | "
        f"{human_time(thread['created_utc_min'])} → {human_time(thread['created_utc_max'])} "
        f"| orphans={thread['orphan_comments']}"
    )
    lines = [header, "=" * len(header)]
    for idx, root in enumerate(thread["roots"], start=1):
        lines.append("")
        lines.append(f"Root #{idx}")
        lines.extend(render_comment(root, max_body_chars=max_body_chars))
    return "\n".join(lines)

