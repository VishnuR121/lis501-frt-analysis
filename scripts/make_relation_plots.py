#!/usr/bin/env python3
"""
Compute topic cooccurrence vs. prevalence correlations and render relation plots
similar to Figure 1 in Chuang et al. (ACL 2017).

Inputs:
- doc_topics.jsonl produced by run_lda.py (one record per thread with topic_distribution and timestamps)
- topics.json from the same LDA run (for naming)

Outputs:
- A multi-panel PNG with one subplot per requested topic pair
- A JSON summary of prevalence, PMI, and correlation values
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


@dataclass
class TopicPair:
    name: str
    topic_a: int
    topic_b: int
    label: str | None = None  # optional friendly label to display instead of top words


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot topic cooccurrence vs prevalence relations (2008-2019)."
    )
    parser.add_argument(
        "--doc-topics",
        type=Path,
        default=Path("lda_2008_2019/doc_topics.jsonl"),
        help="Path to doc_topics.jsonl with topic_distribution and created_utc_min.",
    )
    parser.add_argument(
        "--topics-json",
        type=Path,
        default=Path("lda_2008_2019/topics.json"),
        help="Path to topics.json (to pull top words for names).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to write outputs.",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.05,
        help="Minimum topic weight to count a topic as present in a document.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2008,
        help="Earliest year to consider (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2019,
        help="Latest year to consider (inclusive).",
    )
    return parser.parse_args()


def load_topic_names(path: Path) -> Dict[int, str]:
    data = json.loads(path.read_text())
    names: Dict[int, str] = {}
    for t in data["topics"]:
        top_words = t["top_words"][:2]
        names[int(t["topic_id"])] = ", ".join(top_words)
    return names


def load_pairs() -> List[TopicPair]:
    # Topic IDs are derived from lda_2008_2019/topics.json
    return [
        TopicPair("Immigration vs Build Wall", topic_a=11, topic_b=7),
        TopicPair("Sex/Gender vs Abortion", topic_a=32, topic_b=41),
        TopicPair("Climate Change vs Economy", topic_a=30, topic_b=45),
        TopicPair("Obama vs Trump", topic_a=10, topic_b=39),
        TopicPair("Guns vs Crime", topic_a=46, topic_b=48),
        TopicPair("Religion vs Abortion", topic_a=15, topic_b=41),
    ]


def year_from_ts(ts: int) -> int:
    return datetime.fromtimestamp(ts, tz=timezone.utc).year


def stream_counts(
    doc_topics_path: Path,
    pairs: List[TopicPair],
    threshold: float,
    start_year: int,
    end_year: int,
) -> Tuple[Dict[int, Counter], Counter, Dict[Tuple[int, int], int], int]:
    target_ids = {p.topic_a for p in pairs} | {p.topic_b for p in pairs}
    per_year_counts: Dict[int, Counter] = defaultdict(Counter)
    total_docs_by_year: Counter = Counter()
    cooccur_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    total_docs = 0

    with doc_topics_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            yr = year_from_ts(int(rec["created_utc_min"]))
            if yr < start_year or yr > end_year:
                continue
            total_docs += 1
            total_docs_by_year[yr] += 1

            weights = rec["topic_distribution"]
            active = {tid for tid in target_ids if weights[tid] >= threshold}

            for tid in active:
                per_year_counts[yr][tid] += 1

            for pair in pairs:
                key = (pair.topic_a, pair.topic_b)
                if pair.topic_a in active and pair.topic_b in active:
                    cooccur_counts[key] += 1

            if line_no % 500000 == 0:
                print(f"Processed {line_no:,} documents...", flush=True)

    return per_year_counts, total_docs_by_year, cooccur_counts, total_docs


def prevalence_series(
    per_year_counts: Dict[int, Counter],
    totals_by_year: Counter,
    topic_id: int,
    years: List[int],
) -> List[float]:
    series: List[float] = []
    for y in years:
        total = totals_by_year.get(y, 0)
        count = per_year_counts.get(y, {}).get(topic_id, 0)
        series.append(count / total if total else 0.0)
    return series


def classify_relation(corr: float, pmi: float) -> str:
    if corr >= 0 and pmi >= 0:
        return "friendship (corr+, cooccur+)"
    if corr < 0 and pmi >= 0:
        return "tryst (corr-, cooccur+)"
    if corr >= 0 and pmi < 0:
        return "arms-race (corr+, cooccur-)"
    return "head-to-head (corr-, cooccur-)"


def compute_metrics(
    pairs: List[TopicPair],
    per_year_counts: Dict[int, Counter],
    totals_by_year: Counter,
    cooccur_counts: Dict[Tuple[int, int], int],
    total_docs: int,
    years: List[int],
) -> Dict[str, dict]:
    metrics: Dict[str, dict] = {}
    topic_totals: Counter = Counter()
    for yr, counts in per_year_counts.items():
        for tid, c in counts.items():
            topic_totals[tid] += c

    for pair in pairs:
        series_a = prevalence_series(per_year_counts, totals_by_year, pair.topic_a, years)
        series_b = prevalence_series(per_year_counts, totals_by_year, pair.topic_b, years)
        corr = float(np.corrcoef(series_a, series_b)[0, 1]) if any(series_a) and any(series_b) else 0.0

        pa = topic_totals[pair.topic_a] / total_docs if total_docs else 0.0
        pb = topic_totals[pair.topic_b] / total_docs if total_docs else 0.0
        pab = cooccur_counts.get((pair.topic_a, pair.topic_b), 0) / total_docs if total_docs else 0.0

        eps = 1e-12
        pmi = math.log((pab + eps) / max(pa * pb, eps))

        metrics[pair.name] = {
            "series_a": series_a,
            "series_b": series_b,
            "corr": corr,
            "pmi": pmi,
            "p_a": pa,
            "p_b": pb,
            "p_ab": pab,
            "classification": classify_relation(corr, pmi),
        }
    return metrics


def plot_relations(
    pairs: List[TopicPair],
    metrics: Dict[str, dict],
    topic_names: Dict[int, str],
    years: List[int],
    output_path: Path,
) -> None:
    sns.set(style="whitegrid")
    n = len(pairs)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
    axes = axes.flatten()

    palette = sns.color_palette("tab10", n)

    for idx, pair in enumerate(pairs):
        ax = axes[idx]
        m = metrics[pair.name]
        label_a = pair.label or topic_names.get(pair.topic_a, str(pair.topic_a))
        label_b = pair.label or topic_names.get(pair.topic_b, str(pair.topic_b))
        ax.plot(years, m["series_a"], label=label_a, color=palette[idx], linestyle="-", linewidth=2)
        ax.plot(years, m["series_b"], label=label_b, color=palette[idx], linestyle="--", linewidth=2)
        ax.set_title(m["classification"], fontsize=12, fontweight="bold")
        ax.set_ylabel("Prevalence (share of docs)")
        ax.set_xlabel("Year")
        ax.legend()

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Topic Relations (2008–2019)", fontsize=14, fontweight="bold", y=1.02)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    topic_names = load_topic_names(args.topics_json)
    pairs = load_pairs()
    years = list(range(args.start_year, args.end_year + 1))

    print("Streaming doc_topics.jsonl to collect counts (this may take a few minutes)...", flush=True)
    per_year_counts, totals_by_year, cooccur_counts, total_docs = stream_counts(
        args.doc_topics, pairs, args.presence_threshold, args.start_year, args.end_year
    )
    print(f"Finished counting {total_docs:,} documents.")

    metrics = compute_metrics(pairs, per_year_counts, totals_by_year, cooccur_counts, total_docs, years)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "relations_2008_2019_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {summary_path}")

    fig_path = args.output_dir / "relations_2008_2019.png"
    plot_relations(pairs, metrics, topic_names, years, fig_path)
    print(f"Wrote plot to {fig_path}")


if __name__ == "__main__":
    main()
