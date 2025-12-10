#!/usr/bin/env python3
"""
Create publication-ready visuals for a cleaned Reddit thread corpus:
- Histogram of document word counts with summary statistics.
- pyLDAvis interactive topic map for a sampled LDA fit.

Defaults target the combined 2008-2019 corpus, but paths and hyperparameters
are fully configurable via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyLDAvis
import pyLDAvis.lda_model
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build histogram + pyLDAvis outputs for a cleaned corpus."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpus_threads_2008_2019_clean.jsonl"),
        help="Path to cleaned corpus JSONL (one document with a 'text' field per line).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/2008_2019"),
        help="Directory to write visuals and stats to.",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=50,
        help="Number of topics for the visualization LDA fit (default: 50).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=50,
        help="min_df for the CountVectorizer (default: 50).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20_000,
        help="Vocabulary cap for the CountVectorizer (default: 20k).",
    )
    parser.add_argument(
        "--lda-max-docs",
        type=int,
        default=40_000,
        help="Maximum documents to keep for the LDA/pyLDAvis sample (reservoir sampled).",
    )
    parser.add_argument(
        "--clip-pct",
        type=float,
        default=99.5,
        help="Percentile to cap the x-axis for the length histogram (default: 99.5).",
    )
    parser.add_argument(
        "--title",
        default="Distribution of Thread Document Word Counts (2008-2019)",
        help="Title for the histogram plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reservoir sampling and LDA.",
    )
    return parser.parse_args()


def reservoir_sample_texts(
    path: Path, lda_max_docs: int, seed: int
) -> Tuple[List[int], List[str], int, int]:
    """Stream the corpus, compute word counts, and reservoir-sample texts for LDA."""
    rng = random.Random(seed)
    lengths: List[int] = []
    texts: List[str] = []
    total = 0
    skipped = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            text = (record.get("text") or "").strip()
            if not text:
                continue

            total += 1
            lengths.append(len(text.split()))

            if lda_max_docs:
                if len(texts) < lda_max_docs:
                    texts.append(text)
                else:
                    j = rng.randrange(total)
                    if j < lda_max_docs:
                        texts[j] = text

    return lengths, texts, total, skipped


def summarize_lengths(lengths: List[int]) -> dict:
    arr = np.array(lengths, dtype=np.int64)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def save_length_plot(lengths: List[int], stats: dict, title: str, clip_pct: float, out_path: Path, stats_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.histplot(lengths, bins=120, kde=True)
    plt.title(title)
    plt.xlabel("Word Count per Thread Document")
    plt.ylabel("Frequency")

    clip_at = np.percentile(lengths, clip_pct)
    plt.xlim(0, clip_at)

    annotation = "\n".join(
        [
            f"Documents: {stats['count']:,}",
            f"Mean: {stats['mean']:.1f} words",
            f"Median: {stats['median']:.1f} words",
            f"Min: {stats['min']:,} words",
            f"Max: {stats['max']:,} words",
            f"P90: {stats['p90']:.1f} words",
            f"P99: {stats['p99']:.1f} words",
        ]
    )
    plt.gca().text(
        0.98,
        0.95,
        annotation,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved histogram to {out_path}")
    print(f"Saved stats to {stats_path}")


def fit_lda(
    texts: List[str], num_topics: int, min_df: int, max_features: int, seed: int
) -> Tuple[LatentDirichletAllocation, CountVectorizer, any]:
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words=None,
    )
    doc_term = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        learning_method="online",
        batch_size=1024,
        max_iter=10,
        random_state=seed,
    )
    lda.fit(doc_term)
    return lda, vectorizer, doc_term


def save_pyldavis(
    lda: LatentDirichletAllocation,
    doc_term,
    vectorizer: CountVectorizer,
    output_path: Path,
) -> None:
    vis_data = pyLDAvis.lda_model.prepare(lda, doc_term, vectorizer, mds="tsne")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pyLDAvis.save_html(vis_data, str(output_path))
    print(f"Saved pyLDAvis visualization to {output_path}")


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")

    print(f"Streaming corpus from {args.corpus} ...")
    lengths, texts, total, skipped = reservoir_sample_texts(args.corpus, args.lda_max_docs, args.seed)
    if not lengths:
        raise ValueError("No documents loaded from corpus.")

    print(f"Loaded word counts for {len(lengths):,} documents (total seen: {total:,}).")
    if skipped:
        print(f"Skipped {skipped:,} lines that were not valid JSON.")
    print(f"Reservoir-sampled {len(texts):,} documents for LDA/pyLDAvis.")

    stats = summarize_lengths(lengths)
    save_length_plot(
        lengths,
        stats,
        args.title,
        args.clip_pct,
        args.output_dir / "document_length_distribution.png",
        args.output_dir / "document_length_stats.json",
    )

    print("Fitting LDA for visualization (this may take a few minutes)...")
    lda, vectorizer, doc_term = fit_lda(
        texts,
        args.num_topics,
        args.min_df,
        args.max_features,
        args.seed,
    )

    save_pyldavis(lda, doc_term, vectorizer, args.output_dir / "topics_pyldavis.html")
    print("Done.")


if __name__ == "__main__":
    main()
