#!/usr/bin/env python3
"""
Generate descriptive plots for the 2008 Reddit thread corpus.

Outputs:
1. Histogram of document word counts (+ summary stats overlay).
2. pyLDAvis HTML visualization for the fitted yearly LDA model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.lda_model
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build figures for the 2008 corpus.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("years/2008/comments/corpus/corpus_threads_2008_clean.jsonl"),
        help="Path to the cleaned JSONL corpus.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to store generated figures.",
    )
    parser.add_argument(
        "--lda-output",
        type=Path,
        default=Path("figures/2008_topics_pyldavis.html"),
        help="Destination HTML for the pyLDAvis visualization.",
    )
    parser.add_argument(
        "--num-topics",
        type=int,
        default=50,
        help="Number of topics for LDA (should match prior runs).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=50,
        help="Minimum document frequency for CountVectorizer.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20_000,
        help="Vocabulary cap for CountVectorizer.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Maximum LDA iterations.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_corpus(path: Path) -> pd.DataFrame:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    df["word_count"] = df["text"].str.split().str.len()
    return df


def save_doc_length_plot(df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "count": int(df["word_count"].count()),
        "mean": float(df["word_count"].mean()),
        "median": float(df["word_count"].median()),
        "min": int(df["word_count"].min()),
        "max": int(df["word_count"].max()),
    }

    plt.figure(figsize=(12, 6))
    sns.histplot(df["word_count"], bins=100, kde=True)
    plt.title("Distribution of Thread Document Word Counts (2008)")
    plt.xlabel("Word Count per Thread Document")
    plt.ylabel("Frequency")
    text = "\n".join(
        [
            f"Documents: {stats['count']:,}",
            f"Mean: {stats['mean']:.1f} words",
            f"Median: {stats['median']:.1f} words",
            f"Min: {stats['min']:,} words",
            f"Max: {stats['max']:,} words",
        ]
    )
    plt.gca().text(
        0.98,
        0.95,
        text,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    output_path = fig_dir / "2008_document_length_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    stats_path = fig_dir / "2008_document_length_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved histogram to {output_path}")
    print(f"Saved stats to {stats_path}")


def train_lda(
    texts: List[str],
    *,
    num_topics: int,
    min_df: int,
    max_features: int,
    max_iter: int,
    random_state: int,
) -> Tuple[LatentDirichletAllocation, CountVectorizer, any]:
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words=None,  # text already cleaned
    )
    doc_term = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        max_iter=max_iter,
        learning_method="online",
        batch_size=1024,
        random_state=random_state,
    )
    lda.fit(doc_term)
    return lda, vectorizer, doc_term


def save_pyldavis(
    lda: LatentDirichletAllocation,
    doc_term,
    vectorizer: CountVectorizer,
    output_path: Path,
) -> None:
    vis_data = pyLDAvis.lda_model.prepare(
        lda,
        doc_term,
        vectorizer,
        mds="tsne",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pyLDAvis.save_html(vis_data, str(output_path))
    print(f"Saved pyLDAvis visualization to {output_path}")


def main() -> None:
    args = parse_args()
    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")

    print("Loading corpus...")
    df = load_corpus(args.corpus)
    print(f"Loaded {len(df):,} documents.")

    print("Building document length histogram...")
    save_doc_length_plot(df, args.fig_dir)

    print("Training LDA for visualization...")
    lda, vectorizer, doc_term = train_lda(
        df["text"].tolist(),
        num_topics=args.num_topics,
        min_df=args.min_df,
        max_features=args.max_features,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    print("Preparing pyLDAvis output...")
    save_pyldavis(lda, doc_term, vectorizer, args.lda_output)


if __name__ == "__main__":
    main()
