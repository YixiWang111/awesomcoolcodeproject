r"""
InkMirror - Week 9: lexical statistics
--------------------------------------
Describes the shape of the writing rather than its mood:

    per entry   word count, unique words, type-token ratio, average sentence length
    whole corpus  frequency of every content word, via collections.Counter

Type-token ratio (unique words / total words) is the most interesting of these.
It drops as an entry gets longer, which is expected, but it also drops when the
author is writing on autopilot. Comparing it across two years is one way to see
whether the vocabulary actually widened or just repeated itself.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass

import config
import clean
import load


@dataclass(frozen=True)
class LexicalStats:
    """Writing-shape statistics for a single entry."""

    entry_date: str
    word_count: int
    unique_words: int
    type_token_ratio: float
    sentence_count: int
    avg_sentence_length: float


def stats_for(entry: load.Entry) -> LexicalStats:
    """Compute the per-entry statistics.

    Uniqueness is measured on content words, not raw tokens: counting "the" once
    instead of five times says nothing about vocabulary, it just measures length.
    """
    content = clean.content_words(entry.text)
    lengths = clean.sentence_lengths(entry.text)

    unique = len(set(content))
    total = len(content)

    return LexicalStats(
        entry_date=entry.entry_date.isoformat(),
        word_count=entry.word_count,
        unique_words=unique,
        type_token_ratio=round(unique / total, 4) if total else 0.0,
        sentence_count=len(lengths),
        avg_sentence_length=round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
    )


def stats_for_all(entries: list[load.Entry]) -> list[LexicalStats]:
    return [stats_for(entry) for entry in entries]


def word_frequencies(entries: list[load.Entry]) -> Counter[str]:
    """Count every content word across the whole corpus.

    This is the author's vocabulary habit made countable: the words that come
    back without being noticed.
    """
    counter: Counter[str] = Counter()
    for entry in entries:
        counter.update(clean.content_words(entry.text))
    return counter


def write_stats_csv(stats: list[LexicalStats], path=None) -> None:
    path = path or config.LEXICAL_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "date",
                "word_count",
                "unique_words",
                "type_token_ratio",
                "sentence_count",
                "avg_sentence_length",
            ]
        )
        for row in stats:
            writer.writerow(
                [
                    config.corpus_source(),
                    row.entry_date,
                    row.word_count,
                    row.unique_words,
                    row.type_token_ratio,
                    row.sentence_count,
                    row.avg_sentence_length,
                ]
            )


def write_frequency_csv(counter: Counter[str], path=None, top_n=None) -> None:
    path = path or config.FREQUENCY_CSV
    top_n = top_n or config.TOP_N_WORDS
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "word", "count"])
        for word, count in counter.most_common(top_n):
            writer.writerow([config.corpus_source(), word, count])


def summarise(stats: list[LexicalStats], counter: Counter[str]) -> str:
    if not stats:
        return "No entries analysed."

    total_unique = len(counter)
    total_words = sum(s.word_count for s in stats)
    avg_ttr = sum(s.type_token_ratio for s in stats) / len(stats)
    avg_sentence = sum(s.avg_sentence_length for s in stats) / len(stats)

    top = ", ".join(f"{w} ({c})" for w, c in counter.most_common(10))

    return "\n".join(
        [
            f"Entries: {len(stats)}, total words: {total_words:,}",
            f"Distinct content words across the corpus: {total_unique:,}",
            f"Mean type-token ratio per entry: {avg_ttr:.3f}",
            f"Mean sentence length: {avg_sentence:.1f} words",
            f"Most frequent: {top}",
        ]
    )


if __name__ == "__main__":
    report = load.load_entries()
    stats = stats_for_all(report.entries)
    counter = word_frequencies(report.entries)
    write_stats_csv(stats)
    write_frequency_csv(counter)
    print(summarise(stats, counter))
