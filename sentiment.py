r"""
InkMirror - Week 8: sentiment scoring
-------------------------------------
Scores every entry with TextBlob and writes the result to a CSV, so the numbers
exist as data before anything is drawn from them.

WHAT THE SCORE MEANS
    TextBlob returns `polarity`, a rough number in [-1.0, 1.0]:

        -1.0  strongly negative     0.0  neutral      +1.0  strongly positive

    It is a lexicon lookup, not a model that understands the entry. Each word is
    looked up in a fixed dictionary of pre-scored words and the results averaged.
    This is why the score must be presented honestly: it measures the *vocabulary*
    of an entry, not the mood of the person who wrote it.

    `subjectivity` (0.0 to 1.0) is reported alongside it because a polarity score
    computed from an entry with no opinionated words is close to meaningless and
    should not be plotted with the same confidence as one that has them.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from textblob import TextBlob

import config
import load


@dataclass(frozen=True)
class SentimentRecord:
    """Sentiment for a single entry."""

    entry_date: str          # ISO date, so the CSV sorts correctly as text
    polarity: float
    subjectivity: float
    word_count: int
    confidence: str          # "low" | "medium" | "high", see below

    @property
    def label(self) -> str:
        """A coarse band, used only for colouring the timeline."""
        if self.polarity > 0.05:
            return "positive"
        if self.polarity < -0.05:
            return "negative"
        return "neutral"


def confidence_for(subjectivity: float, word_count: int) -> str:
    """How much the polarity score deserves to be trusted.

    A short entry with few opinionated words produces an unstable score: one
    slightly-charged word can swing it. Flagging that explicitly is the honest
    way to present it, rather than drawing every point with equal weight.
    """
    if word_count >= 20 and subjectivity >= 0.30:
        return "high"
    if word_count >= 12 and subjectivity >= 0.15:
        return "medium"
    return "low"


def score_entry(entry: load.Entry) -> SentimentRecord:
    """Run TextBlob over one entry and package the result."""
    blob = TextBlob(entry.text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    return SentimentRecord(
        entry_date=entry.entry_date.isoformat(),
        polarity=round(polarity, 4),
        subjectivity=round(subjectivity, 4),
        word_count=entry.word_count,
        confidence=confidence_for(subjectivity, entry.word_count),
    )


def score_all(entries: list[load.Entry]) -> list[SentimentRecord]:
    """Score every entry, preserving chronological order."""
    return [score_entry(entry) for entry in entries]


def write_csv(records: list[SentimentRecord], path=None) -> None:
    """Write the scores to disk so later stages read data, not recompute it."""
    path = path or config.SENTIMENT_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "date", "polarity", "subjectivity", "word_count",
             "confidence", "label"]
        )
        for record in records:
            writer.writerow(
                [
                    config.corpus_source(),
                    record.entry_date,
                    record.polarity,
                    record.subjectivity,
                    record.word_count,
                    record.confidence,
                    record.label,
                ]
            )


def summarise(records: list[SentimentRecord]) -> str:
    """Console summary, deliberately understated about what the numbers mean."""
    if not records:
        return "No entries scored."

    values = [r.polarity for r in records]
    mean = sum(values) / len(values)
    positives = sum(1 for r in records if r.label == "positive")
    negatives = sum(1 for r in records if r.label == "negative")
    neutrals = len(records) - positives - negatives
    low_conf = sum(1 for r in records if r.confidence == "low")

    return "\n".join(
        [
            f"Scored {len(records)} entries.",
            f"Mean polarity: {mean:+.3f}  (min {min(values):+.3f}, max {max(values):+.3f})",
            f"Positive {positives} / neutral {neutrals} / negative {negatives}",
            f"Low-confidence scores: {low_conf} "
            f"({low_conf / len(records):.0%}) - short or unopinionated entries.",
        ]
    )


if __name__ == "__main__":
    report = load.load_entries()
    records = score_all(report.entries)
    write_csv(records)
    print(summarise(records))
    print(f"\nWritten to {config.SENTIMENT_CSV}")
