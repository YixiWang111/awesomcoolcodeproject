r"""
InkMirror - stretch goal: readability scoring
---------------------------------------------
Measures how hard each entry is to read, using two standard English formulas:

    Flesch Reading Ease      0-100, HIGHER means easier
                             60-70 is plain English, 90-100 is very easy
    Flesch-Kincaid Grade     US school grade level needed to read it

Both are computed from the same two ingredients: average sentence length in words,
and average syllables per word. textstat supplies the syllable counts, which need
NLTK's cmudict pronunciation dictionary.

WHY THIS IS A STRETCH GOAL AND NOT PART OF THE MAIN ANALYSIS

These formulas were designed to grade *school textbooks* - to answer "what age is
this text suitable for". A private journal has no target audience and no reading
age. Applying the formulas to it measures something adjacent to what they were
built for, and the numbers should be read as "how long are the sentences and how
long are the words", not as "how good is this writing".

They also are not independent of the lexical statistics. Reading ease is mostly
driven by sentence length, which `lexical.py` already measures directly. A change
in Flesch grade and a change in mean sentence length are largely the same finding
seen twice, not two pieces of evidence.

SHORT ENTRIES

A five-word entry produces a real-looking readability score from almost no
information. Entries below the thresholds in `config.py` are still measured and
written to the CSV, but flagged `reliable = no` and left out of the averages, so a
handful of one-line entries cannot drag the headline number around.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import textstat

import clean
import config
import load


@dataclass(frozen=True)
class ReadabilityRecord:
    """Readability of a single entry."""

    entry_date: str
    word_count: int
    sentence_count: int
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    reliable: bool

    @property
    def difficulty(self) -> str:
        """Plain-language band, the way the Flesch scale is normally read."""
        score = self.flesch_reading_ease
        if score >= 80:
            return "easy"
        if score >= 60:
            return "plain"
        if score >= 50:
            return "fairly hard"
        return "hard"


def measure(entry: load.Entry) -> ReadabilityRecord:
    """Score one entry, flagging whether the score means anything."""
    sentences = clean.sentences(entry.text)
    word_count = entry.word_count
    sentence_count = len(sentences)

    reliable = (
        word_count >= config.MIN_READABLE_WORDS
        and sentence_count >= config.MIN_READABLE_SENTENCES
    )

    return ReadabilityRecord(
        entry_date=entry.entry_date.isoformat(),
        word_count=word_count,
        sentence_count=sentence_count,
        flesch_reading_ease=round(textstat.flesch_reading_ease(entry.text), 2),
        flesch_kincaid_grade=round(textstat.flesch_kincaid_grade(entry.text), 2),
        reliable=reliable,
    )


def measure_all(entries: list[load.Entry]) -> list[ReadabilityRecord]:
    return [measure(entry) for entry in entries]


def write_csv(records: list[ReadabilityRecord], path=None) -> None:
    path = path or config.READABILITY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source",
                "date",
                "word_count",
                "sentence_count",
                "flesch_reading_ease",
                "flesch_kincaid_grade",
                "difficulty",
                "reliable",
            ]
        )
        for row in records:
            writer.writerow(
                [
                    config.corpus_source(),
                    row.entry_date,
                    row.word_count,
                    row.sentence_count,
                    row.flesch_reading_ease,
                    row.flesch_kincaid_grade,
                    row.difficulty,
                    "yes" if row.reliable else "no",
                ]
            )


def reliable_only(records: list[ReadabilityRecord]) -> list[ReadabilityRecord]:
    """The records actually worth averaging over."""
    return [r for r in records if r.reliable]


def summarise(records: list[ReadabilityRecord]) -> str:
    if not records:
        return "No entries measured."

    usable = reliable_only(records)
    excluded = len(records) - len(usable)

    lines = [
        f"Measured {len(records)} entries; {excluded} excluded from averages "
        f"(under {config.MIN_READABLE_WORDS} words or "
        f"{config.MIN_READABLE_SENTENCES} sentences).",
    ]

    if not usable:
        lines.append("No entry was long enough for a meaningful readability score.")
        return "\n".join(lines)

    mean_ease = sum(r.flesch_reading_ease for r in usable) / len(usable)
    mean_grade = sum(r.flesch_kincaid_grade for r in usable) / len(usable)
    lines.append(
        f"Mean reading ease: {mean_ease:.1f} "
        f"({min(r.flesch_reading_ease for r in usable):.0f}-"
        f"{max(r.flesch_reading_ease for r in usable):.0f}), "
        f"mean grade level: {mean_grade:.1f}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    report = load.load_entries()
    records = measure_all(report.entries)
    write_csv(records)
    print(summarise(records))
    print(f"-> {config.READABILITY_CSV}")
