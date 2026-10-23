r"""
InkMirror - Week 7: loading the corpus
--------------------------------------
Reads a folder of dated .txt files and returns them as an ordered list of Entry
objects. Nothing here analyses anything - it only gets the text off disk and in
the right order, so that every later stage can assume the data is clean and sorted.

Expected file naming (pathlib does the walking, a regex does the date parsing):

    entries/2024-09-01.txt          -> 2024-09-01
    entries/2024-09-01 morning.txt  -> 2024-09-01
    entries/notes.txt               -> skipped, no date in the filename
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import config

# Matches YYYY-MM-DD anywhere in a filename, so extra words are tolerated.
DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass(frozen=True)
class Entry:
    """One journal entry: its date, where it came from, and its raw text."""

    entry_date: date
    path: Path
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class LoadReport:
    """What the loader found, including everything it refused to load."""

    entries: list[Entry]
    skipped_undated: list[Path]
    skipped_empty: list[Path]

    @property
    def date_range(self) -> tuple[date, date] | None:
        if not self.entries:
            return None
        return self.entries[0].entry_date, self.entries[-1].entry_date

    @property
    def total_words(self) -> int:
        return sum(e.word_count for e in self.entries)


def parse_date_from_filename(path: Path) -> date | None:
    """Extract the date encoded in a filename, or None if there isn't one."""
    match = DATE_PATTERN.search(path.stem)
    if match is None:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        # Looks like a date but isn't one, e.g. 2024-13-45.
        return None


def load_entries(directory: Path | None = None) -> LoadReport:
    """Load every dated .txt file in `directory`, sorted oldest first.

    Files with no date in the name, and files that turn out to be empty, are
    skipped and reported rather than silently dropped - it matters to know when
    part of the corpus was not analysed.
    """
    directory = directory if directory is not None else config.corpus_dir()

    if not directory.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")

    entries: list[Entry] = []
    skipped_undated: list[Path] = []
    skipped_empty: list[Path] = []

    # sorted() over the paths keeps the loading order deterministic.
    for path in sorted(directory.glob("*.txt")):
        entry_date = parse_date_from_filename(path)
        if entry_date is None:
            skipped_undated.append(path)
            continue

        # errors="replace" so an odd encoding never kills the whole run.
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            skipped_empty.append(path)
            continue

        entries.append(Entry(entry_date=entry_date, path=path, text=text))

    entries.sort(key=lambda e: e.entry_date)
    return LoadReport(entries, skipped_undated, skipped_empty)


def describe(report: LoadReport) -> str:
    """A short human-readable summary of what was loaded, for the console."""
    if not report.entries:
        return "No entries loaded."

    first, last = report.date_range
    lines = [
        f"Loaded {len(report.entries)} entries "
        f"({first.isoformat()} to {last.isoformat()}).",
        f"Total words: {report.total_words:,}.",
    ]
    if report.skipped_undated:
        lines.append(
            f"Skipped {len(report.skipped_undated)} file(s) with no date in the name: "
            + ", ".join(p.name for p in report.skipped_undated[:5])
        )
    if report.skipped_empty:
        lines.append(
            f"Skipped {len(report.skipped_empty)} empty file(s): "
            + ", ".join(p.name for p in report.skipped_empty[:5])
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Reading: {config.corpus_dir()}")
    print(describe(load_entries()))
