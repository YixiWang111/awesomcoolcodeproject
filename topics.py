r"""
InkMirror - stretch goal: simple topic tagging
----------------------------------------------
Tags each entry with the subjects it touches, using plain keyword matching, and
shows how those subjects move across the two years.

    topic_tags.csv          per entry: which topics matched, and on which words
    topics_over_time.png    topic by quarter, as a heatmap

HOW THE TAGGING WORKS

Each topic has a list of keywords. An entry is tagged with every topic for which at
least `MIN_KEYWORD_HITS` of its content words appear. Entries get multiple tags
rather than one, because a real entry is usually about more than one thing, and
forcing a single label would invent a certainty the method does not have.

THE KEYWORD LISTS ARE THE WHOLE METHOD, AND THEY ARE GUESSES

This is the honest part. The quality of the tagging is exactly the quality of
`TOPIC_KEYWORDS` below. A topic is only found if the words someone happens to have
picked in advance are the words the writer happens to use. **Edit these lists for
your own journal** - read a sample of entries, note what you actually call things,
and replace the keywords. Out of the box they are a reasonable generic guess, and
nothing more.

The `unmatched` count this module reports is the number that matters. It is the
share of entries for which no keyword fired, meaning the method found nothing at
all in them. If that number is high, the keyword lists are wrong, and no amount of
chart polish fixes it.

WHAT THIS IS NOT

It is not classification. There is no model, no training, and no evaluation. It is a
`word in list` test, and it should be read as an index into the journal rather than
as a finding about it.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import clean
import config
import load
import visualise  # for the shared provenance note on every figure


# ------------------------------------------------------------------ the keyword lists
#
# EDIT THESE for the real journal. Each entry maps a topic name to the words that
# signal it. Words must be lowercase and will be matched against content words
# (stopwords already removed). Multi-word phrases are not supported - this is a
# deliberately simple method.

TOPIC_KEYWORDS: dict[str, set[str]] = {
    "University": {
        "uni", "university", "lecture", "lectures", "seminar", "essay", "essays",
        "deadline", "deadlines", "assignment", "coursework", "revision", "exam",
        "exams", "tutor", "tutorial", "lecturer", "campus", "degree", "module",
    },
    "Work": {
        "work", "worked", "working", "shift", "shifts", "rota", "manager",
        "colleague", "colleagues", "job", "meeting", "meetings", "office",
        "project", "client", "payslip", "overtime",
    },
    "Friends": {
        "friends", "friend", "met", "meet", "talked", "talk", "coffee", "dinner",
        "flatmate", "flatmates", "visited", "message", "messaged", "called",
        "drinks", "party", "together", "catch",
    },
    "Family": {
        "family", "mum", "dad", "mother", "father", "sister", "brother",
        "parents", "grandma", "grandad", "aunt", "uncle", "cousin", "home",
    },
    "Sleep & body": {
        "sleep", "slept", "sleeping", "tired", "woke", "wake", "waking", "awake",
        "exhausted", "rested", "insomnia", "nap", "headache", "ill", "sick",
        "energy", "heavy", "alarm",
    },
    "Movement": {
        "walk", "walks", "walked", "walking", "run", "running", "ran", "gym",
        "swim", "swimming", "cycle", "cycling", "yoga", "exercise", "stretch",
    },
    "Reading & writing": {
        "reading", "read", "book", "books", "novel", "chapter", "poem", "poetry",
        "writing", "wrote", "write", "journal", "notes", "diary", "draft",
    },
    "Home": {
        "flat", "house", "kitchen", "cooking", "cooked", "cleaning", "cleaned",
        "laundry", "tidying", "rent", "washing", "dishes", "shopping", "groceries",
    },
    "Weather & season": {
        "rain", "raining", "rained", "cold", "warm", "sun", "sunny", "grey",
        "gray", "winter", "summer", "autumn", "spring", "snow", "wind",
        "dark", "light",
    },
}

# How many of a topic's keywords must appear before the entry gets that tag.
#
# This single number changes the results more than anything else in the module.
# At 1, one stray word tags an entry, coverage climbs toward 100%, and the tags
# stop distinguishing anything. At 3, only entries that dwell on a subject survive,
# and most entries go untagged. There is no correct value - which is exactly why
# `summarise()` reports the coverage at several thresholds rather than just this
# one. 2 is the default because it is the least flattering defensible choice.
MIN_KEYWORD_HITS = 2


@dataclass(frozen=True)
class TopicTag:
    """The topics found in one entry, and the words that triggered them."""

    entry_date: str
    topics: tuple[str, ...]
    keyword_hits: int
    matched: str  # "Topic:word,word; Topic2:word" - for auditing a tagging run


def _relative_luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG relative luminance of a colour given as 0-1 RGB."""
    channels = []
    for component in rgb[:3]:
        channels.append(
            component / 12.92
            if component <= 0.04045
            else ((component + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """WCAG contrast ratio between two colours."""
    lum_a, lum_b = _relative_luminance(a), _relative_luminance(b)
    lighter, darker = max(lum_a, lum_b), min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def _readable_ink(cell_rgba) -> str:
    """Pick whichever of the two inks has more contrast against this cell.

    Choosing the label colour by a fixed value threshold gets it wrong near the
    crossover, because how dark a cell looks depends on the ramp, not the number.
    Measuring the actual contrast is correct whatever the ramp is.
    """
    white = (1.0, 1.0, 1.0)
    dark = _hex_to_rgb(config.COLOUR_INK)
    return "#FFFFFF" if _contrast(cell_rgba, white) > _contrast(cell_rgba, dark) \
        else config.COLOUR_INK


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def tag_entry(entry: load.Entry, min_hits: int | None = None) -> TopicTag:
    """Find every topic whose keywords appear in this entry."""
    min_hits = MIN_KEYWORD_HITS if min_hits is None else min_hits
    words = set(clean.content_words(entry.text))
    found: list[str] = []
    evidence: list[str] = []
    total_hits = 0

    for topic, keywords in TOPIC_KEYWORDS.items():
        hits = sorted(words & keywords)
        if len(hits) >= min_hits:
            found.append(topic)
            total_hits += len(hits)
            evidence.append(f"{topic}:{','.join(hits)}")

    return TopicTag(
        entry_date=entry.entry_date.isoformat(),
        topics=tuple(found),
        keyword_hits=total_hits,
        matched="; ".join(evidence),
    )


def tag_all(entries: list[load.Entry]) -> list[TopicTag]:
    return [tag_entry(entry) for entry in entries]


def write_csv(tags: list[TopicTag], path=None) -> None:
    path = path or config.TOPIC_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "date", "topics", "topic_count", "keyword_hits", "matched_words"]
        )
        for tag in tags:
            writer.writerow(
                [
                    config.corpus_source(),
                    tag.entry_date,
                    "; ".join(tag.topics),
                    len(tag.topics),
                    tag.keyword_hits,
                    tag.matched,
                ]
            )


# ------------------------------------------------------------------ over-time view

def _quarter_key(iso_date: str) -> str:
    """'2025-04-11' -> '2025 Q2'."""
    year, month, _ = iso_date.split("-")
    return f"{year} Q{(int(month) - 1) // 3 + 1}"


def topic_by_quarter(tags: list[TopicTag]) -> tuple[list[str], list[str], list[list[float]]]:
    """Share of each quarter's entries that carry each topic.

    Expressed as a share rather than a count, because quarters hold different
    numbers of entries and a count would mostly show how much was written.
    """
    quarters = sorted({_quarter_key(tag.entry_date) for tag in tags})
    quarter_sizes: Counter[str] = Counter(_quarter_key(t.entry_date) for t in tags)

    # Only topics that actually fired somewhere; an all-zero row is noise.
    fired = [topic for topic in TOPIC_KEYWORDS
             if any(topic in tag.topics for tag in tags)]

    grid: list[list[float]] = []
    for topic in fired:
        row = []
        for quarter in quarters:
            total = quarter_sizes[quarter] or 1
            hits = sum(
                1 for tag in tags
                if _quarter_key(tag.entry_date) == quarter and topic in tag.topics
            )
            row.append(hits / total)
        grid.append(row)

    return fired, quarters, grid


def plot_topics_over_time(tags: list[TopicTag], path=None) -> "plt.Figure":
    """Heatmap of topic share by quarter, on a single-hue sequential ramp."""
    path = path or config.TOPICS_OVER_TIME_PNG
    if not tags:
        raise ValueError("No tags to plot.")

    topics, quarters, grid = topic_by_quarter(tags)
    if not topics:
        raise ValueError(
            "No topic keyword matched any entry. The keyword lists in "
            "TOPIC_KEYWORDS need editing for this journal."
        )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "figure.facecolor": config.COLOUR_SURFACE,
        "axes.facecolor": config.COLOUR_SURFACE,
        "savefig.facecolor": config.COLOUR_SURFACE,
        "text.color": config.COLOUR_INK,
    })
    cmap = LinearSegmentedColormap.from_list("inkmirror_seq", config.TOPIC_HEATMAP_RAMP)

    width = max(9.0, 1.05 * len(quarters) + 3.6)
    height = max(4.6, 0.52 * len(topics) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.20, right=0.93, top=0.775, bottom=0.145)

    image = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(quarters)))
    ax.set_xticklabels(quarters, fontsize=9)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics, fontsize=9.5)

    # A 2px surface gap between cells, rather than borders drawn around them.
    ax.set_xticks([x - 0.5 for x in range(1, len(quarters))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(topics))], minor=True)
    ax.grid(which="minor", color=config.COLOUR_SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)

    # Every cell carries its value, so the figure is readable without colour and
    # acts as its own table view.
    for row_i, row in enumerate(grid):
        for col_i, value in enumerate(row):
            if value == 0:
                continue  # don't print a wall of zeroes
            ax.text(
                col_i, row_i, f"{value:.0%}",
                ha="center", va="center", fontsize=8.5,
                color=_readable_ink(cmap(value)),
            )

    bar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    bar.set_label("share of that quarter's entries", fontsize=8.5,
                  color=config.COLOUR_INK_SECONDARY)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8, colors=config.COLOUR_INK_MUTED)

    unmatched = sum(1 for tag in tags if not tag.topics)
    fig.text(0.20, 0.945, "InkMirror - what the entries were about",
             fontsize=15, weight="semibold", color=config.COLOUR_INK, ha="left")
    fig.text(
        0.20, 0.885,
        f"Keyword matching, {MIN_KEYWORD_HITS} hit(s) minimum. "
        f"{unmatched} of {len(tags)} entries ({unmatched / len(tags):.0%}) matched no "
        "keyword at all - those entries are simply missing from this figure.",
        fontsize=9.5, color=config.COLOUR_INK_SECONDARY, ha="left",
    )
    fig.text(
        0.20, 0.072,
        "The keyword lists are guesses at this writer's vocabulary; a topic is only "
        "found if the right word was guessed in advance.",
        fontsize=8.5, color=config.COLOUR_INK_MUTED, ha="left",
    )
    fig.text(
        0.20, 0.025, visualise._provenance_note(),
        fontsize=8, color=config.COLOUR_INK_MUTED, ha="left",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI)
    return fig


def coverage_sensitivity(entries: list[load.Entry], thresholds=(1, 2, 3)) -> str:
    """How much the tagging result depends on the hit threshold.

    This is the honest headline. If coverage at 1 hit and coverage at 3 hits are
    wildly different, the tags are an artefact of a number someone chose, not a
    property of the journal.
    """
    parts = []
    for threshold in thresholds:
        tagged = [tag_entry(entry, threshold) for entry in entries]
        matched = sum(1 for tag in tagged if tag.topics)
        parts.append(
            f"{threshold} hit(s): {matched}/{len(entries)} entries tagged "
            f"({matched / len(entries):.0%})"
        )
    return "Coverage by threshold -> " + " | ".join(parts)


def summarise(tags: list[TopicTag], entries: list[load.Entry] | None = None) -> str:
    if not tags:
        return "No entries tagged."

    counts: Counter[str] = Counter()
    for tag in tags:
        counts.update(tag.topics)

    unmatched = sum(1 for tag in tags if not tag.topics)
    never_fired = [t for t in TOPIC_KEYWORDS if counts[t] == 0]

    lines = [
        f"Tagged {len(tags)} entries against {len(TOPIC_KEYWORDS)} topics "
        f"(minimum {MIN_KEYWORD_HITS} keyword hits).",
        f"Entries matching no keyword: {unmatched} ({unmatched / len(tags):.0%})",
    ]
    if entries:
        lines.append(coverage_sensitivity(entries))
    if counts:
        lines.append(
            "Most common: "
            + ", ".join(f"{t} ({c})" for t, c in counts.most_common(6))
        )
    if never_fired:
        lines.append(
            "Topics that never fired: " + ", ".join(never_fired)
            + "  <- the keyword lists may need editing"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    report = load.load_entries()
    tags = tag_all(report.entries)
    write_csv(tags)
    print(summarise(tags, report.entries))
    print(f"-> {config.TOPIC_CSV}")
