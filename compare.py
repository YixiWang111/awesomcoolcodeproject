r"""
InkMirror - stretch goal: side-by-side comparison of two periods
----------------------------------------------------------------
Splits the journal into two halves and asks whether the second half reads
differently from the first:

    period_summary.csv          every measure, per period
    period_comparison.png       small multiples, one panel per measure
    word_change.png             which words gained or lost ground

HOW THE SPLIT IS MADE

At the midpoint *entry*, not at a calendar boundary. A calendar split would give
two periods of different sizes, and every count-based measure would then differ
just because one period is longer. Splitting at the median entry keeps the halves
the same size, which is the only way the comparison says anything.

The split is still arbitrary. Nothing happened at that date; it is simply the
middle of the data. A real change that straddles the boundary gets averaged away,
and a change that happens to sit inside one half gets attributed to "the period"
as a whole.

WHY SMALL MULTIPLES AND NOT ONE CHART

The six measures have incompatible units - polarity runs -1 to 1, reading ease 0
to 100, sentence length about 9 words. Putting them on one chart would need a
second y-axis, and two y-scales on one plot invent a relationship that isn't in the
data. Each measure therefore gets its own small panel with its own axis.

WHY THE BARS ARE GREY

Blue and red already mean "positive" and "negative" everywhere else in this
project. Using them here for "first half" and "second half" would carry the wrong
association into the reader's head. The periods are a plain before/after grouping,
so they take two shades of one neutral hue, validated as an ordinal ramp.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt

import clean
import config
import lexical
import load
import readability
import sentiment
import visualise  # for the shared provenance note on every figure

# Ordinal ramp: one neutral hue, two steps, light end still clears the surface.
PERIOD_COLOURS = ("#898781", "#52514E")

# Diverging pair for "more often / less often", validated at CVD dE 21.6.
COLOUR_MORE = config.COLOUR_POSITIVE
COLOUR_LESS = config.COLOUR_NEGATIVE


@dataclass(frozen=True)
class PeriodMetric:
    """One measure, compared across the two periods."""

    name: str
    first: float
    second: float
    fmt: str          # printf-style format for the value labels
    axis_label: str


@dataclass(frozen=True)
class SplitCorpus:
    """The corpus cut in half, with every per-entry record list split to match."""

    entries: list[load.Entry]
    sentiment: list[sentiment.SentimentRecord]
    stats: list[lexical.LexicalStats]
    reading: list[readability.ReadabilityRecord]

    def halves(self, records: list) -> tuple[list, list]:
        """Split any per-entry list at the same midpoint as the corpus."""
        mid = len(self.entries) // 2
        return records[:mid], records[mid:]

    @property
    def boundary(self):
        """Date of the first entry in the second half."""
        mid = len(self.entries) // 2
        return self.entries[mid].entry_date if self.entries else None


def split(
    corpus: load.LoadReport,
    sentiment_records: list[sentiment.SentimentRecord],
    stats: list[lexical.LexicalStats],
    reading: list[readability.ReadabilityRecord],
) -> SplitCorpus:
    """Bundle the corpus and its record lists so they split consistently."""
    return SplitCorpus(
        entries=corpus.entries,
        sentiment=sentiment_records,
        stats=stats,
        reading=reading,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_metrics(corpus: SplitCorpus) -> list[PeriodMetric]:
    """Every measure, computed for each half."""
    s_first, s_second = corpus.halves(corpus.sentiment)
    l_first, l_second = corpus.halves(corpus.stats)
    r_first, r_second = corpus.halves(corpus.reading)
    e_first, e_second = corpus.halves(corpus.entries)

    # Readability averages use only entries long enough for the formula to mean
    # anything, so a few one-line entries cannot move the headline.
    r_first_ok = readability.reliable_only(r_first)
    r_second_ok = readability.reliable_only(r_second)

    return [
        PeriodMetric("Mean polarity", _mean([r.polarity for r in s_first]),
                     _mean([r.polarity for r in s_second]), "%+.3f",
                     "TextBlob polarity (-1 to +1)"),
        PeriodMetric("Subjectivity", _mean([r.subjectivity for r in s_first]),
                     _mean([r.subjectivity for r in s_second]), "%.3f",
                     "TextBlob subjectivity (0 to 1)"),
        PeriodMetric("Words per entry", _mean([e.word_count for e in e_first]),
                     _mean([e.word_count for e in e_second]), "%.1f",
                     "words"),
        PeriodMetric("Sentence length", _mean([s.avg_sentence_length for s in l_first]),
                     _mean([s.avg_sentence_length for s in l_second]), "%.1f",
                     "words per sentence"),
        PeriodMetric("Type-token ratio", _mean([s.type_token_ratio for s in l_first]),
                     _mean([s.type_token_ratio for s in l_second]), "%.3f",
                     "unique / total content words"),
        PeriodMetric("Reading ease", _mean([r.flesch_reading_ease for r in r_first_ok]),
                     _mean([r.flesch_reading_ease for r in r_second_ok]), "%.1f",
                     "Flesch reading ease (higher = easier)"),
    ]


def write_summary_csv(metrics: list[PeriodMetric], corpus: SplitCorpus, path=None) -> None:
    path = path or config.PERIOD_SUMMARY_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    first_label, second_label = config.PERIOD_LABELS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "measure", first_label, second_label, "change", "unit"])
        for metric in metrics:
            writer.writerow(
                [
                    config.corpus_source(),
                    metric.name,
                    f"{metric.first:.4f}",
                    f"{metric.second:.4f}",
                    f"{metric.second - metric.first:+.4f}",
                    metric.axis_label,
                ]
            )


def plot_period_comparison(metrics: list[PeriodMetric], corpus: SplitCorpus,
                           path=None) -> "plt.Figure":
    """Small multiples: one panel per measure, each with its own axis."""
    path = path or config.PERIOD_COMPARISON_PNG
    if not corpus.entries:
        raise ValueError("No entries to compare.")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "figure.facecolor": config.COLOUR_SURFACE,
        "axes.facecolor": config.COLOUR_SURFACE,
        "savefig.facecolor": config.COLOUR_SURFACE,
        "text.color": config.COLOUR_INK,
    })

    first_label, second_label = config.PERIOD_LABELS
    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12.5, 7.2))
    fig.subplots_adjust(left=0.06, right=0.975, top=0.775, bottom=0.135,
                        wspace=0.42, hspace=0.62)

    for ax, metric in zip(axes.flat, metrics):
        values = [metric.first, metric.second]

        # width 0.62 of a unit spacing leaves a clear gap between the two bars.
        ax.bar([0, 1], values, width=0.62, color=PERIOD_COLOURS, zorder=2)

        # Direct value labels: with only two bars, every value is worth showing.
        for x, value in zip([0, 1], values):
            ax.annotate(
                metric.fmt % value,
                xy=(x, value),
                xytext=(0, 5 if value >= 0 else -13),
                textcoords="offset points",
                ha="center", fontsize=9.5, color=config.COLOUR_INK,
            )

        # A zero rule only matters when values can go negative.
        if min(values) < 0:
            ax.axhline(0, color=config.COLOUR_AXIS, linewidth=1.0, zorder=1)

        ax.set_title(metric.name, fontsize=10.5, color=config.COLOUR_INK,
                     pad=8, loc="left")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([first_label, second_label], fontsize=8.5)
        ax.set_ylabel(metric.axis_label, fontsize=7.5,
                      color=config.COLOUR_INK_MUTED, labelpad=6)

        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(config.COLOUR_AXIS)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(length=0, labelsize=8.5, colors=config.COLOUR_INK_MUTED)

        # A bar encodes its value as a length measured from zero, so zero is always
        # the baseline. Padding the axis below zero would float the bars off their
        # own scale and make the pair look closer together than they are. Headroom
        # is added above only, to clear the value labels.
        low = min(values + [0.0])
        high = max(values + [0.0])
        span = (high - low) or 1.0
        ax.set_ylim(low, high + span * 0.30)

    boundary = corpus.boundary
    fig.text(0.06, 0.945, "InkMirror - the journal in two halves",
             fontsize=15, weight="semibold", color=config.COLOUR_INK, ha="left")
    fig.text(
        0.06, 0.885,
        f"Split at the midpoint entry ({boundary:%d %b %Y}): "
        f"{len(corpus.entries) // 2} entries against "
        f"{len(corpus.entries) - len(corpus.entries) // 2}. Each panel has its own "
        "scale - the measures are not comparable to one another.",
        fontsize=9.5, color=config.COLOUR_INK_SECONDARY, ha="left",
    )
    fig.text(
        0.06, 0.068,
        "A difference between the halves is not evidence of change: the split date is "
        "arbitrary and there is no significance test.",
        fontsize=8.5, color=config.COLOUR_INK_MUTED, ha="left",
    )
    fig.text(
        0.06, 0.025, visualise._provenance_note(),
        fontsize=8, color=config.COLOUR_INK_MUTED, ha="left",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI)
    return fig


# ------------------------------------------------------- word level comparison

def word_change(corpus: SplitCorpus, top_n=None, min_total=None
                ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Words that gained and lost the most ground between the halves.

    Frequencies are normalised to occurrences per 10,000 words before subtracting,
    so the result measures a change in *rate* rather than in raw count - otherwise
    a word would look like it grew simply because one half is longer.
    """
    top_n = top_n or config.WORD_CHANGE_TOP_N
    min_total = min_total if min_total is not None else config.WORD_CHANGE_MIN_TOTAL

    e_first, e_second = corpus.halves(corpus.entries)
    if not e_first or not e_second:
        return [], []

    def rates(entries: list[load.Entry]) -> tuple[Counter[str], int]:
        counter: Counter[str] = Counter()
        for entry in entries:
            counter.update(clean.content_words(entry.text))
        return counter, sum(entry.word_count for entry in entries) or 1

    first_counts, first_total = rates(e_first)
    second_counts, second_total = rates(e_second)

    deltas: list[tuple[str, float]] = []
    for word in set(first_counts) | set(second_counts):
        total = first_counts[word] + second_counts[word]
        if total < min_total:
            continue  # too rare for a change in rate to mean anything
        first_rate = first_counts[word] / first_total * 10_000
        second_rate = second_counts[word] / second_total * 10_000
        deltas.append((word, second_rate - first_rate))

    deltas.sort(key=lambda item: item[1])
    decreases = deltas[:top_n]              # most negative first
    increases = list(reversed(deltas[-top_n:]))  # most positive first
    return increases, decreases


def plot_word_change(increases: list[tuple[str, float]],
                     decreases: list[tuple[str, float]], path=None) -> "plt.Figure":
    """Diverging horizontal bars: words that rose or fell between the halves."""
    path = path or config.WORD_CHANGE_PNG
    if not increases and not decreases:
        raise ValueError("No words qualified for the change comparison.")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "figure.facecolor": config.COLOUR_SURFACE,
        "axes.facecolor": config.COLOUR_SURFACE,
        "savefig.facecolor": config.COLOUR_SURFACE,
        "text.color": config.COLOUR_INK,
    })

    first_label, second_label = config.PERIOD_LABELS
    rows = increases + decreases
    labels = [word for word, _ in rows]
    values = [value for _, value in rows]
    colours = [COLOUR_MORE if v >= 0 else COLOUR_LESS for v in values]
    positions = list(range(len(rows)))[::-1]  # largest increase at the top

    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    fig.subplots_adjust(left=0.175, right=0.965, top=0.815, bottom=0.115)

    ax.barh(positions, values, height=0.62, color=colours, zorder=2)
    ax.axvline(0, color=config.COLOUR_AXIS, linewidth=1.2, zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9.5, color=config.COLOUR_INK)

    # Only the two extremes carry a number; a label on every bar would be noise.
    for i, (value, pos) in enumerate(zip(values, positions)):
        if i == 0 or i == len(values) - 1:
            ax.annotate(
                f"{value:+.1f}",
                xy=(value, pos),
                xytext=(6 if value >= 0 else -6, 0),
                textcoords="offset points",
                va="center", ha="left" if value >= 0 else "right",
                fontsize=9, color=config.COLOUR_INK_SECONDARY,
            )

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(config.COLOUR_AXIS)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", length=0, labelsize=8.5, colors=config.COLOUR_INK_MUTED)
    ax.tick_params(axis="y", length=0)

    limit = max(abs(v) for v in values) * 1.18
    ax.set_xlim(-limit, limit)
    ax.xaxis.grid(True, color=config.COLOUR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        f"change in frequency, occurrences per 10,000 words "
        f"({second_label} minus {first_label})",
        fontsize=9, color=config.COLOUR_INK_SECONDARY, labelpad=8,
    )

    # Legend: two series, so it is always present.
    handles = [
        plt.Line2D([], [], color=COLOUR_MORE, linewidth=7,
                   label=f"more frequent in {second_label.lower()}"),
        plt.Line2D([], [], color=COLOUR_LESS, linewidth=7,
                   label=f"more frequent in {first_label.lower()}"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5)

    fig.text(0.175, 0.955, "InkMirror - words that gained and lost ground",
             fontsize=15, weight="semibold", color=config.COLOUR_INK, ha="left")
    fig.text(
        0.175, 0.905,
        f"Most-changed words between the two halves. Words used fewer than "
        f"{config.WORD_CHANGE_MIN_TOTAL} times in total are excluded.",
        fontsize=9.5, color=config.COLOUR_INK_SECONDARY, ha="left",
    )
    fig.text(
        0.175, 0.058,
        "Words are counted as content words, so stopwords are not eligible.",
        fontsize=8.5, color=config.COLOUR_INK_MUTED, ha="left",
    )
    fig.text(
        0.175, 0.018, visualise._provenance_note(),
        fontsize=8, color=config.COLOUR_INK_MUTED, ha="left",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI)
    return fig


def summarise(metrics: list[PeriodMetric]) -> str:
    """Console summary of the headline comparison."""
    if not metrics:
        return "Nothing to compare."
    first_label, second_label = config.PERIOD_LABELS
    lines = [f"{'measure':<20}{first_label:>14}{second_label:>14}{'change':>12}"]
    for metric in metrics:
        lines.append(
            f"{metric.name:<20}{metric.first:>14.3f}{metric.second:>14.3f}"
            f"{metric.second - metric.first:>+12.3f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    corpus = load.load_entries()
    s_records = sentiment.score_all(corpus.entries)
    l_stats = lexical.stats_for_all(corpus.entries)
    r_records = readability.measure_all(corpus.entries)
    parts = split(corpus, s_records, l_stats, r_records)
    metrics = build_metrics(parts)
    write_summary_csv(metrics, parts)
    print(summarise(metrics))
