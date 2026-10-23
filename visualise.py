r"""
InkMirror - Week 10: visualisation
----------------------------------
Turns the numbers from Weeks 8 and 9 into the two images that make up the
self-portrait: a sentiment timeline and a word cloud.

DESIGN DECISIONS, AND WHY

  Diverging, not categorical.
      Polarity has a real zero and two opposite poles, so it takes a diverging
      palette: blue and red, with neutral gray at the midpoint. The tempting
      choice is green/red, which is the wrong one - red-green colour blindness is
      the most common form, and those two poles collapse into each other. The
      blue/red pair was checked against a colour-vision simulator before use.

  One hue for the cloud.
      Word frequency is a magnitude, so the cloud runs one hue light-to-dark,
      darker meaning more frequent. A multi-coloured word cloud looks livelier
      and encodes nothing.

  Honesty is built into the picture, not just the report.
      - Points are drawn with an opacity set by how much the score can be trusted.
        A one-word entry that scores +0.5 should not look as solid as a paragraph
        that scores +0.5.
      - The smoothed line is the finding; the individual entries are context. The
        line is drawn in ink, not in a pole colour, so it cannot be misread as
        "the trend is positive".
      - If the corpus is the generated sample rather than the real journal, the
        image itself says so along the bottom. A chart of placeholder data should
        never be mistakable for a chart of real data.

  Every figure has a table-view twin.
      Both CSV files this module reads are the accessible equivalent of the
      images: every plotted value is readable as text there.

  Identity is never carried by colour alone.
      The positive / neutral / negative banding of a point is fully derivable from
      its position: above the zero line is positive, below it is negative, inside
      the shaded band is neutral. The hue repeats what the y-position already
      says, and a 2px surface ring separates points that overlap.

This is a static PNG for an academic submission, so there is no hover layer and
no dark mode; the light surface is the only one it renders on.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date

import matplotlib

matplotlib.use("Agg")  # write files without needing a display

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import config
import sentiment

# Opacity per confidence band. This is the secondary encoding: it carries the
# same information as the colour without relying on colour alone.
CONFIDENCE_ALPHA = {"high": 0.85, "medium": 0.45, "low": 0.18}

LABEL_COLOUR = {
    "positive": config.COLOUR_POSITIVE,
    "neutral": config.COLOUR_NEUTRAL,
    "negative": config.COLOUR_NEGATIVE,
}

# Sans-serif stack. Falls back through to matplotlib's default if none are found.
FONT_STACK = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
]


def _apply_style() -> None:
    """Global chart styling: thin marks, recessive chrome, no dashed rules."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "figure.facecolor": config.COLOUR_SURFACE,
            "axes.facecolor": config.COLOUR_SURFACE,
            "savefig.facecolor": config.COLOUR_SURFACE,
            "axes.edgecolor": config.COLOUR_AXIS,
            "axes.labelcolor": config.COLOUR_INK_SECONDARY,
            "text.color": config.COLOUR_INK,
            "xtick.color": config.COLOUR_INK_MUTED,
            "ytick.color": config.COLOUR_INK_MUTED,
            "axes.grid": False,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # solid hairlines, never dashed
        }
    )


def rolling_mean(values: list[float], window: int) -> list[float]:
    """Centred moving average, smoothing over entry index rather than time.

    Entries are irregularly spaced - sometimes three in a week, sometimes none for
    ten days - so a time-window average would weight quiet months differently from
    busy ones. A fixed window of *entries* keeps every point equally supported.
    """
    n = len(values)
    if n == 0:
        return []
    half = max(1, window // 2)
    smoothed = []
    for i in range(n):
        low = max(0, i - half)
        high = min(n, i + half + 1)
        chunk = values[low:high]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def _provenance_note() -> str:
    """A one-line provenance statement, printed onto every figure."""
    if config.is_using_sample_corpus():
        return (
            "Source: generated sample corpus (placeholder data, not a real journal). "
            "Re-run against entries/ once the real entries exist."
        )
    return "Source: private journal, analysed locally. No text left this machine."


def plot_timeline(records: list[sentiment.SentimentRecord], path=None) -> "plt.Figure":
    """Sentiment timeline: per-entry scores plus the smoothed trend."""
    path = path or config.TIMELINE_PNG
    if not records:
        raise ValueError("No sentiment records to plot.")

    _apply_style()

    dates = [date.fromisoformat(r.entry_date) for r in records]
    values = [r.polarity for r in records]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.175)

    # The neutral band: the range of scores that is not meaningfully positive or
    # negative. Drawn as the diverging midpoint fill, behind everything.
    ax.axhspan(-0.05, 0.05, color=config.COLOUR_NEUTRAL_BAND, zorder=0)

    # Horizontal grid, solid hairlines, behind the data.
    ax.yaxis.grid(True, color=config.COLOUR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # One scatter call per confidence band, so opacity can carry reliability.
    for confidence, alpha in CONFIDENCE_ALPHA.items():
        idx = [i for i, r in enumerate(records) if r.confidence == confidence]
        if not idx:
            continue
        ax.scatter(
            [dates[i] for i in idx],
            [values[i] for i in idx],
            s=17,
            c=[LABEL_COLOUR[records[i].label] for i in idx],
            alpha=alpha,
            linewidths=0.4,
            edgecolors=config.COLOUR_SURFACE,  # surface ring, not a border colour
            zorder=2,
        )

    # The zero line. Solid, since it is a real baseline rather than a grid rule.
    ax.axhline(0, color=config.COLOUR_AXIS, linewidth=1.5, zorder=3)

    # The trend: the finding, drawn in ink at 2px over the context points.
    smoothed = rolling_mean(values, config.SMOOTHING_WINDOW)
    ax.plot(
        dates,
        smoothed,
        color=config.COLOUR_TREND,
        linewidth=2.0,
        solid_capstyle="round",
        zorder=4,
        label=f"{config.SMOOTHING_WINDOW}-entry moving average",
    )

    # --- selective direct labels: only the two extremes, never every point ---
    lo_i = min(range(len(values)), key=lambda i: values[i])
    hi_i = max(range(len(values)), key=lambda i: values[i])
    for i, va in ((lo_i, "top"), (hi_i, "bottom")):
        ax.annotate(
            f"{values[i]:+.2f}  {dates[i]:%d %b %Y}",
            xy=(dates[i], values[i]),
            xytext=(0, -18 if va == "top" else 14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=config.COLOUR_INK_SECONDARY,
        )

    # --- chrome ---
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(config.COLOUR_AXIS)
        ax.spines[side].set_linewidth(0.8)

    ax.set_ylim(-0.55, 0.75)
    ax.set_ylabel(
        "TextBlob polarity\n(vocabulary score, not a measure of mood)",
        fontsize=9,
        labelpad=8,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="both", which="both", length=0, labelsize=8.5)

    # --- titles ---
    first, last = dates[0], dates[-1]
    fig.text(
        0.075, 0.945,
        "InkMirror - sentiment across two years of journal entries",
        fontsize=15, weight="semibold", color=config.COLOUR_INK, ha="left",
    )
    fig.text(
        0.075, 0.895,
        f"{len(records)} entries, {first:%d %b %Y} to {last:%d %b %Y}. "
        "Fainter points scored less reliably (short or unopinionated entries).",
        fontsize=9.5, color=config.COLOUR_INK_SECONDARY, ha="left",
    )

    # --- legend: always present for two or more series ---
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=LABEL_COLOUR[name], markeredgecolor="none",
                   label=f"{name.capitalize()}  ({_count(records, name)})")
        for name in ("positive", "neutral", "negative")
    ]
    handles.append(
        plt.Line2D([], [], color=config.COLOUR_TREND, linewidth=2,
                   label=f"{config.SMOOTHING_WINDOW}-entry average")
    )
    ax.legend(
        handles=handles, loc="upper left", ncol=4, frameon=False,
        fontsize=8.5, handletextpad=0.5, columnspacing=1.6,
        bbox_to_anchor=(0.0, 1.10),
    )

    # --- provenance, printed on the figure itself ---
    fig.text(
        0.075, 0.035, _provenance_note(),
        fontsize=8, color=config.COLOUR_INK_MUTED, ha="left",
    )
    fig.text(
        0.975, 0.035,
        "Values are also available in output/sentiment_scores.csv",
        fontsize=8, color=config.COLOUR_INK_MUTED, ha="right",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI)
    return fig


def _count(records: list[sentiment.SentimentRecord], label: str) -> int:
    return sum(1 for r in records if r.label == label)


# ------------------------------------------------------------------ word cloud

def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _sample_ramp(position: float) -> str:
    """Pick a colour from the sequential ramp at `position` in [0, 1]."""
    ramp = config.WORDCLOUD_RAMP
    if position <= 0:
        return ramp[0]
    if position >= 1:
        return ramp[-1]

    scaled = position * (len(ramp) - 1)
    low = int(math.floor(scaled))
    mix = scaled - low
    a = _hex_to_rgb(ramp[low])
    b = _hex_to_rgb(ramp[low + 1])
    blended = tuple(a[i] + (b[i] - a[i]) * mix for i in range(3))
    return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in blended)


def _font_path() -> str | None:
    """First available system sans, or None to let WordCloud choose."""
    import os

    for candidate in FONT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _frequency_colour_func(counter: Counter[str]):
    """Map each word's frequency onto the sequential ramp.

    Colour follows frequency, and frequency also drives font size, so the cloud
    encodes magnitude twice over in the same direction rather than spending hue
    on decoration. Frequencies are long-tailed, so the mapping is logarithmic -
    otherwise one dominant word would flatten everything else to the pale end.
    """
    counts = [c for c in counter.values() if c > 0]
    if not counts:
        return lambda **_: config.WORDCLOUD_RAMP[0]

    low, high = min(counts), max(counts)
    span = math.log(high / low + 1) if low > 0 else 1.0

    def colour_func(word, font_size=None, position=None, orientation=None,
                    random_state=None, **kwargs):
        count = counter.get(word, low)
        if span <= 0:
            return config.WORDCLOUD_RAMP[-1]
        position_on_ramp = math.log(count / low + 1) / span
        # High-frequency words are the point of the picture, so bias upward:
        # the square root keeps mid-frequency words legible instead of washing out.
        return _sample_ramp(position_on_ramp ** 0.6)

    return colour_func


def plot_wordcloud(counter: Counter[str], path=None) -> "plt.Figure":
    """Word cloud of the corpus's most frequent content words."""
    path = path or config.WORDCLOUD_PNG
    if not counter:
        raise ValueError("No word frequencies to plot.")

    _apply_style()

    # WordCloud wants log-ish frequencies to spread sizes; raw counts make one
    # word enormous. The colour function still reads the raw counts.
    frequencies = {word: float(count) for word, count in counter.most_common(config.TOP_N_WORDS)}

    cloud = WordCloud(
        width=1600,
        height=900,
        background_color=config.WORDCLOUD_BG,
        max_words=config.TOP_N_WORDS,
        prefer_horizontal=0.92,
        relative_scaling=0.42,
        margin=6,
        font_path=_font_path(),
        color_func=_frequency_colour_func(counter),
        random_state=7,  # fixed, so the cloud is reproducible
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(11, 6.6))
    fig.subplots_adjust(left=0, right=1, top=0.86, bottom=0.08)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")

    fig.text(
        0.035, 0.945,
        "InkMirror - the vocabulary of two years",
        fontsize=15, weight="semibold", color=config.COLOUR_INK, ha="left",
    )
    fig.text(
        0.035, 0.895,
        f"Top {min(config.TOP_N_WORDS, len(counter))} content words. "
        "Darker means more frequent; stopwords removed, so these are the words the "
        "corpus reaches for rather than what English requires.",
        fontsize=9.5, color=config.COLOUR_INK_SECONDARY, ha="left",
    )
    fig.text(
        0.035, 0.025, _provenance_note(),
        fontsize=8, color=config.COLOUR_INK_MUTED, ha="left",
    )
    fig.text(
        0.965, 0.025, "Counts in output/word_frequencies.csv",
        fontsize=8, color=config.COLOUR_INK_MUTED, ha="right",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI)
    return fig


if __name__ == "__main__":
    import lexical
    import load

    corpus = load.load_entries()
    records = sentiment.score_all(corpus.entries)

    timeline = plot_timeline(records)
    print(f"Wrote {config.TIMELINE_PNG}")
    plt.close(timeline)

    cloud_fig = plot_wordcloud(lexical.word_frequencies(corpus.entries))
    print(f"Wrote {config.WORDCLOUD_PNG}")
    plt.close(cloud_fig)
