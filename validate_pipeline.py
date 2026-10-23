r"""
InkMirror - pipeline validation against a known ground truth
-------------------------------------------------------------
Answers one question: **does this pipeline actually recover a sentiment trend, or
is it producing numbers that merely look like one?**

The synthetic sample corpus was generated from a scripted mood curve
(`make_sample_corpus.mood_for`). That curve is therefore a *known answer*. We can
score the corpus with TextBlob and check whether the detected polarity tracks the
curve we put in. If it does not, the pipeline has a bug somewhere between reading
the files and drawing the chart.

    output/validation.png      intended curve vs detected polarity

WHAT THIS PROVES, AND WHAT IT DOES NOT

It proves the plumbing is wired correctly: dates parse in the right order, entries
are scored, the smoothing runs over the right axis, nothing is dropped or shuffled
between stages. It is a regression test for the code.

**It does not prove TextBlob can read a real person's mood.** The synthetic entries
were written from mood-conditional word pools, so a correlation is guaranteed by
construction - the words were chosen to carry the sentiment. Real journal writing is
not built that way. Irony, understatement, and context are all invisible to a
lexicon, as the README sets out at length in section 8.

So the honest reading of a high correlation here is: *"the pipeline does not have a
bug"*, and nothing more. A low correlation would be a real alarm. A high one is the
expected result, not a discovery.

If the real journal is present, there is no ground truth to check against and this
module reports that and stops.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported

import matplotlib.pyplot as plt

import config
import load
import sentiment
import visualise

# This chart is deliberately monochrome. It is about whether two series agree, not
# about polarity sign or category, so spending hue on it would be noise. Ink for the
# measurement, grey for the reference - consistent with the timeline figure, where
# the detected trend is also drawn in ink.
COLOUR_DETECTED = config.COLOUR_INK
COLOUR_INTENDED = config.COLOUR_INK_MUTED


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient, computed directly.

    Avoids pulling in scipy for one number.
    """
    n = len(xs)
    if n < 2:
        return float("nan")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denominator = math.sqrt(var_x * var_y)
    return cov / denominator if denominator else float("nan")


def intended_curve(entries: list[load.Entry]) -> list[float]:
    """The mood value each day was generated from.

    Imported lazily so the pipeline does not depend on the generator at runtime.
    """
    from make_sample_corpus import mood_for

    return [mood_for(entry.entry_date) for entry in entries]


def strength(r: float) -> str:
    """Plain-language label for a correlation coefficient."""
    magnitude = abs(r)
    if math.isnan(r):
        return "undefined"
    if magnitude >= 0.8:
        return "very strong"
    if magnitude >= 0.6:
        return "strong"
    if magnitude >= 0.4:
        return "moderate"
    if magnitude >= 0.2:
        return "weak"
    return "negligible"


def plot_validation(dates, intended, detected, smoothed, r_entry, r_trend,
                    path=None) -> "plt.Figure":
    """Two panels: the curves over time, and the agreement as a scatter."""
    path = path or config.VALIDATION_PNG

    visualise._apply_style()
    fig, (ax_time, ax_scatter) = plt.subplots(
        1, 2, figsize=(13.5, 5.9), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    fig.subplots_adjust(left=0.062, right=0.975, top=0.755, bottom=0.135, wspace=0.20)

    # ---------------------------------------------------------- panel 1: over time
    ax_time.axhline(0, color=config.COLOUR_AXIS, linewidth=1.0, zorder=1)
    ax_time.plot(dates, intended, color=COLOUR_INTENDED, linewidth=1.6,
                 zorder=2, label="Mood the corpus was generated from")
    ax_time.plot(dates, smoothed, color=COLOUR_DETECTED, linewidth=2.1,
                 zorder=3, label="Detected by TextBlob (smoothed)")

    ax_time.set_title("Does the detected trend follow the known curve?",
                      fontsize=10.5, loc="left", pad=9)
    ax_time.set_ylabel("mood / polarity", fontsize=8.5,
                       color=config.COLOUR_INK_SECONDARY)
    # Upper left: both series sit low at the start of the period, so the top-left
    # corner is the one region neither line occupies. Lower left is where the
    # grey reference line dips to its minimum, and the legend sat on top of it.
    ax_time.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=1,
                   handlelength=1.8, borderaxespad=0.6)
    ax_time.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    ax_time.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b\n%Y"))
    ax_time.yaxis.grid(True, color=config.COLOUR_GRID, linewidth=0.8, zorder=0)
    ax_time.set_axisbelow(True)
    for side in ("top", "right"):
        ax_time.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_time.spines[side].set_color(config.COLOUR_AXIS)
        ax_time.spines[side].set_linewidth(0.8)
    ax_time.tick_params(length=0, labelsize=8.5, colors=config.COLOUR_INK_MUTED)

    # ---------------------------------------------------------- panel 2: scatter
    ax_scatter.axhline(0, color=config.COLOUR_GRID, linewidth=1.0, zorder=0)
    ax_scatter.axvline(0, color=config.COLOUR_GRID, linewidth=1.0, zorder=0)
    ax_scatter.scatter(intended, detected, s=17, c=COLOUR_DETECTED, alpha=0.35,
                       linewidths=0, zorder=2)

    # A best-fit line, drawn to show the direction and spread of agreement.
    if len(intended) > 1:
        n = len(intended)
        mean_x = sum(intended) / n
        mean_y = sum(detected) / n
        var_x = sum((x - mean_x) ** 2 for x in intended)
        if var_x:
            slope = sum((x - mean_x) * (y - mean_y)
                        for x, y in zip(intended, detected)) / var_x
            intercept = mean_y - slope * mean_x
            lo, hi = min(intended), max(intended)
            ax_scatter.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                            color=COLOUR_INTENDED, linewidth=1.8, zorder=3)

    ax_scatter.set_title(f"Agreement between the two curves (r = {r_entry:+.2f})",
                         fontsize=10.5, loc="left", pad=9)
    ax_scatter.set_xlabel("mood the corpus was generated from", fontsize=8.5,
                          color=config.COLOUR_INK_SECONDARY)
    ax_scatter.set_ylabel("TextBlob polarity", fontsize=8.5,
                          color=config.COLOUR_INK_SECONDARY)
    for side in ("top", "right"):
        ax_scatter.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax_scatter.spines[side].set_color(config.COLOUR_AXIS)
        ax_scatter.spines[side].set_linewidth(0.8)
    ax_scatter.tick_params(length=0, labelsize=8.5, colors=config.COLOUR_INK_MUTED)

    # ---------------------------------------------------------- titles
    fig.text(0.062, 0.945, "InkMirror - does the pipeline work?",
             fontsize=15, weight="semibold", color=config.COLOUR_INK, ha="left")
    fig.text(
        0.062, 0.885,
        f"The sample corpus was generated from a known mood curve, so the pipeline "
        f"can be checked against the right answer. Per-entry r = {r_entry:+.2f} "
        f"({strength(r_entry)}); smoothed trend r = {r_trend:+.2f} ({strength(r_trend)}).",
        fontsize=9.5, color=config.COLOUR_INK_SECONDARY, ha="left",
    )
    fig.text(
        0.062, 0.035,
        "This validates the code, not the method: the entries were written from "
        "mood-weighted word pools, so the correlation is built in. It shows there is "
        "no bug between reading the files and drawing the chart.",
        fontsize=8.5, color=config.COLOUR_INK_MUTED, ha="left",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI)
    return fig


def run(path=None) -> dict[str, float] | None:
    """Validate the pipeline. Returns None when there is no ground truth."""
    if not config.is_using_sample_corpus():
        print(
            "Skipped: the corpus is a real journal, so there is no known answer to\n"
            "         check the pipeline against. This validation is only possible\n"
            "         because the sample corpus was generated from a scripted curve."
        )
        return None

    report = load.load_entries()
    records = sentiment.score_all(report.entries)

    intended = intended_curve(report.entries)
    detected = [r.polarity for r in records]
    smoothed = visualise.rolling_mean(detected, config.SMOOTHING_WINDOW)
    smoothed_intended = visualise.rolling_mean(intended, config.SMOOTHING_WINDOW)

    r_entry = pearson(intended, detected)
    r_trend = pearson(smoothed_intended, smoothed)

    figure = plot_validation(
        [entry.entry_date for entry in report.entries],
        intended, detected, smoothed, r_entry, r_trend, path,
    )
    plt.close(figure)

    return {
        "entries": float(len(report.entries)),
        "r_entry": r_entry,
        "r_trend": r_trend,
    }


if __name__ == "__main__":
    result = run()
    if result:
        print(
            f"Checked {result['entries']:.0f} entries against the generating curve.\n"
            f"  per-entry correlation : r = {result['r_entry']:+.3f} "
            f"({strength(result['r_entry'])})\n"
            f"  smoothed trend        : r = {result['r_trend']:+.3f} "
            f"({strength(result['r_trend'])})"
        )
        print(f"-> {config.VALIDATION_PNG}")
        print(
            "\nThis confirms the pipeline has no bug between reading the files and\n"
            "drawing the chart. It is not evidence about how well a lexicon reads a\n"
            "real person's writing - the sample entries were built to carry a signal."
        )
