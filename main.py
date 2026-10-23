r"""
InkMirror - Week 11: the whole pipeline, start to finish
--------------------------------------------------------
Runs Weeks 7 to 10 in order against the corpus and writes every output:

    Week 7   load.py, clean.py      read the folder, tokenise
    Week 8   sentiment.py           score every entry        -> sentiment_scores.csv
    Week 9   lexical.py             describe the vocabulary  -> lexical_stats.csv,
                                                               word_frequencies.csv
    Week 10  visualise.py           draw the results         -> sentiment_timeline.png,
                                                               wordcloud.png
    Week 11  main.py                this file - run it all, report, and note limits

It also writes output/run_report.md, a summary of what this particular run found,
which is the artefact to hand in alongside the images.

Usage:
    .\.venv\Scripts\python.exe main.py

Scope note. The core analysis is deliberately limited to one language and a single
polarity score, with no emotion classification. The stretch goals named in the pitch
- readability scoring, comparing two time periods, topic tagging - are implemented
in `readability.py`, `compare.py` and `topics.py`, and run as a stage at the end.

They are kept in a separate stage writing to output/stretch/ so the declared scope
stays legible: everything in output/ is core, everything in output/stretch/ is the
optional extra. Each of the three is weaker than it looks, and the report says so.
"""

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt

import clean
import compare
import config
import lexical
import load
import readability
import sentiment
import topics
import validate_pipeline
import visualise


def banner(step: str, title: str) -> None:
    print(f"\n{'-' * 68}\n  {step}  {title}\n{'-' * 68}")


def compare_periods(records: list[sentiment.SentimentRecord]) -> dict[str, float]:
    """Mean polarity per calendar year.

    This is used only as a headline in the run report, NOT as the side-by-side
    period comparison listed as a stretch goal - it is one number per year with no
    confidence interval and no split by season, which would be needed to claim a
    real change. Stretch goal, deliberately out of scope.
    """
    by_year: dict[str, list[float]] = {}
    for record in records:
        year = record.entry_date[:4]
        by_year.setdefault(year, []).append(record.polarity)
    return {year: sum(vals) / len(vals) for year, vals in sorted(by_year.items())}


def run_stretch_goals(corpus: load.LoadReport, records, stats) -> dict[str, str]:
    """Run the three stretch goals and return a line of finding for each.

    These are the extras named in the pitch as "if ahead": readability scoring,
    side-by-side comparison of two periods, and simple topic tagging. They are kept
    in a separate stage and write to output/stretch/ so the boundary of the
    declared scope stays visible rather than blurring into the core work.
    """
    findings: dict[str, str] = {}

    # --- readability -------------------------------------------------------
    reading = readability.measure_all(corpus.entries)
    readability.write_csv(reading)
    print(readability.summarise(reading))
    print(f"-> {config.READABILITY_CSV}")
    findings["readability"] = readability.summarise(reading).replace("\n", " ")

    # --- two-period comparison --------------------------------------------
    parts = compare.split(corpus, records, stats, reading)
    metrics = compare.build_metrics(parts)
    compare.write_summary_csv(metrics, parts)
    print()
    print(compare.summarise(metrics))
    print(f"-> {config.PERIOD_SUMMARY_CSV}")

    figure = compare.plot_period_comparison(metrics, parts)
    plt.close(figure)
    print(f"-> {config.PERIOD_COMPARISON_PNG}")

    increases, decreases = compare.word_change(parts)
    figure = compare.plot_word_change(increases, decreases)
    plt.close(figure)
    print(f"-> {config.WORD_CHANGE_PNG}")

    polarity = next(m for m in metrics if m.name == "Mean polarity")
    findings["periods"] = (
        f"mean polarity {polarity.first:+.3f} then {polarity.second:+.3f} "
        f"({polarity.second - polarity.first:+.3f})"
    )

    # --- topic tagging -----------------------------------------------------
    tags = topics.tag_all(corpus.entries)
    topics.write_csv(tags)
    print()
    print(topics.summarise(tags, corpus.entries))
    print(f"-> {config.TOPIC_CSV}")

    figure = topics.plot_topics_over_time(tags)
    plt.close(figure)
    print(f"-> {config.TOPICS_OVER_TIME_PNG}")

    unmatched = sum(1 for tag in tags if not tag.topics)
    findings["topics"] = (
        f"{unmatched} of {len(tags)} entries ({unmatched / len(tags):.0%}) "
        "matched no topic keyword"
    )

    return findings


def build_run_report(
    corpus: load.LoadReport,
    records: list[sentiment.SentimentRecord],
    stats: list[lexical.LexicalStats],
    counter,
    stretch: dict[str, str] | None = None,
) -> str:
    """Markdown summary of one run, ready to hand in."""
    first, last = corpus.date_range
    mean_polarity = sum(r.polarity for r in records) / len(records)
    subjective = sum(r.subjectivity for r in records) / len(records)
    low_conf = sum(1 for r in records if r.confidence == "low")
    top_words = counter.most_common(15)
    per_year = compare_periods(records)

    word_habits = ", ".join(f"`{w}` ({c})" for w, c in top_words[:8])
    year_lines = "\n".join(
        f"| {year} | {value:+.3f} |" for year, value in per_year.items()
    )

    provenance = (
        "**Generated sample corpus - placeholder data, not a real journal.** "
        "Every number below describes the synthetic corpus in `entries_sample/`, "
        "which was produced by `make_sample_corpus.py` from a scripted mood curve. "
        "It is a test fixture, not a record of anyone's life. See README section 1."
        if config.is_using_sample_corpus()
        else "Analysed locally from the private journal in `entries/`. "
             "No entry text left this machine."
    )

    if stretch:
        stretch_section = f"""
## Stretch goals

Optional extras, declared as "if ahead" in the pitch and kept out of the core
scope. Their outputs are in `output/stretch/` rather than alongside the main
results, so the boundary stays visible.

| Goal | Finding |
|---|---|
| Readability scoring | {stretch.get('readability', 'not run')} |
| Two-period comparison | {stretch.get('periods', 'not run')} |
| Topic tagging | {stretch.get('topics', 'not run')} |

Each of these carries its own caveat, and none of them is as solid as it looks.
**Readability** formulas were built to grade school textbooks, and the score is
mostly driven by sentence length - which the lexical statistics already measure, so
it is the same finding seen twice rather than independent evidence. **The period
split** is arbitrary: nothing happened at the midpoint, and a real change that
straddles the boundary would be averaged away. **Topic tagging** is plain keyword
matching, so it only finds a subject if the right word was guessed in advance, and
the tagged share swings widely depending on how many keywords are required to fire.
"""
    else:
        stretch_section = ""

    return f"""# InkMirror - run report

{provenance}

## Corpus

| | |
|---|---|
| Entries analysed | {len(corpus.entries)} |
| Period | {first.isoformat()} to {last.isoformat()} |
| Total words | {corpus.total_words:,} |
| Distinct content words | {len(counter):,} |
| Mean entry length | {corpus.total_words / len(corpus.entries):.1f} words |
| Files skipped (no date in name) | {len(corpus.skipped_undated)} |
| Files skipped (empty) | {len(corpus.skipped_empty)} |

## Sentiment (TextBlob polarity, -1 to +1)

| | |
|---|---|
| Mean polarity | {mean_polarity:+.3f} |
| Mean subjectivity | {subjective:.3f} |
| Positive entries | {sum(1 for r in records if r.label == 'positive')} |
| Neutral entries | {sum(1 for r in records if r.label == 'neutral')} |
| Negative entries | {sum(1 for r in records if r.label == 'negative')} |
| Low-confidence scores | {low_conf} ({low_conf / len(records):.0%}) |

Mean polarity by calendar year:

| Year | Mean polarity |
|---|---|
{year_lines}

## Vocabulary

Most frequent content words across the corpus:

{word_habits}

| | |
|---|---|
| Mean type-token ratio per entry | {sum(s.type_token_ratio for s in stats) / len(stats):.3f} |
| Mean sentence length | {sum(s.avg_sentence_length for s in stats) / len(stats):.1f} words |

## What this does and does not show

The polarity score is a **lexicon average**: TextBlob looks each word up in a fixed
dictionary of pre-scored words and averages the result. It measures the vocabulary
of an entry, not the mood of the person who wrote it. A single strongly-charged word
moves it a long way, and irony, understatement and context are invisible to it.

A rising line in the timeline is therefore evidence about *word choice*, and only
weakly about how the year actually felt. See README section 8 for the full list.
{stretch_section}"""


def main() -> None:
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------- Week 7
    banner("Week 7", "Load and tokenise")
    print(f"Corpus: {config.corpus_dir()}")
    if config.is_using_sample_corpus():
        print(
            "NOTE: no entries/ folder found, so the generated sample corpus is being\n"
            "      used. These results describe placeholder data, not a real journal."
        )
    corpus = load.load_entries()
    print(load.describe(corpus))

    sample_tokens = clean.content_words(corpus.entries[0].text)
    print(f"\nFirst entry, {len(sample_tokens)} content words: {sample_tokens[:12]} ...")

    # ---------------------------------------------------------- Week 8
    banner("Week 8", "Sentiment scoring")
    records = sentiment.score_all(corpus.entries)
    sentiment.write_csv(records)
    print(sentiment.summarise(records))
    print(f"-> {config.SENTIMENT_CSV}")

    # ---------------------------------------------------------- Week 9
    banner("Week 9", "Lexical statistics")
    stats = lexical.stats_for_all(corpus.entries)
    counter = lexical.word_frequencies(corpus.entries)
    lexical.write_stats_csv(stats)
    lexical.write_frequency_csv(counter)
    print(lexical.summarise(stats, counter))
    print(f"-> {config.LEXICAL_CSV}")
    print(f"-> {config.FREQUENCY_CSV}")

    # ---------------------------------------------------------- Week 10
    banner("Week 10", "Visualisation")
    figure = visualise.plot_timeline(records)
    plt.close(figure)
    print(f"-> {config.TIMELINE_PNG}")

    cloud = visualise.plot_wordcloud(counter)
    plt.close(cloud)
    print(f"-> {config.WORDCLOUD_PNG}")

    # ---------------------------------------------------------- Validation
    banner("CHECK", "Pipeline validation against a known curve")
    validation = validate_pipeline.run()
    if validation:
        print(
            f"Checked against the generating curve: "
            f"per-entry r = {validation['r_entry']:+.3f} "
            f"({validate_pipeline.strength(validation['r_entry'])}), "
            f"smoothed r = {validation['r_trend']:+.3f} "
            f"({validate_pipeline.strength(validation['r_trend'])})."
        )
        print(f"-> {config.VALIDATION_PNG}")
        print(
            "   This shows the code has no bug from file to chart. It is NOT evidence\n"
            "   that a lexicon reads real writing well - the sample was built to carry\n"
            "   a signal."
        )

    # ------------------------------------------- Stretch goals (not in scope)
    banner("EXTRA", "Stretch goals - readability, two periods, topic tagging")
    print(
        "Optional extras, declared as \"if ahead\" in the pitch. Outputs go to\n"
        f"{config.STRETCH_DIR} so they stay separate from the core deliverables.\n"
    )
    stretch = run_stretch_goals(corpus, records, stats)

    # ---------------------------------------------------------- Week 11
    banner("Week 11", "Run report and limitations")
    report_path = config.OUTPUT_DIR / "run_report.md"
    report_path.write_text(
        build_run_report(corpus, records, stats, counter, stretch), encoding="utf-8"
    )
    print(f"-> {report_path}")
    print("-> README.md  (sections 1 and 8: corpus statement, limitations)")

    print("\nDone.")


if __name__ == "__main__":
    main()
