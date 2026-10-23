# InkMirror - run report

**Generated sample corpus - placeholder data, not a real journal.** Every number below describes the synthetic corpus in `entries_sample/`, which was produced by `make_sample_corpus.py` from a scripted mood curve. It is a test fixture, not a record of anyone's life. See README section 1.

## Corpus

| | |
|---|---|
| Entries analysed | 275 |
| Period | 2024-09-01 to 2026-08-29 |
| Total words | 9,983 |
| Distinct content words | 124 |
| Mean entry length | 36.3 words |
| Files skipped (no date in name) | 0 |
| Files skipped (empty) | 0 |

## Sentiment (TextBlob polarity, -1 to +1)

| | |
|---|---|
| Mean polarity | +0.121 |
| Mean subjectivity | 0.441 |
| Positive entries | 161 |
| Neutral entries | 59 |
| Negative entries | 55 |
| Low-confidence scores | 1 (0%) |

Mean polarity by calendar year:

| Year | Mean polarity |
|---|---|
| 2024 | +0.012 |
| 2025 | +0.105 |
| 2026 | +0.207 |

## Vocabulary

Most frequent content words across the corpus:

`felt` (236), `nothing` (179), `not` (154), `woke` (132), `everything` (124), `walk` (106), `could` (105), `afternoon` (98)

| | |
|---|---|
| Mean type-token ratio per entry | 0.932 |
| Mean sentence length | 8.7 words |

## What this does and does not show

The polarity score is a **lexicon average**: TextBlob looks each word up in a fixed
dictionary of pre-scored words and averages the result. It measures the vocabulary
of an entry, not the mood of the person who wrote it. A single strongly-charged word
moves it a long way, and irony, understatement and context are invisible to it.

A rising line in the timeline is therefore evidence about *word choice*, and only
weakly about how the year actually felt. See README section 8 for the full list.

## Stretch goals

Optional extras, declared as "if ahead" in the pitch and kept out of the core
scope. Their outputs are in `output/stretch/` rather than alongside the main
results, so the boundary stays visible.

| Goal | Finding |
|---|---|
| Readability scoring | Measured 275 entries; 13 excluded from averages (under 20 words or 2 sentences). Mean reading ease: 74.6 (25-100), mean grade level: 5.0 |
| Two-period comparison | mean polarity +0.065 then +0.177 (+0.112) |
| Topic tagging | 102 of 275 entries (37%) matched no topic keyword |

Each of these carries its own caveat, and none of them is as solid as it looks.
**Readability** formulas were built to grade school textbooks, and the score is
mostly driven by sentence length - which the lexical statistics already measure, so
it is the same finding seen twice rather than independent evidence. **The period
split** is arbitrary: nothing happened at the midpoint, and a real change that
straddles the boundary would be averaged away. **Topic tagging** is plain keyword
matching, so it only finds a subject if the right word was guessed in advance, and
the tagged share swings widely depending on how many keywords are required to fire.
