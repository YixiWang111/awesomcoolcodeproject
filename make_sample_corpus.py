r"""
InkMirror - placeholder corpus generator
----------------------------------------
PURPOSE
    Creates two years of *synthetic* diary entries so the InkMirror pipeline can be
    built and tested before the real journal is available.

    *** THESE ARE NOT REAL JOURNAL ENTRIES. ***
    Output goes to `entries_sample/`, never to `entries/`, so the real corpus and the
    test corpus can never be confused with one another. Point config.CORPUS_DIR at
    `entries/` once the real entries exist - no other code changes are needed.

DESIGN
    Entries are generated from a mood curve so the resulting sentiment timeline has
    genuine structure to detect, rather than noise:

      - slow two-year recovery trend (mildly upward)
      - seasonal dip every winter
      - sharp drops during exam periods (November and May)
      - day-to-day noise

    Vocabulary is drawn from mood-weighted pools, so TextBlob polarity should track
    the intended curve. Recurring verbal tics are injected on purpose so the
    frequency analysis and word cloud have something real to find.

    A fixed seed makes the corpus reproducible.

USAGE
    .\.venv\Scripts\python.exe make_sample_corpus.py
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------- configuration

SEED = 42
START = date(2024, 9, 1)
END = date(2026, 8, 31)
OUT_DIR = Path(__file__).parent / "entries_sample"
ENTRIES_PER_WEEK = (2, 4)  # min / max entries written in a given week

# Verbal tics the author allegedly leans on without noticing. These are injected
# into most entries so the lexical statistics surface them as top-frequency items.
TICS = [
    "I keep circling back to the same thing",
    "somehow",
    "I don't really know why",
    "the same walk",
    "tired",
    "I suppose",
    "and that is fine",
    "for now",
]

TOPICS = [
    "uni", "work", "friends", "running", "sleep", "family", "reading", "flat",
]

# Sentence pools keyed by mood band. Each entry draws from the band matching the
# day's mood, which is what makes TextBlob's polarity follow the curve.
OPENERS = {
    "high": [
        "Woke up early and the light through the blinds was genuinely lovely.",
        "A good day, for once.",
        "Slept properly and it changed everything.",
        "Woke before the alarm, which almost never happens.",
    ],
    "mid": [
        "Woke up at seven, made coffee, sat down.",
        "An ordinary day with nothing much in it.",
        "Woke up tired but functional.",
        "Nothing remarkable happened today.",
    ],
    "low": [
        "Woke up heavy again.",
        "Could not get out of bed properly this morning.",
        "Woke at four and could not get back to sleep.",
        "The alarm went off and I just lay there.",
    ],
}

BODY = {
    "high": [
        "Work went well and I actually felt proud of what I finished.",
        "Spent the afternoon with {topic} and came home grateful and calm.",
        "There was a moment on the walk where everything felt simple and good.",
        "I finished the thing I had been avoiding and felt relieved and light.",
        "A quiet, warm, ordinary kind of happiness that I want to remember.",
    ],
    "mid": [
        "Did some reading for {topic}, nothing especially memorable.",
        "Met {topic} briefly and talked about nothing in particular.",
        "Worked through the afternoon, steady but unremarkable.",
        "Managed a short walk and a bit of {topic} before it got dark.",
        "Everything was fine, honestly, just flat.",
    ],
    "low": [
        "Everything with {topic} felt like too much effort today.",
        "I was anxious all afternoon and could not settle to anything.",
        "Felt lonely in a way I could not explain to anyone.",
        "The whole day felt like wading through something heavy.",
        "I kept putting things off and then felt guilty and exhausted about it.",
    ],
}

CLOSERS = {
    "high": [
        "I want more days like this.",
        "Writing it down so I remember it was real.",
        "Good night, genuinely.",
    ],
    "mid": [
        "Tomorrow, then.",
        "Nothing more to say about it really.",
        "Same again tomorrow, probably.",
    ],
    "low": [
        "I keep circling back to the same thing and getting nowhere.",
        "I suppose it will look different in a week. It usually does.",
        "I don't really know why I am writing this. For now, that is enough.",
    ],
}


# ---------------------------------------------------------------- mood curve

def mood_for(day: date) -> float:
    """Intended mood on a given day, roughly in [-1, 1].

    Composed of a slow upward trend, a winter seasonal dip, exam-period shocks,
    and a little deterministic jitter so consecutive days are not identical.
    """
    elapsed = (day - START).days
    span = (END - START).days

    # Slow two-year recovery: starts negative, ends mildly positive.
    trend = -0.25 + 0.50 * (elapsed / span)

    # Seasonal component: index 0.0 at Jan 1, dips through winter.
    doy = day.timetuple().tm_yday
    seasonal = -0.28 * math.cos(2 * math.pi * (doy - 15) / 365.0)

    # Exam periods: November and May are rough.
    exam = 0.0
    if day.month == 11:
        exam -= 0.45
    if day.month == 5:
        exam -= 0.30

    # Deterministic per-day jitter: a hash-free but stable pseudo-noise.
    jitter = 0.18 * math.sin(elapsed * 12.9898) * math.cos(elapsed * 4.1414)

    value = trend + seasonal + exam + jitter
    return max(-1.0, min(1.0, value))


def mood_band(value: float) -> str:
    if value > 0.18:
        return "high"
    if value < -0.20:
        return "low"
    return "mid"


# ---------------------------------------------------------------- generation

def compose_entry(day: date, rng: random.Random) -> str:
    mood = mood_for(day)
    band = mood_band(mood)

    parts: list[str] = [rng.choice(OPENERS[band])]

    # Longer entries at the weekend, shorter on busy weekdays, as a real journal
    # would be. This gives the "sentence length" statistic something to measure.
    n_body = rng.randint(1, 2) if day.weekday() < 5 else rng.randint(2, 4)
    # Sample without replacement so a single entry never repeats itself, and let
    # the topic change sentence to sentence rather than fixing it for the day.
    templates = rng.sample(BODY[band], k=min(n_body, len(BODY[band])))
    for template in templates:
        # Never let the topic duplicate a noun the template already contains,
        # otherwise we get "did some reading for reading".
        options = [t for t in TOPICS if t not in template] or TOPICS
        parts.append(template.format(topic=rng.choice(options)))

    # Inject a verbal tic in most entries.
    if rng.random() < 0.75:
        parts.append(rng.choice(TICS).capitalize() + ".")

    parts.append(rng.choice(CLOSERS[band]))
    return " ".join(parts)


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    written = 0
    day = START
    while day <= END:
        # Choose 2-4 weekdays this week to write on.
        for _ in range(rng.randint(*ENTRIES_PER_WEEK)):
            offset = rng.randint(0, 6)
            entry_day = day + timedelta(days=offset)
            if entry_day > END:
                continue
            path = OUT_DIR / f"{entry_day.isoformat()}.txt"
            if path.exists():
                continue
            path.write_text(compose_entry(entry_day, rng), encoding="utf-8")
            written += 1
        day += timedelta(days=7)

    print(f"Wrote {written} synthetic entries to {OUT_DIR}")
    print("These are placeholder data, not real journal entries.")


if __name__ == "__main__":
    main()
