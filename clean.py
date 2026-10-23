r"""
InkMirror - Week 7: cleaning and tokenising
-------------------------------------------
Turns raw entry text into the two forms the rest of the pipeline needs:

    tokens()            -> every word, lowercased, for sentiment scoring
    content_words()     -> the same minus stopwords, for frequency analysis

The distinction matters. Sentiment scoring wants the stopwords left in, because
negation and pronouns carry meaning ("I am not happy" is not "happy"). Frequency
analysis wants them gone, because "the" would otherwise dominate every result
and tell us nothing about the author's vocabulary habits.
"""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import config

# Anything that is not a letter, digit or internal apostrophe is a separator.
# Keeps "don't" together while dropping surrounding punctuation.
TOKEN_CLEANER = re.compile(r"[^a-z0-9']+")

# Common contractions left after tokenising. Expanding these means "don't" and
# "do not" reach the sentiment lexicon as the same thing.
CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i'm": "i am",
    "i've": "i have",
    "i'd": "i would",
    "i'll": "i will",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
}

# Words that carry real signal about a person's writing but are technically
# stopwords. Kept out of the stopword list so they survive into the frequency
# table and the word cloud, where they are genuinely informative.
KEEP_WORDS = {"not", "no", "nor", "never", "very", "too", "only", "again", "still"}


@lru_cache(maxsize=1)
def _stopword_set() -> frozenset[str]:
    """NLTK's English stopword list, minus the words we deliberately keep.

    If the NLTK stopwords corpus has not been downloaded, fall back to a small
    built-in list rather than crashing - the analysis is slightly noisier but
    still runs, and the caller is told.
    """
    try:
        words = set(stopwords.words(config.LANGUAGE))
    except LookupError:
        print(
            "Warning: NLTK stopwords not found, using a reduced built-in list.\n"
            "         Run: python -c \"import nltk; nltk.download('stopwords')\""
        )
        words = {
            "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
            "with", "about", "into", "to", "from", "in", "on", "is", "are", "was",
            "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "i", "you", "he", "she", "it", "we", "they", "my", "your",
            "his", "her", "its", "our", "their", "this", "that", "these", "those",
        }
    return frozenset(words - KEEP_WORDS)


def normalise(text: str) -> str:
    """Lowercase, expand contractions and collapse whitespace.

    Sentiment lexicons match on exact surface forms, so "Today was NOT good"
    has to become "today was not good" before scoring.
    """
    lowered = text.lower()
    # Pad with spaces so a contraction at the start or end still matches.
    padded = f" {lowered} "
    for contraction, expansion in CONTRACTIONS.items():
        padded = padded.replace(f" {contraction} ", f" {expansion} ")
    return " ".join(padded.split())


def tokens(text: str) -> list[str]:
    """Every word, lowercased, punctuation stripped. Stopwords kept.

    Used for sentiment scoring, where removing stopwords would break negation.
    """
    normalised = normalise(text)
    raw = word_tokenize(normalised)
    cleaned = (TOKEN_CLEANER.sub("", token) for token in raw)
    return [token for token in cleaned if token]


def content_words(text: str) -> list[str]:
    """Tokens with stopwords and very short words removed.

    Used for frequency analysis and the word cloud, where function words would
    otherwise swamp everything the author actually chose to write about.
    """
    stops = _stopword_set()
    return [
        token
        for token in tokens(text)
        if token not in stops
        and len(token) >= config.MIN_TOKEN_LENGTH
        and not token.isdigit()
    ]


def sentences(text: str) -> list[str]:
    """Split text into sentences, for measuring average sentence length."""
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def sentence_lengths(text: str) -> list[int]:
    """Word count of each sentence in the text."""
    return [len(word_tokenize(s)) for s in sentences(text)]


if __name__ == "__main__":
    sample = "Today was NOT good, and I don't know why.  I suppose it will pass."
    print("normalised:   ", normalise(sample))
    print("tokens:       ", tokens(sample))
    print("content:      ", content_words(sample))
    print("sentences:    ", sentences(sample))
