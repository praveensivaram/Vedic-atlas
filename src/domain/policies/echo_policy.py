"""EchoPolicy — did the model hand the user's own message back instead of answering?

Small quantized models fall into "continue the input" mode: a short statement in the user's language comes
back nearly verbatim, sometimes with a word completed. This is a reply filter, not a language detector and
not a paraphrase check: only near-identity counts. Pure text logic, no I/O.

Normalisation keeps every combining mark (Devanagari/Telugu matras are Unicode category Mn, not punctuation)
and drops only whitespace runs and punctuation, so "आज मेरा जन्मदिन है?" and "आज मेरा जन्मदिन है" compare equal.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

DEFAULT_THRESHOLD = 0.85
_MIN_CONTAINMENT_CHARS = 12   # a one-word reply must not count as an echo of a longer message

# What to say when the model echoed twice. A controlled sentence, never the user's own words; English is the
# fallback for a language this table does not cover.
_FALLBACK_REPLIES = {
    "english": "Sorry, I couldn't answer that. Could you say it another way?",
    "hindi": "माफ़ कीजिए, मैं इसका जवाब नहीं दे पाई।",
    "telugu": "క్షమించండి, దానికి నేను సమాధానం చెప్పలేకపోయాను.",
    "german": "Entschuldigung, darauf konnte ich keine Antwort geben.",
    "japanese": "すみません、うまく答えられませんでした。",
}


def fallback_reply(language: str | None) -> str:
    """The fixed sentence for a turn that could not be answered, in the turn's language when known."""
    return _FALLBACK_REPLIES.get((language or "").strip().lower(), _FALLBACK_REPLIES["english"])


def _normalise(text: str) -> str:
    out: list[str] = []
    for char in (text or "").casefold():
        if char.isspace():
            out.append(" ")
        elif unicodedata.category(char).startswith("P"):
            continue
        else:
            out.append(char)
    return " ".join("".join(out).split())


def is_echo(reply: str, user_message: str, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True when `reply` is essentially `user_message` (equal, contained, or >= `threshold` similar)."""
    replied, asked = _normalise(reply), _normalise(user_message)
    if not replied or not asked:
        return False
    if replied == asked:
        return True
    if (
        len(replied) >= _MIN_CONTAINMENT_CHARS
        and len(asked) >= _MIN_CONTAINMENT_CHARS
        and (replied in asked or asked in replied)
    ):
        return True
    return SequenceMatcher(None, replied, asked).ratio() >= threshold
