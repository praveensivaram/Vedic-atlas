"""LanguagePolicy — what language a user message is in.

Two sources, in this order:

  1. `language_of()` — deterministic, free, and exact for the scripts it knows
     (Telugu, Devanagari/Hindi) plus a two-word German heuristic for the one
     Latin-script language this device was validated on. It overrides the model
     so the three tested languages can never be mislabelled.
  2. `normalise_language()` — the label the control model wrote for its own
     turn (`ControlDecision.language`). This is what makes the feature open-
     ended: the model names any language it can read, not just a fixed list.

`looks_like_text()` guards the "I didn't understand" reply so it can only fire
for something that really reads as language (not "2+2" or "ok").

Pure text logic, no I/O, no dependency on an LLM or a language-detection model.
"""

from __future__ import annotations

import re

_TELUGU = range(0x0C00, 0x0C80)
_DEVANAGARI = range(0x0900, 0x0980)
_GERMAN_MARKS = frozenset("äöüßÄÖÜ")
_GERMAN_WORDS = frozenset({
    "ich", "du", "nicht", "und", "ist", "heute", "abend", "bitte", "danke", "sehr",
    "auch", "mein", "dein", "wir", "für", "aber", "oder", "mag", "möchte", "wie",
    "was", "wo", "wann", "guten", "tag", "dir", "mir", "gerne", "schön", "wetter",
})
_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_LABEL_RE = re.compile(r"[A-Za-z][A-Za-z .'()-]{0,23}")
_UNKNOWN_LABELS = frozenset({"", "unknown", "none", "n/a", "other", "undetermined", "cannot tell"})
MAX_LANGUAGE_LABEL = 24


def language_of(text: str) -> str | None:
    """The deterministic answer for `text`, or None when this policy cannot name it."""
    if not text:
        return None
    has_devanagari = False
    for char in text:
        code = ord(char)
        if code in _TELUGU:
            return "Telugu"
        if code in _DEVANAGARI:
            has_devanagari = True
    if has_devanagari:
        return "Hindi"
    if any(char in _GERMAN_MARKS for char in text):
        return "German"
    words = {word.lower() for word in _WORD_RE.findall(text)}
    if len(words & _GERMAN_WORDS) >= 2:
        return "German"
    return None


def normalise_language(label: str | None) -> str | None:
    """The model's label as a clean language name, or None when it is not one.

    The control model is asked for an English language name ("German", "Telugu",
    "Japanese") or "unknown". Anything that is not a short alphabetic name is
    treated as unknown so a stray token can never be pasted into a prompt.
    """
    if not label:
        return None
    text = " ".join(str(label).strip().strip('".').split())
    if text.lower() in _UNKNOWN_LABELS:
        return None
    if len(text) > MAX_LANGUAGE_LABEL or not _LABEL_RE.fullmatch(text):
        return None
    return text[:1].upper() + text[1:]


def looks_like_text(text: str, min_letters: int = 8) -> bool:
    """True when the message is plausibly natural language (not maths, a number, or "ok").

    Used only to decide whether an unidentifiable message deserves "I didn't
    understand that": two alphabetic words and at least `min_letters` letters.
    Works for any script via `str.isalpha()`.
    """
    words = [word for word in (text or "").split() if any(char.isalpha() for char in word)]
    letters = sum(1 for char in text or "" if char.isalpha())
    return len(words) >= 2 and letters >= min_letters
