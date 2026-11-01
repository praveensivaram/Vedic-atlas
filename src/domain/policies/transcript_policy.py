"""TranscriptPolicy — is this "transcript" real speech, or a speech-to-text artefact?

Whisper-family models invent text on noise and near-silence: stock YouTube
phrases ("please subscribe to our channel", "thanks for watching"), a word
repeated dozens of times ("Bye. Bye. Bye. ..."), or a phrase looping until
the length cap. Passed on as a user turn, these make the assistant answer
things nobody said. This policy recognises them so the voice loop can drop
the turn. Pure: no I/O, no model.

It is deliberately conservative about *what counts as an artefact* — only
shapes real speech essentially never has — so a genuine short command is
never discarded.
"""

from __future__ import annotations

import re
from collections import Counter

# Any script: an ASCII-only word regex made every Devanagari/Telugu/Japanese transcript
# look like "symbols only" and the loop dropped it as an artefact.
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_SOUND_TAG_RE = re.compile(r"\s*[\[\(\*].{0,40}[\]\)\*]\s*")  # "[music]", "(applause)", "*sigh*"

# Stock phrases Whisper emits for noise/silence (training-data artefacts).
_HALLUCINATED_PHRASES = (
    "thanks for watching", "thank you for watching", "thank you so much for watching",
    "subscribe to our channel", "subscribe to my channel", "please subscribe", "like and subscribe",
    "see you in the next video", "see you next week", "see you next time",
    "subtitles by", "subtitles made by", "amara.org", "transcription by", "transcribed by", "captions by",
    "for more information visit", "visit www", "we'll see you next",
)

# Filler Whisper emits for a short burst of noise; only dropped when the clip is short.
_SHORT_FILLERS = frozenset({"you", "the", "uh", "um", "hmm", "mm", "bye", "thanks", "thank you", "so", "oh"})
_SHORT_CLIP_SEC = 1.2

_MIN_WORDS_FOR_REPETITION = 5
_SINGLE_WORD_SHARE = 0.6
_NGRAM_REPEAT_LIMITS = {2: 4, 3: 3, 4: 3}  # n-gram size -> occurrences that make it a loop


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def artifact_reason(text: str, duration_sec: float = 0.0) -> str | None:
    """Why `text` is probably not real speech, or None if it looks genuine."""
    raw = (text or "").strip()
    if not raw:
        return "empty"
    if not any(char.isalpha() for char in raw):
        return "no letters (symbols or music notes only)"
    if _SOUND_TAG_RE.fullmatch(raw):
        return "sound tag"

    norm = _normalize(raw)
    for phrase in _HALLUCINATED_PHRASES:
        if phrase in norm or phrase in raw.lower():
            return f"stock hallucination ({phrase!r})"

    words = norm.split()
    if len(words) >= _MIN_WORDS_FOR_REPETITION:
        top_word, top_count = Counter(words).most_common(1)[0]
        if top_count / len(words) >= _SINGLE_WORD_SHARE:
            return f"one word repeated ({top_word!r} x{top_count})"
        for n, limit in _NGRAM_REPEAT_LIMITS.items():
            if len(words) < n * limit:
                continue
            grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
            gram, count = grams.most_common(1)[0]
            if count >= limit and len(set(gram)) > 1:
                return f"phrase loop ({' '.join(gram)!r} x{count})"

    if duration_sec and duration_sec < _SHORT_CLIP_SEC and norm in _SHORT_FILLERS:
        return f"short noise burst heard as {norm!r}"
    return None
