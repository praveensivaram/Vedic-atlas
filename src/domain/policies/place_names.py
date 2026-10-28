"""Place/name normalisation shared by the control argument policy and the geocoder.

Two steps, in order:
  1. a spelling fix for the names Whisper reliably mangles (Hindi/Telugu -> the English exonym);
  2. a general script-to-Latin transliteration (`unidecode`) for any other non-Latin name.
No city list is required for step 2; the alias map only corrects spellings the microphone produces.
"""

from __future__ import annotations

_PLACE_ALIASES = {
    "\u0939\u0948\u0926\u0930\u092c\u093e\u0926": "Hyderabad",
    "\u0939\u0948\u0926\u0930\u093e\u092c\u093e\u0926": "Hyderabad",
    "\u0c39\u0c48\u0c26\u0c30\u0c3e\u0c2c\u0c3e\u0c26\u0c4d": "Hyderabad",
    "\u0c39\u0c48\u0c26\u0c30\u0c3e\u0c2c\u0c3e\u0c26": "Hyderabad",
    "\u092e\u0941\u0902\u092c\u0908": "Mumbai",
    "\u092e\u0941\u092e\u094d\u092c\u0908": "Mumbai",
    "\u0c2e\u0c41\u0c02\u0c2c\u0c48": "Mumbai",
    "\u0926\u093f\u0932\u094d\u0932\u0940": "Delhi",
    "\u0928\u0908 \u0926\u093f\u0932\u094d\u0932\u0940": "New Delhi",
    "\u0c22\u0c3f\u0c32\u0c4d\u0c32\u0c40": "Delhi",
    "\u0c28\u0c4d\u0c2f\u0c42\u0c22\u0c3f\u0c32\u0c4d\u0c32\u0c40": "New Delhi",
    "\u091a\u0947\u0928\u094d\u0928\u0908": "Chennai",
    "\u0c1a\u0c46\u0c28\u0c4d\u0c28\u0c48": "Chennai",
    "\u092c\u0947\u0902\u0917\u0932\u0941\u0930\u0941": "Bengaluru",
    "\u092c\u0902\u0917\u0932\u094c\u0930": "Bengaluru",
    "\u0c2c\u0c46\u0c02\u0c17\u0c33\u0c42\u0c30\u0c41": "Bengaluru",
    "\u0915\u094b\u0932\u0915\u093e\u0924\u093e": "Kolkata",
    "\u0c15\u0c4b\u0c32\u0c4d\u0c15\u0c24\u0c3e": "Kolkata",
    "\u092a\u0941\u0923\u0947": "Pune",
    "\u0c2a\u0c41\u0c23\u0c46": "Pune",
    "\u091c\u092f\u092a\u0941\u0930": "Jaipur",
    "\u092c\u0940\u091c\u093f\u0902\u0917": "Beijing",
    "\u0c2c\u0c40\u0c1c\u0c3f\u0c02\u0c17\u0c4d": "Beijing",
    "\u091f\u094b\u0915\u094d\u092f\u094b": "Tokyo",
    "\u0c1f\u0c4b\u0c15\u0c4d\u0c2f\u0c4b": "Tokyo",
}


def normalize_place_name(name: str | None) -> str:
    """The English exonym for a known spelling, or the input unchanged."""
    text = " ".join((name or "").split())
    return _PLACE_ALIASES.get(text, text)


def transliterate_to_latin(name: str) -> str:
    """Phonetic Latin form of a non-Latin string; unchanged when no transliterator is installed."""
    try:
        from unidecode import unidecode
    except ImportError:
        return name
    return " ".join(unidecode(name).split())


def to_latin(name: str | None) -> str:
    """Alias first, then transliteration; Latin input is returned unchanged."""
    text = " ".join((name or "").split())
    known = normalize_place_name(text)
    if known != text:
        return known
    if any(ord(char) > 127 for char in text):
        return transliterate_to_latin(text) or text
    return text
