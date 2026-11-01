"""The language policy: deterministic detection, model labels, and the text guard."""

import pytest

from domain.policies.language_policy import language_of, looks_like_text, normalise_language

HINDI = "\u092e\u0941\u091d\u0947 \u0915\u093f\u0924\u093e\u092c\u0947\u0902 \u092a\u0922\u093c\u0928\u093e \u092a\u0938\u0902\u0926 \u0939\u0948"
TELUGU = "\u0c28\u0c2e\u0c38\u0c4d\u0c24\u0c47, \u0c2e\u0c40\u0c30\u0c41 \u0c0e\u0c32\u0c3e \u0c09\u0c28\u0c4d\u0c28\u0c3e\u0c30\u0c41?"


@pytest.mark.parametrize(
    "text,expected",
    [
        (HINDI, "Hindi"),
        (TELUGU, "Telugu"),
        ("Ich mag Jazzmusik am Abend", "German"),
        ("Was ist die Hauptstadt von Deutschland?", "German"),
        ("Guten Tag, wie geht es dir", "German"),
        ("I like jazz in the evening", None),
        ("the weather is nice and I am happy", None),
        ("2+2", None),
        ("ok", None),
        ("", None),
    ],
)
def test_language_of_is_exact_for_scripts_and_conservative_for_latin(text, expected):
    assert language_of(text) == expected


@pytest.mark.parametrize(
    "label,expected",
    [
        ("German", "German"),
        (" german ", "German"),
        ('"Japanese"', "Japanese"),
        ("Brazilian Portuguese", "Brazilian Portuguese"),
        ("unknown", None),
        ("UNKNOWN", None),
        ("none", None),
        ("", None),
        (None, None),
        ("x" * 40, None),
        ("Deutsch!", None),
        ("123", None),
    ],
)
def test_normalise_language_accepts_short_names_and_rejects_stray_values(label, expected):
    assert normalise_language(label) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Klingon gibberish sentence here", True),
        (HINDI, True),
        ("2+2", False),
        ("ok", False),
        ("", False),
    ],
)
def test_looks_like_text_guards_the_did_not_understand_reply(text, expected):
    assert looks_like_text(text, min_letters=8) is expected
