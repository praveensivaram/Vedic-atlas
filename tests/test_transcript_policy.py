"""The transcript filter must work in every script, not just ASCII."""

from domain.policies.transcript_policy import artifact_reason

HINDI = "\u0928\u092e\u0938\u094d\u0924\u0947, \u0915\u0948\u0938\u0947 \u0939\u0948\u0902 \u0906\u092a?"
TELUGU = "\u0c28\u0c2e\u0c38\u0c4d\u0c24\u0c47"
JAPANESE = "\u3053\u3093\u306b\u3061\u306f"
HINDI_REPEAT = "\u0928\u092e\u0938\u094d\u0924\u0947 " * 5


def test_non_latin_scripts_are_real_speech_not_artefacts():
    assert artifact_reason(HINDI, duration_sec=2.5) is None
    assert artifact_reason(TELUGU, duration_sec=2.5) is None
    assert artifact_reason(JAPANESE, duration_sec=2.5) is None


def test_ascii_still_works():
    assert artifact_reason("what is the weather today", duration_sec=2.5) is None


def test_symbols_only_is_still_rejected():
    assert artifact_reason("...", duration_sec=2.5) == "no letters (symbols or music notes only)"
    assert artifact_reason("1234 !!!", duration_sec=2.5) == "no letters (symbols or music notes only)"


def test_english_stock_phrases_are_still_rejected():
    assert artifact_reason("thanks for watching", duration_sec=2.5) is not None


def test_a_repeated_word_in_hindi_is_caught():
    assert artifact_reason(HINDI_REPEAT, duration_sec=2.5) is not None
