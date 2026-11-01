"""Latin-argument contract: manifest flag, normalisation, and transliteration."""

from domain.entities.control_decision import ControlDecision
from domain.policies.argument_policy import normalize_latin_arguments
from domain.policies.place_names import normalize_place_name, to_latin
from tpa.filestore.yaml_tool_manifest_store import YamlToolManifestStore

MANIFESTS = {m.name: m for m in YamlToolManifestStore().load_all()}

HINDI_HYDERABAD = "\u0939\u0948\u0926\u0930\u093e\u092c\u093e\u0926"
HINDI_MISHEARD = "\u0939\u0948\u0926\u0930\u092c\u093e\u0926"   # Whisper dropped the long 'aa'
HINDI_BHOPAL = "\u092d\u094b\u092a\u093e\u0932"
HINDI_TITLE = "\u0926\u0942\u0927 \u0916\u0930\u0940\u0926\u0928\u093e"


def _decision(tool: str, args: dict) -> ControlDecision:
    return ControlDecision(calls=({"tool": tool, "args": args},), needs_live_data=True)


def test_weather_place_is_declared_latin():
    place = next(p for p in MANIFESTS["get_weather"].params if p.name == "place")
    assert place.format == "latin"


def test_native_place_is_normalised_before_the_tool_runs():
    out, notes = normalize_latin_arguments(_decision("get_weather", {"place": HINDI_HYDERABAD}), MANIFESTS)
    assert out.calls[0]["args"]["place"] == "Hyderabad"
    assert notes


def test_the_misheard_spelling_is_corrected():
    out, _ = normalize_latin_arguments(_decision("get_weather", {"place": HINDI_MISHEARD}), MANIFESTS)
    assert out.calls[0]["args"]["place"] == "Hyderabad"


def test_an_unknown_native_place_is_transliterated_to_latin():
    out, notes = normalize_latin_arguments(_decision("get_weather", {"place": HINDI_BHOPAL}), MANIFESTS)
    latin = out.calls[0]["args"]["place"]
    assert notes and all(ord(char) < 128 for char in latin)


def test_a_non_latin_field_without_the_flag_is_untouched():
    out, notes = normalize_latin_arguments(
        _decision("tasks", {"action": "add", "title": HINDI_TITLE}), MANIFESTS
    )
    assert out.calls[0]["args"]["title"] == HINDI_TITLE
    assert notes == []


def test_latin_input_is_left_alone():
    out, notes = normalize_latin_arguments(_decision("get_weather", {"place": "Mumbai"}), MANIFESTS)
    assert out.calls[0]["args"]["place"] == "Mumbai"
    assert notes == []


def test_place_names_helpers():
    assert normalize_place_name(HINDI_HYDERABAD) == "Hyderabad"
    assert to_latin("Mumbai") == "Mumbai"
