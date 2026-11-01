"""Per-language Piper voices: selection, fallback, and builder wiring."""

from pathlib import Path

from bootstrap.builders.tts import build_tts
from schemas.config_schemas import AppConfig
from tpa.tts.piper import PiperProvider, language_key


def test_language_names_and_codes_normalise():
    assert language_key("Hindi") == "hi"
    assert language_key("hi") == "hi"
    assert language_key("Telugu") == "te"
    assert language_key("German") == "de"
    assert language_key("Japanese") == "ja"
    assert language_key("Kannada") == "kn"
    assert language_key("English") == "en"
    assert language_key(None) == "en"


def test_voice_selection_falls_back_to_the_default(tmp_path):
    en = tmp_path / "en.onnx"
    hi = tmp_path / "hi.onnx"
    en.write_bytes(b"")
    hi.write_bytes(b"")
    provider = PiperProvider(model_path=en, voices={"en": str(en), "hi": str(hi)})
    assert provider.path_for("Hindi") == hi
    assert provider.path_for("hi") == hi
    assert provider.path_for("Kannada") == en      # no kn voice -> default
    assert provider.path_for(None) == en


def test_builder_wires_the_map_and_skips_missing_voices(tmp_path):
    en = tmp_path / "en.onnx"
    hi = tmp_path / "hi.onnx"
    en.write_bytes(b"")
    hi.write_bytes(b"")
    cfg = AppConfig()
    cfg.audio.tts_engine = "piper"
    cfg.audio.tts_model_path = str(en)
    cfg.audio.tts_voice_map = {"en": str(en), "hi": str(hi), "kn": str(tmp_path / "missing.onnx")}

    provider = build_tts(cfg)

    assert isinstance(provider, PiperProvider)
    assert provider.path_for("Hindi") == hi
    assert provider.path_for("Kannada") == en
