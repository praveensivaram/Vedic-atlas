"""Multilingual STT configuration and language resolution (no model needed)."""

from bootstrap.builders.audio import build_stt
from schemas.config_schemas import AppConfig, AudioConfig
from tpa.stt.faster_whisper import FasterWhisperProvider


def test_auto_means_detect_for_a_multilingual_model():
    provider = FasterWhisperProvider(model_size="small", language="auto")
    assert provider._language is None


def test_a_language_code_is_kept():
    assert FasterWhisperProvider(model_size="small", language="hi")._language == "hi"
    assert FasterWhisperProvider(model_size="small", language=" te ")._language == "te"


def test_an_english_only_model_is_pinned_to_english_when_auto_is_requested():
    assert FasterWhisperProvider(model_size="base.en", language="auto")._language == "en"
    assert FasterWhisperProvider(model_size="data/models/whisper-base.en", language="auto")._language == "en"


def test_audio_config_defaults_are_multilingual_ready():
    config = AudioConfig()
    assert config.stt_backend == "faster_whisper"
    assert config.stt_language == "auto"
    assert config.stt_auto_min_sec == 2.5
    assert config.stt_timeout == 120.0


def test_build_stt_falls_back_while_the_pi_backend_is_unwired(tmp_path):
    cfg = AppConfig()
    cfg.audio.stt_backend = "whisper_cpp"
    cfg.audio.stt_model_path = str(tmp_path)
    cfg.audio.stt_language = "auto"
    provider = build_stt(cfg)
    assert isinstance(provider, FasterWhisperProvider)
    assert provider._language is None


def test_auto_reuses_the_last_language_for_short_clips_only():
    provider = FasterWhisperProvider(model_size="small", language="auto")
    assert provider._language_for(1.0) is None            # nothing learned yet: let Whisper detect
    provider._last_language = "hi"
    assert provider._language_for(1.0) == "hi"            # short clip: sticky
    assert provider._language_for(4.0) is None            # long clip: detect


def test_set_language_pins_and_returns_to_auto():
    provider = FasterWhisperProvider(model_size="small", language="auto")
    provider.set_language("en")
    assert provider.current_language == "en"
    assert provider._language_for(1.0) == "en"
    provider.set_language("auto")
    assert provider.current_language == "auto"


def test_only_long_confident_clips_update_the_sticky_language():
    provider = FasterWhisperProvider(model_size="small", language="auto")
    provider._remember_language("hi", 1.0, 0.9)
    assert provider._last_language is None
    provider._remember_language("hi", 4.0, 0.9)
    assert provider._last_language == "hi"


def test_an_english_only_model_is_still_pinned_to_english():
    provider = FasterWhisperProvider(model_size="base.en", language="auto")
    assert provider.current_language == "en"
