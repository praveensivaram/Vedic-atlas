"""PiperProvider ? implements TTSPort via the local Piper neural TTS engine.

New ? no donor equivalent. Piper is the offline default: a fully local neural voice with no network call.
Requires pre-downloaded `.onnx` voices (no auto-download, consistent with every other model-loading adapter).

Multiple voices: `voices` maps a language (code or English name) to its `.onnx` path. `synthesize`'s `voice`
argument is the turn's language; unknown languages fall back to the default voice. Voices are loaded lazily and
cached (Piper voices are small; the voice session serialises synthesis).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncIterator

from core.logging_config import logger

# The reply language reaches us as an English name from the control decode ("Hindi") or a code from config.
_LANGUAGE_ALIASES = {
    "english": "en", "german": "de", "deutsch": "de", "hindi": "hi", "telugu": "te",
    "japanese": "ja", "kannada": "kn",
}


def language_key(value: str | None) -> str:
    """'Hindi' / 'hi' / 'HI' -> 'hi'; unknown words keep their first two letters."""
    text = (value or "en").strip().lower()
    if text in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[text]
    return text[:2] if len(text) >= 2 else "en"


class PiperProvider:
    """Implements TTSPort via locally installed Piper voice models."""

    def __init__(self, model_path: str | Path, sample_rate: int = 22050, voices: dict[str, str] | None = None):
        default = Path(model_path)
        if not default.exists():
            raise FileNotFoundError(
                f"Piper voice model not found at {default} ? this build does not auto-download "
                "voice models. Run the installer's model-fetch step first."
            )
        self._default_path = default
        self._voice_paths = {language_key(k): Path(v) for k, v in (voices or {}).items()}
        self._voices: dict[str, object] = {}
        self._sample_rate = self._rate_from_voice_config(default) or sample_rate

    @staticmethod
    def _rate_from_voice_config(model_path: Path) -> int | None:
        """A Piper voice ships `<voice>.onnx.json` with its sample rate (16000 for "low", 22050 for "medium")."""
        config = model_path.with_name(model_path.name + ".json")
        try:
            return int(json.loads(config.read_text(encoding="utf-8"))["audio"]["sample_rate"])
        except Exception:
            return None

    def path_for(self, language: str | None) -> Path:
        """The voice path for `language`, falling back to the default voice."""
        return self._voice_paths.get(language_key(language), self._default_path)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _ensure(self, path: Path):
        key = str(path)
        if key not in self._voices:
            from piper import PiperVoice

            logger.info(f"[tts] loading Piper voice {path.name}")
            self._voices[key] = PiperVoice.load(str(path))
        return self._voices[key]

    def warmup(self) -> None:
        """Load the voice and run it once, so the first reply does not wait for it (6 s on the Pi, measured)."""
        for _ in self._ensure().synthesize("Ready."):
            pass

    async def synthesize(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        import asyncio

        path = self.path_for(voice)
        voice_model = await asyncio.to_thread(self._ensure, path)
        rate = getattr(getattr(voice_model, "config", None), "sample_rate", None)
        self._sample_rate = int(rate or self._rate_from_voice_config(path) or self._sample_rate)

        def _synth_chunks() -> list[bytes]:
            return [chunk.audio_int16_bytes for chunk in voice_model.synthesize(text)]

        for chunk in await asyncio.to_thread(_synth_chunks):
            yield chunk
