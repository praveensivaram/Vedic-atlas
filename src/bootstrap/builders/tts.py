"""Text-to-speech engine selection."""

from __future__ import annotations

import platform
from pathlib import Path

from core.logging_config import logger
from schemas.config_schemas import AppConfig

_IS_WINDOWS = platform.system() == "Windows"


def resolve_tts_engine(cfg: AppConfig) -> str:
    """`auto` = Windows speech (pyttsx3) on Windows, Piper everywhere else."""
    engine = (cfg.audio.tts_engine or "auto").strip().lower()
    if engine == "auto":
        return "pyttsx3" if _IS_WINDOWS else "piper"
    return engine


def build_tts(cfg: AppConfig):
    """TTS adapter chosen by config, or None.

    Every requirement is checked HERE, at boot, so a missing voice or package is a clear
    startup error ("voice not started - missing: tts") and not a silent device that logs
    "TTS failed" on every reply.
    """
    engine = resolve_tts_engine(cfg)
    try:
        if engine == "piper":
            model = cfg.audio.tts_model_path
            if not model:
                if _IS_WINDOWS:
                    logger.warning("[voice] tts_engine=piper has no tts_model_path; using Windows speech (pyttsx3) instead")
                    engine = "pyttsx3"
                else:
                    raise ValueError(
                        "Piper needs a voice: copy a Piper voice (.onnx and its .onnx.json) to the device and set "
                        "audio.tts_model_path in config/audio.yaml, e.g. data/models/piper/en_US-lessac-medium.onnx"
                    )
            if engine == "piper":
                import piper  # noqa: F401  (fail now, not on the first reply)
                from core.constants import PROJECT_ROOT
                from tpa.tts.piper import PiperProvider

                path = Path(model)
                resolved = path if path.is_absolute() else PROJECT_ROOT / path
                voices: dict[str, str] = {}
                for language, raw in (cfg.audio.tts_voice_map or {}).items():
                    voice_path = Path(raw)
                    resolved_voice = voice_path if voice_path.is_absolute() else PROJECT_ROOT / voice_path
                    if not resolved_voice.exists():
                        logger.warning(
                            f"[voice] TTS voice for {language!r} missing at {resolved_voice}; "
                            "that language will use the default voice"
                        )
                        continue
                    voices[language] = str(resolved_voice)
                logger.info(
                    f"[voice] TTS engine: piper (default={resolved.name}"
                    + (f", languages={sorted(voices)}" if voices else "")
                    + ")"
                )
                return PiperProvider(model_path=resolved, voices=voices)
        if engine == "pyttsx3":
            import pyttsx3  # noqa: F401  (Windows speech; not installed on Linux/Pi)
            from tpa.tts.pyttsx3_provider import Pyttsx3Provider

            logger.info("[voice] TTS engine: pyttsx3 (Windows speech)")
            return Pyttsx3Provider(voice_hint=cfg.audio.tts_voice)
        raise ValueError(f"unknown audio.tts_engine {cfg.audio.tts_engine!r} (use auto | piper | pyttsx3)")
    except ImportError as e:
        hint = (
            "pyttsx3 is Windows-only: set audio.tts_engine: piper (or auto) and give it a voice"
            if engine == "pyttsx3" else "install it with: pip install piper-tts"
        )
        logger.error(f"[voice] TTS unavailable: {e}. {hint}. Voice replies are disabled until this is fixed.")
        return None
    except Exception as e:
        logger.error(f"[voice] TTS unavailable: {e}. Voice replies are disabled until this is fixed.")
        return None
