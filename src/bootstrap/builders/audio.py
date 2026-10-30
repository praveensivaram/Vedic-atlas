"""Mic capture, VAD, STT, playback, mute switch, indicator and wake-word builders."""

from __future__ import annotations

from pathlib import Path

from core.logging_config import logger
from schemas.config_schemas import AppConfig


def build_mute_switch(cfg: AppConfig):
    """Mute switch selected by privacy.mute_switch; software if it can't start.

    Boots muted unless privacy.start_muted is explicitly false. Any adapter
    that cannot be constructed or started falls back to SoftwareMuteSwitch —
    an input-device failure must never crash the server or leave it unable to
    mute. 'keyboard' reads the global hotkey from config/audio.yaml.
    """
    from tpa.hardware.software_mute_switch import SoftwareMuteSwitch

    start_muted = cfg.privacy.start_muted
    choice = (cfg.privacy.mute_switch or "software").lower()

    if choice == "software":
        return SoftwareMuteSwitch(start_muted=start_muted)

    try:
        if choice == "keyboard":
            from tpa.hardware.keyboard_mute_fallback import KeyboardMuteFallback
            switch = KeyboardMuteFallback(hotkey=cfg.audio.hotkey, start_muted=start_muted)
        elif choice == "gpio":
            from tpa.hardware.gpio_mute_switch import GpioMuteSwitch
            switch = GpioMuteSwitch(pin=cfg.privacy.mute_gpio_pin, muted_level=cfg.privacy.mute_gpio_muted_level)
        elif choice == "hid":
            raise RuntimeError("hid mute switch needs vendor_id/product_id (not configured)")
        else:
            logger.warning(f"[privacy] unknown mute_switch '{choice}'; using software")
            return SoftwareMuteSwitch(start_muted=start_muted)

        starter = getattr(switch, "start", None)
        if callable(starter):
            starter()
        logger.info(f"[privacy] mute switch '{choice}' active (source={switch.source_name!r})")
        return switch
    except Exception as e:
        logger.warning(f"[privacy] mute_switch '{choice}' unavailable ({e}); using software")
        return SoftwareMuteSwitch(start_muted=start_muted)


def build_indicator():
    """Try the BlinkStick USB LED; fall back to the software (log-only) indicator."""
    from tpa.hardware.laptop_indicator import BlinkStickIndicator, SoftwareIndicator
    stick = BlinkStickIndicator()
    try:
        stick.start()
        return stick
    except Exception as e:
        logger.info(f"[privacy] BlinkStick not available ({e}); using SoftwareIndicator")
        return SoftwareIndicator()


def build_audio_capture(cfg: AppConfig, frame_length: int | None = None):
    """Mic capture adapter, or None when `sounddevice` isn't installed.

    `frame_length` is dictated by the VAD frame size so every captured
    block is exactly one VAD frame — a mismatch silently makes the
    detector reject every frame.
    """
    try:
        from tpa.audio.sounddevice_capture import SAMPLE_RATE, SoundDeviceCapture
    except Exception as e:
        logger.info(f"[privacy] audio capture unavailable ({e}); capture gate will run mic-less")
        return None
    try:
        blocksize = frame_length or int(SAMPLE_RATE * cfg.audio.vad.frame_duration_ms / 1000)
        return SoundDeviceCapture(frame_length=blocksize, device_index=cfg.audio.mic_device_index)
    except Exception as e:
        logger.warning(f"[privacy] failed to build audio capture: {e}")
        return None


def build_vad(cfg: AppConfig):
    """VAD adapter chosen by config. Falls back to energy if webrtc is absent."""
    from tpa.audio.sounddevice_capture import SAMPLE_RATE

    v = cfg.audio.vad
    if v.engine == "webrtc":
        try:
            from tpa.audio.webrtc_vad import WebRtcVadDetector

            return WebRtcVadDetector(
                aggressiveness=v.aggressiveness,
                frame_duration_ms=v.frame_duration_ms,
                sample_rate=SAMPLE_RATE,
            )
        except Exception as e:
            logger.warning(f"[voice] webrtc VAD unavailable ({e}); falling back to energy VAD")

    try:
        from tpa.audio.energy_vad import EnergyVadDetector

        return EnergyVadDetector(
            frame_duration_ms=v.frame_duration_ms,
            sample_rate=SAMPLE_RATE,
            min_rms=v.min_rms,
            speech_ratio=v.speech_ratio,
            noise_adapt=v.noise_adapt,
        )
    except Exception as e:
        logger.info(f"[voice] VAD unavailable ({e}); continuous listening disabled")
        return None


def build_stt(cfg: AppConfig):
    """Local STT, or None. Never downloads — the model must be staged."""
    try:
        from core.constants import PROJECT_ROOT
        from tpa.stt.faster_whisper import FasterWhisperProvider

        backend = (cfg.audio.stt_backend or "faster_whisper").strip().lower()
        if backend not in ("faster_whisper", "whisper_cpp"):
            logger.warning(f"[voice] unknown stt_backend {backend!r}; using faster_whisper")
            backend = "faster_whisper"
        if backend == "whisper_cpp":
            # The adapter lands in the Pi phase; until then the laptop backend keeps voice working.
            logger.warning("[voice] stt_backend=whisper_cpp is not wired yet; using faster_whisper")
        provider_options = {
            "noise_suppression_enabled": cfg.audio.noise_suppression_enabled,
            "noise_suppression_strength": cfg.audio.noise_suppression_strength,
            "language": cfg.audio.stt_language,
            "threads": cfg.audio.stt_threads,
            "timeout": cfg.audio.stt_timeout,
            "auto_min_sec": cfg.audio.stt_auto_min_sec,
        }
        path = cfg.audio.stt_model_path
        if path:
            resolved = (PROJECT_ROOT / path) if not Path(path).is_absolute() else Path(path)
            if not resolved.exists():
                logger.warning(f"[voice] STT model not found at {resolved}; STT disabled")
                return None
            # faster-whisper accepts a directory in place of a size name.
            logger.info(
                f"[voice] STT backend={backend} model={resolved} language={cfg.audio.stt_language!r}"
            )
            return FasterWhisperProvider(model_size=str(resolved), **provider_options)
        logger.info(
            f"[voice] STT backend={backend} model={cfg.audio.stt_model!r} language={cfg.audio.stt_language!r}"
        )
        return FasterWhisperProvider(model_size=cfg.audio.stt_model, **provider_options)
    except Exception as e:
        logger.info(f"[voice] STT unavailable ({e})")
        return None


def build_speaker(cfg: AppConfig):
    try:
        from tpa.audio.playback import SpeakerPlayback

        return SpeakerPlayback(device_index=cfg.audio.speaker_device_index)
    except Exception as e:
        logger.info(f"[voice] speaker unavailable ({e})")
        return None


def build_wake_word(cfg: AppConfig):
    """Wake trigger selected by audio.wake_engine; None (always-transcribe) on any failure.

    'hotkey' arms a listening window on a global key (does not work headless —
    see docs/07-backlog.md). 'openwakeword' is spoken-wake via a local ONNX
    model, no API key, works headless — the default. 'porcupine' is
    Picovoice's spoken-wake, needs PICOVOICE_ACCESS_KEY + a .ppn keyword file;
    kept available, not installed/used by default (see
    docs/voice/open-source-wake-speaker-design.md). A failure never crashes
    the voice loop — it returns None and the loop transcribes while unmuted,
    exactly as before.
    """
    if not cfg.audio.wake_word_enabled:
        return None
    engine = (cfg.audio.wake_engine or "none").lower()
    if engine == "none":
        return None
    try:
        if engine == "hotkey":
            from tpa.wake_word.hotkey import HotkeyWakeWord
            wake = HotkeyWakeWord(hotkey=cfg.audio.wake_hotkey)
            wake.start()
            logger.info(f"[voice] wake engine 'hotkey' active ({cfg.audio.wake_hotkey})")
            return wake
        if engine == "openwakeword":
            model_path = cfg.audio.wake_oww_model_path
            if not model_path:
                logger.warning("[voice] openwakeword needs audio.wake_oww_model_path; wake disabled")
                return None
            from core.constants import PROJECT_ROOT
            from tpa.wake_word.openwakeword_engine import OpenWakeWordEngine
            resolved = model_path if Path(model_path).is_absolute() else PROJECT_ROOT / model_path
            wake = OpenWakeWordEngine(
                model_name=cfg.audio.wake_oww_model,
                model_path=str(resolved),
                threshold=cfg.audio.wake_threshold,
            )
            logger.info(
                "[voice] wake engine 'openwakeword' active "
                f"(model={cfg.audio.wake_oww_model!r}, threshold={cfg.audio.wake_threshold:.3f})"
            )
            return wake
        if engine == "porcupine":
            access_key = picovoice_access_key()
            if not access_key or not cfg.audio.wake_keyword_path:
                logger.warning("[voice] porcupine needs PICOVOICE_ACCESS_KEY + audio.wake_keyword_path; wake disabled")
                return None
            from core.constants import PROJECT_ROOT
            from tpa.wake_word.porcupine import PorcupineWakeWord
            kw = cfg.audio.wake_keyword_path
            resolved = kw if Path(kw).is_absolute() else PROJECT_ROOT / kw
            wake = PorcupineWakeWord(access_key=access_key, keyword_path=str(resolved))
            if getattr(wake, "sample_rate", 16000) not in (0, 16000):
                logger.warning("[voice] porcupine sample rate != 16kHz; wake disabled")
                return None
            starter = getattr(wake, "start", None)
            if callable(starter):
                starter()
            logger.info("[voice] wake engine 'porcupine' active")
            return wake
        logger.warning(f"[voice] unknown wake_engine '{engine}'; wake disabled")
        return None
    except Exception as e:
        logger.warning(f"[voice] wake engine '{engine}' unavailable ({e}); wake disabled")
        return None


def picovoice_access_key() -> str:
    """One key covers every Picovoice SDK (Porcupine, Eagle, ...). Prefer the
    generic name; fall back to the wake-word-specific one some installs set."""
    import os
    return os.environ.get("PICOVOICE_ACCESS_KEY") or os.environ.get("PORCUPINE_ACCESS_KEY", "")
