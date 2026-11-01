"""FasterWhisperProvider — implements STTPort via `faster-whisper`.

Donor: veda/voice/stt.py's WhisperSTT, read in full and adapted:
  - **Behaviour change, deliberate**: the donor's docstring says "First
    call downloads the model (~50-150MB depending on size)" — that's a
    silent runtime download, which violates the no-auto-download rule
    (REQ-M-02 spirit: nothing should reach the network to make the
    device work). This adapter requires the model to already be present
    in the local cache directory and raises clearly if it isn't, instead
    of silently fetching it.
  - Takes a domain AudioWindow (PCM bytes) instead of a bare numpy array,
    converting internally — callers never need to know this adapter
    wants float32 normalized samples.
  - The donor's separate silence/RMS utterance-detection helpers
    (`rms`, `SILENCE_RMS_THRESHOLD`, etc.) are VAD logic, not STT —
    they belong with the ambient-loop / VAD work, not this file.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

from core.logging_config import logger
from domain.value_objects.audio_window import AudioWindow
from domain.value_objects.transcript import Transcript
from tpa.stt.noise_suppression import SpectralNoiseSuppressor


def allow_unavailable_av() -> bool:
    """Let faster-whisper import even when PyAV's native files cannot be loaded. True when a stand-in was installed.

    faster-whisper imports PyAV (`av`) at import time, but only uses it to decode audio *files*. Veda passes a
    float32 array, so decoding never runs. On Windows with Smart App Control, one of PyAV's compiled files
    (av/sidedata/sidedata.pyd) can be blocked ("DLL load failed ... An Application Control policy has blocked
    this file"), and that import error would otherwise take the whole voice loop down on the first utterance.
    """
    try:
        import av  # noqa: F401
        return False
    except ImportError as exc:
        reason = str(exc)
    for name in [m for m in sys.modules if m == "av" or m.startswith("av.")]:
        del sys.modules[name]  # drop the half-imported package

    stub = types.ModuleType("av")

    def _unavailable(attr: str):
        raise AttributeError(
            f"PyAV is unavailable on this machine ({reason}), so audio files cannot be decoded (looked up av.{attr}). "
            "Veda only transcribes in-memory audio, which does not need it."
        )

    stub.__getattr__ = _unavailable  # type: ignore[attr-defined]  # PEP 562: only called for names the stub lacks
    sys.modules["av"] = stub
    logger.warning(f"[stt] PyAV cannot be loaded ({reason}); continuing without it (only needed for audio files)")
    return True


class FasterWhisperProvider:
    """Implements STTPort via a locally cached faster-whisper model."""

    def __init__(
        self,
        model_size: str = "base.en",
        cache_dir: str | Path | None = None,
        *,
        noise_suppression_enabled: bool = True,
        noise_suppression_strength: float = 0.75,
        language: str = "auto",
        threads: int | None = None,
        timeout: float = 120.0,
        auto_min_sec: float = 2.5,
    ):
        self._model_size = model_size
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._model = None
        self._threads = threads
        self._timeout = timeout
        self._auto_min_sec = auto_min_sec
        # Sticky hint: the last confidently detected language. Short clips misdetect, so below auto_min_sec
        # they reuse this instead of letting Whisper guess.
        self._last_language: str | None = None
        self.set_language(language, announce=False)
        self._noise_suppressor = (
            SpectralNoiseSuppressor(strength=noise_suppression_strength)
            if noise_suppression_enabled
            else None
        )
        logger.info(
            "[stt] noise suppression "
            f"{'enabled' if self._noise_suppressor is not None else 'disabled'} "
            f"strength={noise_suppression_strength:.2f} "
            f"language={self._language or 'auto'}"
        )

    @property
    def current_language(self) -> str:
        """The pinned language, or "auto" when detection/sticky is in charge."""
        return self._language or "auto"

    def set_language(self, language: str | None, *, announce: bool = True) -> None:
        """Pin a language code ("en", "hi", ...) or "auto" for detection + sticky."""
        value = (language or "auto").strip().lower()
        self._language = None if value in ("", "auto") else value
        if self._language is None and str(self._model_size).rstrip("/\\").lower().endswith(".en"):
            # An .en checkpoint cannot auto-detect; pin it instead of letting the library fail or translate.
            self._language = "en"
            logger.warning("[stt] English-only model with language=auto: pinning to en")
        if announce:
            logger.info(f"[stt] language set to {self.current_language!r}")

    def _language_for(self, duration_sec: float) -> str | None:
        """The language to pass to Whisper for this clip (None = let it detect)."""
        if self._language:
            return self._language
        if duration_sec < self._auto_min_sec and self._last_language:
            return self._last_language
        return None

    def _remember_language(self, detected: str | None, duration_sec: float, confidence: float) -> None:
        if self._language is not None or not detected:
            return
        if duration_sec >= self._auto_min_sec and confidence >= 0.6 and detected != self._last_language:
            self._last_language = detected
            logger.info(f"[stt] sticky language -> {detected!r}")

    def _ensure(self):
        if self._model is None:
            started = time.perf_counter()
            allow_unavailable_av()
            from faster_whisper import WhisperModel

            kwargs = {"device": "cpu", "compute_type": "int8"}
            if self._threads:
                kwargs["cpu_threads"] = self._threads
            if self._cache_dir:
                kwargs["download_root"] = self._cache_dir
                if not Path(self._cache_dir).exists():
                    raise FileNotFoundError(
                        f"whisper model cache not found at {self._cache_dir} — this build does "
                        "not auto-download models. Run the installer's model-fetch step first."
                    )
            try:
                self._model = WhisperModel(self._model_size, **kwargs)
            except Exception:
                logger.exception(f"[stt] failed to load Whisper model {self._model_size!r}")
                raise
            logger.info(
                "[stt][timing] stage=model_load "
                f"elapsed_ms={(time.perf_counter() - started) * 1000:.0f} "
                f"model={self._model_size!r}"
            )
        return self._model

    async def transcribe(self, audio: AudioWindow) -> Transcript:
        import numpy as np

        def _do():
            pcm = np.frombuffer(audio.pcm, dtype=np.int16).astype(np.float32) / 32768.0
            model = self._ensure()
            if self._noise_suppressor is not None:
                suppression_started = time.perf_counter()
                input_rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
                input_peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
                pcm = self._noise_suppressor.process(pcm)
                output_rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
                output_peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
                logger.info(
                    "[stt][timing] stage=noise_suppression "
                    f"elapsed_ms={(time.perf_counter() - suppression_started) * 1000:.0f} "
                    f"strength={self._noise_suppressor.strength:.2f} "
                    f"input_rms={input_rms:.5f} input_peak={input_peak:.5f} "
                    f"output_rms={output_rms:.5f} output_peak={output_peak:.5f}"
                )
            inference_started = time.perf_counter()
            segments, info = model.transcribe(
                pcm, language=self._language_for(audio.duration_sec), task="transcribe", vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                # Whisper invents text on noise ("subscribe to our channel", "Bye. Bye. Bye..."), and
                # conditioning on its own previous output makes it loop: turn that off and reject
                # low-confidence / repetitive output.
                condition_on_previous_text=False,
                no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
            )
            all_segments = list(segments)
            inference_ms = (time.perf_counter() - inference_started) * 1000
            # Whisper's own recommended rule: a segment that is probably silence AND low confidence is noise.
            kept = [
                seg for seg in all_segments
                if not (getattr(seg, "no_speech_prob", 0.0) > 0.6 and getattr(seg, "avg_logprob", 0.0) < -1.0)
            ]
            text = "".join(seg.text for seg in kept).strip()
            confidence = float(getattr(info, "language_probability", 1.0))
            # A sub-1.2 s clip that Whisper itself calls silence is noise, not a command ("Yo, yo, yo.").
            # The segment filter above needs BOTH no_speech and logprob to be bad; a short noisy clip often has
            # only the first, so it would otherwise become a turn.
            if text and audio.duration_sec < 1.2 and all_segments:
                no_speech = max(float(getattr(seg, "no_speech_prob", 0.0)) for seg in all_segments)
                if no_speech >= 0.6 and confidence < 0.7:
                    logger.info(
                        f"[stt] dropping short noise clip (audio_sec={audio.duration_sec:.2f} "
                        f"no_speech={no_speech:.2f} language_probability={confidence:.3f}) text={text!r}"
                    )
                    text = ""
            segment_quality = [
                (
                    max(0.0, float(getattr(seg, "end", 0.0)) - float(getattr(seg, "start", 0.0))),
                    float(getattr(seg, "avg_logprob", 0.0)),
                    float(getattr(seg, "no_speech_prob", 0.0)),
                )
                for seg in all_segments
            ]
            detected_language = getattr(info, "language", None) or self._language or "en"
            return text, confidence, segment_quality, inference_ms, detected_language

        import asyncio

        started = time.perf_counter()
        try:
            # CTranslate2's call runs in C++ and cannot be cancelled: wrapping it in asyncio.wait_for abandons
            # the worker thread, which then keeps burning CPU while TTS/playback starve (observed: a 120 s
            # timeout was followed by 24 s TTS and 49 s playback for 2 s of audio). Wait for it instead; the
            # timeout below is a warning, not a cancellation. A killable worker process is the proper fix.
            text, confidence, segment_quality, inference_ms, detected_language = await asyncio.to_thread(_do)
        except Exception:
            logger.exception("[stt] Whisper transcription failed")
            raise
        self._remember_language(detected_language, audio.duration_sec, float(confidence))
        elapsed_ms = (time.perf_counter() - started) * 1000
        if self._timeout and elapsed_ms > self._timeout * 1000:
            logger.warning(
                f"[stt] transcription took {elapsed_ms / 1000:.0f}s (soft limit {self._timeout:.0f}s); "
                "the machine is CPU/ memory starved"
            )
        rtf = elapsed_ms / max(audio.duration_sec * 1000, 1)
        logger.info(
            "[stt][timing] stage=whisper_model_inference "
            f"elapsed_ms={inference_ms:.0f} audio_sec={audio.duration_sec:.2f} "
            f"rtf={inference_ms / max(audio.duration_sec * 1000, 1):.2f} "
            f"segments={len(segment_quality)} "
            f"language={detected_language} language_probability={float(confidence):.3f}"
        )
        logger.info(
            "[stt][timing] stage=stt_preprocess_and_total "
            f"elapsed_ms={elapsed_ms:.0f} audio_sec={audio.duration_sec:.2f} rtf={rtf:.2f}"
        )
        if segment_quality:
            quality = " ".join(
                f"segment{index + 1}(sec={duration:.2f},logprob={logprob:.2f},no_speech={no_speech:.2f})"
                for index, (duration, logprob, no_speech) in enumerate(segment_quality)
            )
            logger.info(f"[stt] segment diagnostics: {quality}")
        return Transcript(
            text=text, confidence=float(confidence), language=detected_language, duration_sec=audio.duration_sec
        )