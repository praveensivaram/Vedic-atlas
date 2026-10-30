"""VoiceSession — the always-on listen -> think -> speak loop.

★ New, PS-mandatory (REQ-M-01). This is the orchestrator the migration
never built: every adapter it needs already existed in tpa/, but nothing
connected them. It owns no I/O itself — every capability arrives as a
Port, so the Pi profile swaps adapters (whisper.cpp, Piper, GPIO mute)
with zero changes here.

    mic ──frames──> CaptureGate (muted?) ──> VAD ──> UtteranceCollector
      └──> STT ──> SupervisorAgent ──> TTS ──> speaker

Three properties that matter more than the happy path:

  * **Mute is checked every single frame**, and a mute mid-sentence
    discards the in-progress utterance rather than transcribing it. The
    gate is consulted, never cached — see service/privacy/capture_gate.py.

  * **Echo suppression.** While Veda speaks, its own voice reaches the
    mic. Capture is drained and ignored for the duration of playback,
    otherwise the assistant transcribes itself and talks in a loop. This
    is the single most common way an always-on voice demo fails.

  * **`from_voice=True`** flows into AgentContext so the responder keeps
    replies to one or two sentences — a paragraph that reads fine on
    screen is unbearable spoken aloud.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from datetime import datetime
from typing import Any, Callable

from core.logging_config import logger, log_text, reset_voice_turn_id, set_voice_turn_id
from domain.entities.agent_context import AgentContext
from domain.entities.utterance import Utterance
from domain.policies.transcript_policy import artifact_reason
from domain.ports.audio_capture_port import AudioCapturePort
from domain.ports.stt_port import STTPort
from domain.ports.tts_port import TTSPort
from service.voice.utterance_collector import UtteranceCollector

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s)")
_WAKE_ACKNOWLEDGEMENT = "Hey there, what's up?"
_PROCESSING_ACKNOWLEDGEMENT = "Working on that."
_PROCESSING_ACK_DELAY_SEC = 1.0


def _format_request_duration(elapsed_ms: float) -> str:
    if elapsed_ms < 60_000:
        return f"{elapsed_ms:.0f} ms"
    if elapsed_ms < 3_600_000:
        return f"{elapsed_ms / 60_000:.2f} mins"
    return f"{elapsed_ms / 3_600_000:.2f} hr"


class VoiceSession:
    """Continuous listen/think/speak loop, gated by the mute switch."""

    def __init__(
        self,
        audio: AudioCapturePort,
        collector: UtteranceCollector,
        stt: STTPort,
        tts: TTSPort,
        speaker: Any,
        supervisor: Any,
        capture_gate: Any,
        *,
        min_chars: int = 2,
        speak_replies: bool = True,
        speak_timeout_sec: float = 60.0,
        post_speak_settle_sec: float = 0.4,
        echo_guard: bool = True,
        wake_word: Any | None = None,
        wake_engine: str = "none",
        wake_window_sec: float = 8.0,
        speaker_id: Any | None = None,
        require_known_speaker: bool = False,
        speaker_match_threshold: float = 0.6,
        on_event: Callable[[dict], None] | None = None,
    ):
        self._audio = audio
        self._collector = collector
        self._stt = stt
        self._tts = tts
        self._speaker = speaker
        self._supervisor = supervisor
        self._gate = capture_gate
        self._min_chars = min_chars
        self._speak_replies = speak_replies
        self._speak_timeout_sec = speak_timeout_sec
        self._post_speak_settle_sec = post_speak_settle_sec
        self._echo_guard = echo_guard
        self._on_event = on_event or (lambda _e: None)

        # Wake-word gating. When _wake is None the loop transcribes while
        # unmuted (original behaviour); otherwise it stays PASSIVE until the
        # trigger arms a listening window.
        self._wake = wake_word
        self._wake_engine = wake_engine
        self._wake_window_sec = wake_window_sec
        self._wake_frame_bytes = (getattr(wake_word, "frame_length", 0) or 0) * 2
        self._wake_buffer = b""
        self._armed_until = 0.0
        self._armed_since: float | None = None
        self._wake_score_peak: float | None = None
        self._wake_audio_square_sum = 0.0
        self._wake_audio_sample_count = 0
        self._wake_audio_peak = 0
        self._last_wake_score_log_at = time.monotonic()

        # Multi-speaker recognition. Scored continuously, frame by frame,
        # while armed/collecting (same slicing idiom as the wake buffer
        # above) - but the accept/reject decision is made once per
        # utterance, off the accumulated mean score, not frame by frame.
        # A single noisy frame scoring low must not clip real speech from
        # an enrolled speaker.
        self._speaker_id = speaker_id
        self._require_known_speaker = require_known_speaker
        self._speaker_match_threshold = speaker_match_threshold
        self._speaker_frame_bytes = (getattr(speaker_id, "frame_length", 0) or 0) * 2
        self._speaker_buffer = b""
        self._speaker_score_sums: list[float] = []
        self._speaker_score_count = 0

        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._speech_lock = asyncio.Lock()
        self._speaking = False
        self._processing_turn = False
        self._turn_active = False   # a heard utterance is being transcribed / thought about / answered
        self._turns = 0
        self._turn_sequence = 0
        self._wake_sequence = 0
        self._wake_id: str | None = None
        self._active_turn_id: str | None = None
        self._capture_started_at: float | None = None
        self._last_transcript = ""
        self._last_reply = ""
        self._last_processing_acknowledgement = ""

    # -- Lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._speaker.start()
        self._task = asyncio.create_task(self._run(), name="VoiceSession")
        self._warm_task = asyncio.create_task(self._warm_output(), name="VoiceWarmup")
        mode = (
            f"wake-gated engine={self._wake_engine!r} window_sec={self._wake_window_sec:.1f}"
            if self._wake is not None
            else "continuous while unmuted"
        )
        logger.info(
            f"[voice][flow] session started mode={mode}; "
            "say the wake phrase, wait for ARMED, then speak the command"
        )
        if self._wake is not None:
            threshold = getattr(self._wake, "threshold", None)
            if isinstance(threshold, (int, float)):
                logger.info(
                    "[voice][wake] monitoring started "
                    f"engine={self._wake_engine!r} "
                    f"model={getattr(self._wake, 'model_name', 'configured')} "
                    f"threshold={threshold:.3f} score_log_interval_sec=1.0"
                )

    async def _warm_output(self) -> None:
        """Load the TTS voice and open the speaker stream now, not on the first reply. Both optional on the adapter."""
        try:
            warm = getattr(self._tts, "warmup", None)
            if callable(warm):
                started = time.perf_counter()
                await asyncio.to_thread(warm)
                logger.info(f"[voice] TTS voice loaded in {(time.perf_counter() - started) * 1000:.0f} ms")
            keep_open = getattr(self._speaker, "keep_open", None)
            if callable(keep_open):
                await asyncio.to_thread(keep_open, self._tts.sample_rate)
        except Exception as e:
            logger.warning(f"[voice] output warmup skipped: {e}")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        self._collector.reset()
        try:
            self._speaker.stop()
        except Exception:
            pass
        wake_stop = getattr(self._wake, "stop", None)
        if callable(wake_stop):
            try:
                wake_stop()
            except Exception:
                pass
        logger.info("[voice] session stopped")

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> dict:
        return {
            "running": self.is_running,
            "muted": self._gate.is_muted(),
            "speaking": self._speaking,
            "processing": self._processing_turn,
            "listening": (
                self.is_running and not self._gate.is_muted()
                and not self._speaking and not self._processing_turn
                and (self._wake is None or self._is_armed())
            ),
            "collecting_utterance": self._collector.is_speaking,
            "turns": self._turns,
            "last_transcript": self._last_transcript,
            "last_reply": self._last_reply,
            "wake_engine": self._wake_engine,
            "armed": self._is_armed(),
            "speaker_id_enabled": self._speaker_id is not None,
            "require_known_speaker": self._require_known_speaker,
            "stt_language": getattr(self._stt, "current_language", "?"),
        }

    def set_stt_language(self, language: str) -> str:
        """Pin the STT language ("en", "hi", ...) or "auto" for detection + sticky, at runtime."""
        setter = getattr(self._stt, "set_language", None)
        if not callable(setter):
            raise ValueError("the configured STT backend has no language switch")
        setter(language)
        return getattr(self._stt, "current_language", language)

    # -- Loop -------------------------------------------------------------

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[voice] loop error: {e}")
                await asyncio.sleep(0.2)

    async def _tick(self) -> None:
        # Muted: never read the mic at all. Not "read and discard" — the
        # frames must not enter the process (REQ-M-04).
        if self._gate.is_muted():
            if self._collector.is_speaking:
                logger.info(
                    "[voice][flow] stage=speech_capture_aborted "
                    f"turn={self._active_turn_id or 'unassigned'} reason=muted"
                )
                self._collector.reset()
                self._active_turn_id = None
                self._capture_started_at = None
            self._disarm(silent=True, reason="muted")
            await asyncio.sleep(0.15)
            return

        # While speaking, ignore the mic so we don't transcribe ourselves.
        if self._speaking:
            await asyncio.sleep(0.05)
            return

        frame = await asyncio.to_thread(self._audio.read, 0.5)
        if frame is None:
            return

        # Wake-word gate: while not armed, only listen for the trigger. When no
        # wake engine is configured (_wake is None) this block is skipped and
        # the loop transcribes continuously, exactly as before.
        if self._wake is not None and not self._is_armed():
            self._record_wake_audio_level(frame.pcm)
            wake_detected = self._wake_triggered(frame)
            self._log_wake_score(detected=wake_detected)
            if wake_detected:
                self._wake_sequence += 1
                self._wake_id = f"W{self._wake_sequence:04d}"
                score = getattr(self._wake, "last_score", None)
                threshold = getattr(self._wake, "threshold", None)
                score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                threshold_text = f"{threshold:.3f}" if isinstance(threshold, (int, float)) else "n/a"
                logger.info(
                    "[voice][flow] stage=wake_detected "
                    f"wake={self._wake_id} engine={self._wake_engine!r} "
                    f"score={score_text} threshold={threshold_text}; "
                    "wake phrase opens the command window and is not sent as a command"
                )
                if self._tts is None:
                    logger.warning(
                        "[voice][wake] acknowledgement skipped because no TTS engine is available"
                    )
                else:
                    acknowledgement_started = time.perf_counter()
                    logger.info(
                        "[voice][flow] stage=wake_acknowledgement_started "
                        f"wake={self._wake_id} chars={len(_WAKE_ACKNOWLEDGEMENT)}"
                    )
                    await self._speak(_WAKE_ACKNOWLEDGEMENT)
                    logger.info(
                        "[voice][flow] stage=wake_acknowledgement_complete "
                        f"wake={self._wake_id} "
                        f"elapsed_ms={(time.perf_counter() - acknowledgement_started) * 1000:.0f}"
                    )
                self._arm(reason="wake_word")
            return

        if self._speaker_id is not None:
            self._feed_speaker_frame(frame)

        was_collecting = self._collector.is_speaking
        utterance = self._collector.push(frame)
        if not was_collecting and self._collector.is_speaking:
            self._turn_sequence += 1
            self._active_turn_id = f"T{self._turn_sequence:04d}"
            self._capture_started_at = time.perf_counter()
            logger.info(
                "[voice][flow] stage=speech_detected "
                f"turn={self._active_turn_id} wake={self._wake_id or 'none'} "
                f"mode={'armed' if self._wake is not None else 'continuous'}; "
                "VAD confirmed speech, collecting command audio"
            )
        if utterance is not None:
            if self._active_turn_id is None:
                self._turn_sequence += 1
                self._active_turn_id = f"T{self._turn_sequence:04d}"
            turn_id = self._active_turn_id
            capture_ms = (
                (time.perf_counter() - self._capture_started_at) * 1000
                if self._capture_started_at is not None
                else 0.0
            )
            logger.info(
                "[voice][flow] stage=utterance_ready "
                f"turn={turn_id} capture_ms={capture_ms:.0f} "
                f"audio_sec={utterance.duration_sec:.2f} "
                f"end_reason={'max_duration' if utterance.truncated else 'silence'} "
                f"frames={utterance.frame_count}"
            )
            try:
                completed = await self._run_turn(utterance, turn_id)
            finally:
                self._active_turn_id = None
                self._capture_started_at = None
            if not completed:
                return  # muted mid-turn: it was cancelled and dropped; stay passive
            if self._gate.is_muted():
                self._disarm(silent=True, reason="muted")
                return
            if self._wake is not None:
                # Allow a quick follow-up without requiring the wake phrase again.
                self._arm(reason="follow_up")
            logger.info("[voice][flow] MIC IS ON 'U CAN TALK'")
        elif self._wake is not None and self._armed_expired():
            self._disarm(reason="timeout")

    # -- Running a turn, with mute as an interrupt -------------------------------

    async def _run_turn(self, utterance: Utterance, turn_id: str | None = None) -> bool:
        """Run one turn as a task and watch the mute switch while it runs.

        Muting must mean *stop*: not just "no new audio", but also no further
        thinking, no tool calls started and no reply spoken for something heard
        before the mute. Returns False if the turn was cancelled by a mute.
        """
        if turn_id is None:
            self._turn_sequence += 1
            turn_id = f"T{self._turn_sequence:04d}"
        token = set_voice_turn_id(turn_id)
        turn_started = time.perf_counter()
        self._processing_turn = True
        self._gate.pause_capture(source=f"voice_turn:{turn_id}")
        logger.info(
            "[voice][flow] stage=turn_started "
            f"turn={turn_id} audio_sec={utterance.duration_sec:.2f}; "
            "microphone capture paused until this reply finishes"
        )
        logger.info(
            "[voice][flow] stage=input_listening_state "
            f"turn={turn_id} listening=false microphone_stream=paused "
            "action=pause_capture_during_reply"
        )
        discarded_frames = self._discard_captured_audio()
        cancel = asyncio.Event()
        self._turn_active = True
        completion_reason = "turn_failed"
        task = asyncio.create_task(self._handle(utterance, cancel), name="VoiceTurn")
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=0.1)
                if done:
                    task.result()  # re-raise a turn failure to the loop's error handler, as before
                    discarded_frames += self._discard_captured_audio()
                    completion_reason = "turn_complete"
                    logger.info(
                        "[voice][flow] stage=turn_finished "
                        f"turn={turn_id} elapsed_ms={(time.perf_counter() - turn_started) * 1000:.0f}; "
                        "turn complete, resuming capture"
                    )
                    logger.info(
                        "[voice][flow] stage=turn_audio_discarded "
                        f"turn={turn_id} frames={discarded_frames} "
                        "reason=audio_queued_at_turn_boundary"
                    )
                    return True
                if self._gate.is_muted():
                    await self._abort_turn(task, cancel)
                    completion_reason = "muted"
                    return False
        except asyncio.CancelledError:
            task.cancel()
            discarded_frames += self._discard_captured_audio()
            logger.info(
                "[voice][flow] stage=turn_audio_discarded "
                f"turn={turn_id} frames={discarded_frames} reason=session_cancelled"
            )
            logger.info(
                "[voice][flow] stage=input_listening_state "
                f"turn={turn_id} listening=false microphone_stream=paused "
                "reason=session_cancelled"
            )
            completion_reason = "session_cancelled"
            raise
        except Exception:
            discarded_frames += self._discard_captured_audio()
            logger.info(
                "[voice][flow] stage=turn_audio_discarded "
                f"turn={turn_id} frames={discarded_frames} reason=turn_failed"
            )
            logger.info(
                "[voice][flow] stage=input_listening_state "
                f"turn={turn_id} listening=false microphone_stream=paused "
                "reason=turn_failed"
            )
            logger.exception(
                "[voice][flow] stage=turn_failed "
                f"turn={turn_id} elapsed_ms={(time.perf_counter() - turn_started) * 1000:.0f}"
            )
            raise
        finally:
            elapsed_ms = (time.perf_counter() - turn_started) * 1000
            logger.info(
                "[voice][timing] TIME TAKEN TO PROCESS THE REQUEST : "
                f"{_format_request_duration(elapsed_ms)} "
                f"(elapsed_ms={elapsed_ms:.0f}) turn={turn_id} status={completion_reason}"
            )
            capture_resumed = self._gate.resume_capture(
                source=f"voice_turn:{turn_id}"
            )
            self._processing_turn = False
            self._turn_active = False
            listening = capture_resumed and (
                self._wake is None or self._is_armed()
            )
            logger.info(
                "[voice][flow] stage=input_listening_state "
                f"turn={turn_id} listening={str(listening).lower()} "
                f"microphone_stream={'active' if capture_resumed else 'stopped'} "
                f"reason={completion_reason if capture_resumed else 'muted'}"
            )
            reset_voice_turn_id(token)

    def _discard_captured_audio(self) -> int:
        """Drop mic frames queued as capture pauses or resumes."""
        drain = getattr(self._audio, "drain", None)
        if not callable(drain):
            return 0
        try:
            discarded = drain()
        except Exception as e:
            logger.warning(f"[voice][flow] failed to discard queued mic audio: {e}")
            return 0
        return discarded if isinstance(discarded, int) else 0

    async def _abort_turn(self, task: asyncio.Task, cancel: asyncio.Event) -> None:
        logger.info("[voice] muted mid-turn: cancelling the turn and stopping playback")
        cancel.set()  # lets a streaming model stop generating
        interrupt = getattr(self._speaker, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt()
            except Exception as e:
                logger.warning(f"[voice] could not interrupt playback: {e}")
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        self._speaking = False
        self._collector.reset()
        self._reset_speaker_scoring()
        discarded_frames = self._discard_captured_audio()
        logger.info(
            "[voice][flow] stage=turn_audio_discarded "
            f"frames={discarded_frames} reason=turn_cancelled"
        )
        self._disarm(silent=True)
        self._emit("turn_cancelled", {"reason": "muted"})

    # -- Wake-word gating -------------------------------------------------

    def _is_armed(self) -> bool:
        return self._armed_until > time.monotonic()

    def _arm(self, *, reason: str = "wake_word") -> None:
        was_armed = self._is_armed()
        self._armed_until = time.monotonic() + self._wake_window_sec
        if not was_armed:
            self._armed_since = time.monotonic()
        if not was_armed:
            logger.info(
                "[voice][flow] stage=command_window_open "
                f"wake={self._wake_id or 'none'} reason={reason} "
                f"duration_sec={self._wake_window_sec:.1f}; waiting for command speech"
            )
            self._emit("wake", {"armed": True})
        else:
            logger.info(
                "[voice][flow] stage=command_window_refreshed "
                f"wake={self._wake_id or 'none'} reason={reason} "
                f"duration_sec={self._wake_window_sec:.1f}; follow-up can be spoken without repeating wake phrase"
            )

    def _disarm(self, *, silent: bool = False, reason: str = "timeout") -> None:
        self._reset_speaker_scoring()
        if self._armed_until == 0.0:
            return
        elapsed = (
            time.monotonic() - self._armed_since
            if self._armed_since is not None
            else 0.0
        )
        self._armed_until = 0.0
        self._armed_since = None
        self._wake_buffer = b""
        if self._collector.is_speaking:
            self._collector.reset()
        logger.info(
            "[voice][flow] stage=command_window_closed "
            f"wake={self._wake_id or 'none'} reason={reason} elapsed_sec={elapsed:.1f}"
        )
        if not silent:
            self._emit("wake", {"armed": False})
        if reason in {"timeout", "turn_complete"}:
            self._wake_id = None

    def _armed_expired(self) -> bool:
        return (
            self._armed_until > 0.0
            and time.monotonic() >= self._armed_until
            and not self._collector.is_speaking
        )

    def _log_wake_score(self, *, detected: bool, now: float | None = None) -> None:
        """Log periodic wake scores without emitting an entry for every audio frame."""
        score = getattr(self._wake, "last_score", None)
        threshold = getattr(self._wake, "threshold", None)
        if not isinstance(score, (int, float)) or not isinstance(threshold, (int, float)):
            self._wake_audio_square_sum = 0.0
            self._wake_audio_sample_count = 0
            self._wake_audio_peak = 0
            return

        score = float(score)
        threshold = float(threshold)
        self._wake_score_peak = (
            score if self._wake_score_peak is None else max(self._wake_score_peak, score)
        )
        current = time.monotonic() if now is None else now
        if not detected and current - self._last_wake_score_log_at < 1.0:
            return

        peak = self._wake_score_peak
        if self._wake_audio_sample_count:
            audio_rms = (
                self._wake_audio_square_sum / self._wake_audio_sample_count
            ) ** 0.5
            audio_peak = self._wake_audio_peak / 32768.0
            audio_rms_dbfs = 20.0 * math.log10(max(audio_rms, 1e-8))
            audio_peak_dbfs = 20.0 * math.log10(max(audio_peak, 1e-8))
            audio_level = (
                f"raw_audio_rms={audio_rms:.5f} raw_audio_peak={audio_peak:.5f} "
                f"raw_audio_rms_dbfs={audio_rms_dbfs:.1f} "
                f"raw_audio_peak_dbfs={audio_peak_dbfs:.1f} "
                f"raw_audio_samples={self._wake_audio_sample_count}"
            )
        else:
            audio_level = (
                "raw_audio_rms=n/a raw_audio_peak=n/a raw_audio_rms_dbfs=n/a "
                "raw_audio_peak_dbfs=n/a raw_audio_samples=0"
            )
        logger.info(
            "[voice][wake] stage=score "
            f"engine={self._wake_engine!r} model={getattr(self._wake, 'model_name', 'configured')} "
            f"score={score:.3f} peak_1s={peak:.3f} threshold={threshold:.3f} "
            f"{audio_level} detected={str(detected).lower()}"
        )
        self._wake_score_peak = None
        self._wake_audio_square_sum = 0.0
        self._wake_audio_sample_count = 0
        self._wake_audio_peak = 0
        self._last_wake_score_log_at = current

    def _record_wake_audio_level(self, pcm: bytes) -> None:
        """Accumulate raw mic levels for the periodic wake-score diagnostic."""
        import numpy as np

        usable_bytes = len(pcm) - (len(pcm) % 2)
        if usable_bytes == 0:
            return
        samples = np.frombuffer(pcm[:usable_bytes], dtype=np.int16).astype(np.int32)
        normalized = samples.astype(np.float32) / 32768.0
        self._wake_audio_square_sum += float(np.dot(normalized, normalized))
        self._wake_audio_sample_count += int(samples.size)
        self._wake_audio_peak = max(
            self._wake_audio_peak, int(np.max(np.abs(samples)))
        )

    def _wake_triggered(self, frame) -> bool:
        # Push-to-talk engines (hotkey) aren't frame-driven; they expose consume().
        consume = getattr(self._wake, "consume", None)
        if callable(consume) and self._wake_frame_bytes == 0:
            return bool(consume())
        if self._wake_frame_bytes <= 0:
            return False
        # Frame-driven engine (Porcupine): slice to its exact frame size.
        self._wake_buffer += frame.pcm
        while len(self._wake_buffer) >= self._wake_frame_bytes:
            chunk = self._wake_buffer[: self._wake_frame_bytes]
            self._wake_buffer = self._wake_buffer[self._wake_frame_bytes:]
            try:
                if self._wake.process(chunk):
                    self._wake_buffer = b""
                    return True
            except Exception as e:
                logger.warning(f"[voice] wake-word process failed: {e}")
                return False
        return False

    # -- Speaker recognition ------------------------------------------------

    def _feed_speaker_frame(self, frame) -> None:
        """Slice raw PCM into the recognizer's frame size and accumulate a
        running score per enrolled speaker. Mirrors _wake_triggered's
        buffering idiom - same reason: the mic callback's blocksize rarely
        matches a vendor SDK's required frame size exactly."""
        if self._speaker_frame_bytes <= 0:
            return
        self._speaker_buffer += frame.pcm
        while len(self._speaker_buffer) >= self._speaker_frame_bytes:
            chunk = self._speaker_buffer[: self._speaker_frame_bytes]
            self._speaker_buffer = self._speaker_buffer[self._speaker_frame_bytes:]
            try:
                scores = self._speaker_id.process(chunk)
            except Exception as e:
                logger.warning(f"[voice] speaker-id process failed: {e}")
                return
            if not scores:
                continue
            if not self._speaker_score_sums:
                self._speaker_score_sums = [0.0] * len(scores)
            for i, s in enumerate(scores):
                self._speaker_score_sums[i] += s
            self._speaker_score_count += 1

    def _resolve_speaker(self) -> tuple[str | None, bool]:
        """Mean score over the whole utterance, not a single frame - Eagle's
        scores stabilise with more audio, so deciding off one noisy frame
        would risk rejecting a real enrolled speaker, not just stray talk."""
        if self._speaker_id is None or not self._speaker_score_sums or self._speaker_score_count == 0:
            return None, False
        means = [s / self._speaker_score_count for s in self._speaker_score_sums]
        best_i = max(range(len(means)), key=lambda i: means[i])
        if means[best_i] < self._speaker_match_threshold:
            return None, False
        names = getattr(self._speaker_id, "speaker_names", [])
        name = names[best_i] if best_i < len(names) else None
        return name, name is not None

    def _reset_speaker_scoring(self) -> None:
        self._speaker_buffer = b""
        self._speaker_score_sums = []
        self._speaker_score_count = 0

    # -- One turn ---------------------------------------------------------

    async def _handle(self, utterance: Utterance, cancel: asyncio.Event | None = None) -> None:
        turn_started = time.perf_counter()
        capture_elapsed_ms = max(
            0.0, (datetime.now() - utterance.started_at).total_seconds() * 1000
        )
        logger.info(
            "[voice][timing] stage=utterance_capture "
            f"elapsed_ms={capture_elapsed_ms:.0f} audio_sec={utterance.duration_sec:.2f} "
            f"frames={utterance.frame_count} truncated={utterance.truncated}"
        )
        self._emit("utterance", {"duration_sec": round(utterance.duration_sec, 2)})

        stt_started = time.perf_counter()
        transcript = await self._stt.transcribe(utterance.audio)
        stt_ms = (time.perf_counter() - stt_started) * 1000
        text = (transcript.text or "").strip()
        logger.info(
            "[voice][timing] stage=stt "
            f"elapsed_ms={stt_ms:.0f} audio_sec={utterance.duration_sec:.2f} "
            f"rtf={stt_ms / max(utterance.duration_sec * 1000, 1):.2f} "
            f"language={transcript.language} language_probability={transcript.confidence:.3f} "
            f"chars={len(text)}"
        )
        if len(text) < self._min_chars:
            logger.info("[voice] transcript too short; ignoring")
            return

        # Re-check: the user may have muted while we were transcribing.
        if self._gate.is_muted():
            logger.info("[voice] muted during transcription - dropping turn")
            return

        # Speech-to-text invents text on noise ("subscribe to our channel", "Bye. Bye. Bye..."):
        # that is not something the user said, so it must not become a turn.
        reason = artifact_reason(text, utterance.duration_sec)
        if reason is not None:
            logger.info(f"[voice] dropping probable STT artefact ({reason}): {log_text(text, 80)}")
            self._emit("artifact_ignored", {"reason": reason})
            return

        # Resolve who (if anyone enrolled) said this, then clear the running
        # scores so the next utterance starts clean. Reading before
        # resetting is safe - no other frames are fed while _handle runs,
        # this coroutine owns the only consumer of _tick's frame loop.
        speaker_name, speaker_known = self._resolve_speaker()
        self._reset_speaker_scoring()
        if self._require_known_speaker and self._speaker_id is not None and not speaker_known:
            logger.info("[voice] unrecognized speaker; dropping turn as stray talk")
            self._emit("speaker_rejected", {})
            return

        # Second line of defence against self-conversation. Even with correct
        # playback waiting, a loud room or an open speaker can bleed our own
        # words back in. If what we just "heard" is essentially what we just
        # said, it isn't a user turn.
        if self._echo_guard and self._is_echo(text):
            logger.warning(f"[voice] ignoring probable self-echo: {log_text(text, 60)}")
            self._emit("echo_ignored", {"text": text})
            return

        self._last_transcript = text
        logger.info(f"[voice] heard: {log_text(text)}")
        self._emit("transcript", {"text": text})

        ctx = AgentContext(user_message=text, from_voice=True)
        if speaker_name:
            ctx.metadata["speaker_name"] = speaker_name
        if self._speak_replies and self._tts is not None:
            # Stream + speak per sentence so the first words play while the rest
            # of the reply is still generating (low time-to-first-audio).
            reply = await self._stream_and_speak(ctx, cancel)
        else:
            generation_started = time.perf_counter()
            logger.info("[voice][flow] stage=assistant_generation_started mode=non_streaming")
            result = await self._supervisor.execute(ctx)
            logger.info(
                "[voice][flow] stage=assistant_generation_complete "
                f"elapsed_ms={(time.perf_counter() - generation_started) * 1000:.0f} "
                "mode=non_streaming"
            )
            reply = (result.response or "").strip()

        self._last_reply = reply
        self._turns += 1
        logger.info(
            "[voice][timing] stage=turn_complete "
            f"elapsed_ms={(time.perf_counter() - turn_started) * 1000:.0f} "
            f"reply_chars={len(reply)}"
        )
        logger.info(f"[voice] reply: {log_text(reply, 120)}")
        self._emit("reply", {"text": reply})

    def _is_echo(self, heard: str) -> bool:
        """True if `heard` looks like our own recent speech coming back."""
        got = {w for w in _WORD_RE.findall(heard.lower()) if len(w) > 2}
        if not got:
            return False
        for spoken in (self._last_reply, self._last_processing_acknowledgement):
            said = {w for w in _WORD_RE.findall(spoken.lower()) if len(w) > 2}
            if said and len(said & got) / len(got) >= 0.6:
                return True
        return False

    async def speak(self, text: str) -> None:
        """Speak arbitrary text through the same echo-guarded path as a reply."""
        await self._speak(text)

    # -- Announcements (reminders): speak only when nothing else is going on -------------

    def available(self) -> bool:
        return self.is_running

    def is_busy(self) -> bool:
        """True while we are speaking, the user is talking (or the wake window is open for them to), or a turn is in
        flight. Being muted is NOT busy: a closed mic does not stop the speaker."""
        return self._speaking or self._turn_active or self._collector.is_speaking or self._is_armed()

    async def announce(self, text: str) -> bool:
        """Speak `text` only if idle right now; False means busy, try later. The idle check and `_speaking = True`
        (set at the top of `_speak`) happen with no `await` between them, so a turn cannot start in the gap, and the
        mic stays closed while we talk exactly as for a reply."""
        if not self.is_running or self.is_busy():
            return False
        await self._speak(text)
        return True

    async def _speak(self, text: str, language: str | None = None) -> None:
        """Synthesise and play with the mic held closed until playback truly ends."""
        self._speaking = True
        self._emit("speaking", {"text": text})
        try:
            spoke, _ = await self._say_pcm(text, language=language)
            if spoke:
                await self._wait_playback()
        except Exception as e:
            logger.error(f"[voice] TTS failed: {e}")
        finally:
            await self._after_speaking()

    async def _stream_and_speak(self, ctx: AgentContext, cancel: asyncio.Event | None = None) -> str:
        """Stream the reply and speak each sentence as it completes.

        Speaking on sentence boundaries plays the first words while the rest of
        the reply is still generating — the dominant perceived-latency win for a
        CPU-bound voice assistant.
        """
        self._speaking = True
        self._emit("speaking", {"text": ""})
        stream_started = time.perf_counter()
        timing = {"stream_started": stream_started, "tts_ms": 0.0, "first_audio_ms": None}
        logger.info("[voice][flow] stage=assistant_generation_started mode=streaming")
        parts: list[str] = []
        buffer = ""
        spoke_anything = False
        first_chunk_ms: float | None = None
        chunk_count = 0
        stream_finished_ms: float | None = None
        reply_audio_started = asyncio.Event()
        acknowledgement_started = asyncio.Event()
        acknowledgement_task = asyncio.create_task(
            self._speak_processing_acknowledgement(reply_audio_started, acknowledgement_started)
        )
        try:
            async for chunk in self._supervisor.execute_stream(ctx, cancel_event=cancel):
                if not chunk:
                    continue
                chunk_count += 1
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - stream_started) * 1000
                    logger.info(
                        "[voice][timing] stage=model_first_chunk "
                        f"elapsed_ms={first_chunk_ms:.0f}"
                    )
                parts.append(chunk)
                buffer += chunk
                sentence, buffer = self._pop_sentence(buffer)
                while sentence:
                    reply_audio_started.set()
                    spoke, _ = await self._say_pcm(sentence, timing, language=ctx.metadata.get("language"))
                    spoke_anything = spoke or spoke_anything
                    sentence, buffer = self._pop_sentence(buffer)
            tail = buffer.strip()
            if tail:
                reply_audio_started.set()
                spoke, _ = await self._say_pcm(tail, timing, language=ctx.metadata.get("language"))
                spoke_anything = spoke or spoke_anything
            stream_finished_ms = (time.perf_counter() - stream_started) * 1000
            if spoke_anything:
                await self._wait_playback()
        except Exception as e:
            logger.error(f"[voice] streaming speak failed: {e}")
        finally:
            stream_ms = stream_finished_ms or (time.perf_counter() - stream_started) * 1000
            first_chunk = f"{first_chunk_ms:.0f}" if first_chunk_ms is not None else "n/a"
            first_audio = (
                f"{timing['first_audio_ms']:.0f}"
                if timing["first_audio_ms"] is not None
                else "n/a"
            )
            logger.info(
                "[voice][timing] stage=model_stream_complete "
                f"elapsed_ms={stream_ms:.0f} "
                f"work_excluding_tts_ms={max(0.0, stream_ms - timing['tts_ms']):.0f} "
                f"tts_ms={timing['tts_ms']:.0f} "
                f"first_chunk_ms={first_chunk} first_audio_queued_ms={first_audio} "
                f"chunks={chunk_count} chars={sum(map(len, parts))}"
            )
            if not acknowledgement_started.is_set() and not acknowledgement_task.done():
                acknowledgement_task.cancel()
            try:
                await acknowledgement_task
            except asyncio.CancelledError:
                pass
            await self._after_speaking()
        return "".join(parts).strip()

    async def _speak_processing_acknowledgement(
        self,
        reply_audio_started: asyncio.Event,
        acknowledgement_started: asyncio.Event,
    ) -> None:
        await asyncio.sleep(_PROCESSING_ACK_DELAY_SEC)
        try:
            async with self._speech_lock:
                if reply_audio_started.is_set():
                    return
                acknowledgement_started.set()
                self._last_processing_acknowledgement = _PROCESSING_ACKNOWLEDGEMENT
                logger.info("[voice][flow] stage=processing_acknowledgement")
                spoke, _ = await self._synthesize_pcm(_PROCESSING_ACKNOWLEDGEMENT)
                if spoke:
                    await self._wait_playback()
        except Exception as e:
            logger.warning(f"[voice] processing acknowledgement failed: {e}")

    @staticmethod
    def _pop_sentence(buffer: str) -> tuple[str, str]:
        match = _SENTENCE_END_RE.search(buffer)
        if not match:
            return "", buffer
        cut = match.end()
        return buffer[:cut].strip(), buffer[cut:]

    async def _say_pcm(self, text: str, timing: dict | None = None, language: str | None = None) -> tuple[bool, float]:
        async with self._speech_lock:
            return await self._synthesize_pcm(text, timing, language=language)

    async def _synthesize_pcm(
        self, text: str, timing: dict | None = None, language: str | None = None
    ) -> tuple[bool, float]:
        started = time.perf_counter()
        spoke = False
        audio_bytes = 0
        logger.info(f"[voice][flow] stage=tts_synthesis_started chars={len(text)}")
        async for pcm in self._tts.synthesize(text, voice=language):
            if pcm:
                self._speaker.play_pcm(pcm, self._tts.sample_rate)
                spoke = True
                audio_bytes += len(pcm)
                if timing is not None and timing["first_audio_ms"] is None:
                    timing["first_audio_ms"] = (time.perf_counter() - timing["stream_started"]) * 1000
                    logger.info(
                        "[voice][timing] stage=first_audio_queued "
                        f"elapsed_ms={timing['first_audio_ms']:.0f}"
                    )
        elapsed_ms = (time.perf_counter() - started) * 1000
        if timing is not None:
            timing["tts_ms"] += elapsed_ms
        logger.info(
            "[voice][flow] stage=tts_synthesis_complete "
            f"elapsed_ms={elapsed_ms:.0f} chars={len(text)} "
            f"audio_sec={audio_bytes / (2 * max(self._tts.sample_rate, 1)):.2f} "
            f"produced_audio={spoke}"
        )
        if not spoke:
            logger.warning("[voice] TTS produced no audio for this text")
        return spoke, elapsed_ms

    async def _wait_playback(self) -> None:
        wait_done = getattr(self._speaker, "wait_done", None)
        if callable(wait_done):
            started = time.perf_counter()
            finished = await asyncio.to_thread(wait_done, self._speak_timeout_sec)
            logger.info(
                "[voice][timing] stage=playback_wait "
                f"elapsed_ms={(time.perf_counter() - started) * 1000:.0f} "
                f"finished={finished}"
            )
            if not finished:
                logger.warning("[voice] playback did not finish within timeout")
        else:
            # Adapter without completion signalling — conservative sleep rather
            # than reopening the mic blind.
            await asyncio.sleep(1.0)

    async def _after_speaking(self) -> None:
        # Settle THEN drain so the tail of our own voice cannot leak into the
        # next turn. The outer turn resumes capture after all processing ends.
        await asyncio.sleep(self._post_speak_settle_sec)
        drain = getattr(self._audio, "drain", None)
        if callable(drain):
            drain()
        self._collector.reset()
        self._reset_speaker_scoring()
        self._speaking = False
        self._emit("idle", {})

    def _emit(self, kind: str, payload: dict) -> None:
        try:
            self._on_event({"kind": kind, "at": datetime.now().isoformat(), **payload})
        except Exception:
            pass