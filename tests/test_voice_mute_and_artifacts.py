"""Voice loop: mute interrupts the turn in progress, STT artefacts are dropped, wake word fails closed.

No audio hardware or model: fakes only. Run: pytest tests/test_voice_mute_and_artifacts.py
"""

import asyncio
import logging
import time
from datetime import datetime

import pytest

from domain.entities.agent_result import AgentResult
from domain.entities.utterance import Utterance
from domain.policies.transcript_policy import artifact_reason
from domain.value_objects.audio_window import AudioWindow
from domain.value_objects.transcript import Transcript
from service.voice.voice_session import VoiceSession, _format_request_duration


# ---- transcript policy -------------------------------------------------------

HALLUCINATIONS = [
    ("Yes. If you want to see more of this video, please subscribe to our channel. Thank you. "
     "We'll see you next week. " + "Bye. " * 25, 1.2),
    ("So I am going to help you and help you with a good time. I am going to help you with a good time "
     "with a good time. I'm going to help you with a good time.", 15.0),
    ("Bye. Bye. Bye. Bye. Bye. Bye.", 3.0),
    ("Thanks for watching!", 2.0),
    ("[music]", 2.0),
    ("(applause)", 1.0),
    ("♪ ♪", 1.0),
    ("Thank you.", 0.8),
    ("you", 0.5),
    ("", 1.0),
]
GENUINE = [
    ("what are my tasks", 2.0), ("Can you tell me the upcoming movies this month from tollywood?", 4.0),
    ("yes", 0.9), ("no no no", 2.0), ("remove the milk task", 2.0), ("Thank you.", 3.0),
    ("Okay, so I have several tasks in my pocket. What are the tasks I have?", 6.0),
    ("I need to buy milk, milk, and more milk", 3.0), ("one two three four five six", 3.0),
]


@pytest.mark.parametrize("text,dur", HALLUCINATIONS, ids=[f"artefact{i}" for i in range(len(HALLUCINATIONS))])
def test_stt_artefacts_are_recognised(text, dur):
    assert artifact_reason(text, dur) is not None


@pytest.mark.parametrize("text,dur", GENUINE)
def test_genuine_speech_is_never_dropped(text, dur):
    assert artifact_reason(text, dur) is None


# ---- fakes -----------------------------------------------------------------

class _Gate:
    def __init__(self):
        self.muted = False
        self.paused = False
        self.capture_events = []

    def is_muted(self):
        return self.muted

    def pause_capture(self, source="voice_turn"):
        self.paused = True
        self.capture_events.append(("pause", source))

    def resume_capture(self, source="voice_turn"):
        self.paused = False
        self.capture_events.append(("resume", source))
        return not self.muted


class _Collector:
    is_speaking = False

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1


class _Speaker:
    def __init__(self):
        self.interrupted = 0

    def interrupt(self):
        self.interrupted += 1


class _STT:
    def __init__(self, text):
        self._text = text

    async def transcribe(self, audio):
        return Transcript(text=self._text, confidence=1.0)


class _Supervisor:
    def __init__(self, work_sec=0.0):
        self.calls = 0
        self.cancel_events = []
        self._work = work_sec

    async def execute(self, ctx):
        self.calls += 1
        await asyncio.sleep(self._work)
        return AgentResult(agent_name="responder", response="ok")

    async def execute_stream(self, ctx, cancel_event=None):
        self.calls += 1
        self.cancel_events.append(cancel_event)
        await asyncio.sleep(self._work)
        yield "hello."


def _utterance(sec=2.0):
    now = datetime.now()
    window = AudioWindow(pcm=b"\x00\x00" * 160, sample_rate=16000, channels=1, started_at=now, duration_sec=sec)
    return Utterance(audio=window, started_at=now, duration_sec=sec, frame_count=1)


def _session(gate, supervisor, stt_text="what are my tasks", speaker=None, collector=None):
    return VoiceSession(
        audio=object(), collector=collector or _Collector(), stt=_STT(stt_text), tts=None,
        speaker=speaker or _Speaker(), supervisor=supervisor, capture_gate=gate, speak_replies=False,
    )


# ---- artefacts never become turns ---------------------------------------------

def test_a_hallucinated_transcript_does_not_reach_the_assistant():
    sup = _Supervisor()
    session = _session(_Gate(), sup, stt_text="Please subscribe to our channel. Bye. Bye. Bye. Bye. Bye. Bye.")
    asyncio.run(session._handle(_utterance(1.2)))
    assert sup.calls == 0 and session.status()["turns"] == 0


def test_a_real_transcript_still_gets_an_answer():
    sup = _Supervisor()
    session = _session(_Gate(), sup)
    asyncio.run(session._handle(_utterance()))
    assert sup.calls == 1 and session.status()["turns"] == 1


def test_voice_turn_logs_stage_timings(caplog):
    session = VoiceSession(
        audio=object(), collector=_Collector(), stt=_STT("what are my tasks"), tts=_FakeTTS(),
        speaker=_PlayingSpeaker(), supervisor=_Supervisor(work_sec=0.01),
        capture_gate=_Gate(), speak_replies=True,
    )
    with caplog.at_level(logging.INFO, logger="veda"):
        asyncio.run(session._handle(_utterance()))

    for stage in (
        "utterance_capture", "stt", "assistant_generation_started",
        "model_first_chunk", "first_audio_queued", "tts_synthesis_complete",
        "playback_wait", "model_stream_complete", "turn_complete",
    ):
        assert f"stage={stage}" in caplog.text


def test_slow_voice_turn_speaks_processing_acknowledgement(monkeypatch):
    class RecordingTTS(_FakeTTS):
        def __init__(self):
            self.spoken_texts = []

        async def synthesize(self, text, voice=None):
            self.spoken_texts.append(text)
            yield b"\x00\x00" * 10

    monkeypatch.setattr("service.voice.voice_session._PROCESSING_ACK_DELAY_SEC", 0.01)
    tts = RecordingTTS()
    session = VoiceSession(
        audio=object(), collector=_Collector(), stt=_STT("what are my tasks"), tts=tts,
        speaker=_PlayingSpeaker(), supervisor=_Supervisor(work_sec=0.05),
        capture_gate=_Gate(), speak_replies=True,
    )

    asyncio.run(session._handle(_utterance()))

    assert "Working on that." in tts.spoken_texts
    assert "hello." in tts.spoken_texts
    assert session._is_echo("Working on that")


def test_fast_voice_turn_skips_processing_acknowledgement(monkeypatch):
    class RecordingTTS(_FakeTTS):
        def __init__(self):
            self.spoken_texts = []

        async def synthesize(self, text, voice=None):
            self.spoken_texts.append(text)
            yield b"\x00\x00" * 10

    monkeypatch.setattr("service.voice.voice_session._PROCESSING_ACK_DELAY_SEC", 1.0)
    tts = RecordingTTS()
    session = VoiceSession(
        audio=object(), collector=_Collector(), stt=_STT("what are my tasks"), tts=tts,
        speaker=_PlayingSpeaker(), supervisor=_Supervisor(work_sec=0.001),
        capture_gate=_Gate(), speak_replies=True,
    )

    asyncio.run(session._handle(_utterance()))

    assert "Working on that." not in tts.spoken_texts
    assert "hello." in tts.spoken_texts


def test_wake_window_refreshes_after_a_completed_command(caplog):
    session = VoiceSession(
        audio=object(), collector=_Collector(), stt=_STT("hello"), tts=None,
        speaker=_Speaker(), supervisor=_Supervisor(), capture_gate=_Gate(),
        wake_word=object(), wake_engine="openwakeword", wake_window_sec=4.0,
    )
    session._wake_id = "W0001"

    with caplog.at_level(logging.INFO, logger="veda"):
        session._arm(reason="wake_word")
        session._arm(reason="follow_up")

    assert "stage=command_window_open wake=W0001 reason=wake_word" in caplog.text
    assert "stage=command_window_refreshed wake=W0001 reason=follow_up" in caplog.text
    assert session._is_armed()


def test_passive_wake_monitor_logs_score_and_threshold_at_one_second_intervals(caplog):
    class Wake:
        last_score = 0.31
        threshold = 0.4
        model_name = "hey_veda"

    session = VoiceSession(
        audio=object(), collector=_Collector(), stt=_STT("hello"), tts=None,
        speaker=_Speaker(), supervisor=_Supervisor(), capture_gate=_Gate(),
        wake_word=Wake(), wake_engine="openwakeword",
    )
    session._last_wake_score_log_at = 10.0
    session._record_wake_audio_level(b"\x00\x40\x00\xc0")

    with caplog.at_level(logging.INFO, logger="veda"):
        session._log_wake_score(detected=False, now=10.5)
        assert "stage=score" not in caplog.text
        session._wake.last_score = 0.39
        session._log_wake_score(detected=False, now=10.8)
        session._wake.last_score = 0.37
        session._log_wake_score(detected=False, now=11.1)

    assert "model=hey_veda score=0.370 peak_1s=0.390 threshold=0.400" in caplog.text
    assert (
        "raw_audio_rms=0.50000 raw_audio_peak=0.50000 "
        "raw_audio_rms_dbfs=-6.0 raw_audio_peak_dbfs=-6.0 raw_audio_samples=2"
    ) in caplog.text
    assert "detected=false" in caplog.text


def test_voice_flow_logs_wake_command_turn_and_follow_up(caplog):
    class Wake:
        frame_length = 1
        last_score = 0.9
        threshold = 0.5

        def process(self, _pcm):
            return True

    class Audio:
        def read(self, _timeout):
            return _utterance().audio

    class Collector:
        is_speaking = False

        def __init__(self):
            self.pushes = 0

        def push(self, _frame):
            self.pushes += 1
            if self.pushes == 1:
                self.is_speaking = True
                return None
            self.is_speaking = False
            return _utterance()

        def reset(self):
            self.is_speaking = False

    class CapturingTTS:
        sample_rate = 16000

        def __init__(self):
            self.spoken = []

        async def synthesize(self, text, voice=None):
            self.spoken.append(text)
            yield b"\x00\x00"

    class Speaker(_Speaker):
        def play_pcm(self, pcm, rate):
            pass

        def wait_done(self, timeout):
            return True

    tts = CapturingTTS()
    speaker = Speaker()
    session = VoiceSession(
        audio=Audio(), collector=Collector(), stt=_STT("hello there"), tts=tts,
        speaker=speaker, supervisor=_Supervisor(), capture_gate=_Gate(),
        wake_word=Wake(), wake_engine="openwakeword", wake_window_sec=4.0,
        speak_replies=False,
    )

    async def scenario():
        await session._tick()  # "Hey Veda" detected; its audio is not transcribed
        await session._tick()  # VAD confirms command speech
        await session._tick()  # silence closes utterance; process the turn

    with caplog.at_level(logging.INFO, logger="veda"):
        asyncio.run(scenario())

    stages = [
        record.message.split("stage=", 1)[1].split()[0]
        for record in caplog.records
        if "[voice][flow]" in record.message and "stage=" in record.message
    ]
    for expected in (
        "wake_detected", "wake_acknowledgement_started", "wake_acknowledgement_complete",
        "command_window_open", "speech_detected", "utterance_ready",
        "turn_started", "turn_finished", "command_window_refreshed",
    ):
        assert expected in stages
    assert tts.spoken == ["Hey there, what's up?"]
    assert "MIC IS ON 'U CAN TALK'" in caplog.text
    turn_records = [
        record for record in caplog.records
        if getattr(record, "voice_turn_id", None) == "T0001"
    ]
    assert turn_records


# ---- mute interrupts the turn ----------------------------------------------------

def test_mute_mid_turn_cancels_it_and_stops_playback():
    async def scenario():
        gate, speaker, collector = _Gate(), _Speaker(), _Collector()
        sup = _Supervisor(work_sec=5.0)  # a long-running turn (slow model / tool)
        session = _session(gate, sup, speaker=speaker, collector=collector)
        turn = asyncio.create_task(session._run_turn(_utterance()))
        await asyncio.sleep(0.25)       # the turn is now in flight
        gate.muted = True               # the user mutes
        started = time.monotonic()
        completed = await asyncio.wait_for(turn, timeout=3)
        return completed, time.monotonic() - started, speaker, collector, session, gate

    completed, elapsed, speaker, collector, session, gate = asyncio.run(scenario())
    assert completed is False                 # reported as cancelled
    assert elapsed < 1.0                      # stopped promptly, not after the 5 s of work
    assert speaker.interrupted == 1           # playback cut off
    assert collector.resets >= 1
    assert session.status()["turns"] == 0 and session.status()["speaking"] is False
    assert gate.capture_events[0][0] == "pause"
    assert gate.capture_events[-1][0] == "resume" and gate.paused is False


def test_the_streaming_model_is_told_to_stop_on_mute():
    async def scenario():
        gate = _Gate()
        sup = _Supervisor(work_sec=5.0)
        session = VoiceSession(
            audio=object(), collector=_Collector(), stt=_STT("what are my tasks"), tts=_FakeTTS(),
            speaker=_PlayingSpeaker(), supervisor=sup, capture_gate=gate, speak_replies=True,
        )
        turn = asyncio.create_task(session._run_turn(_utterance()))
        await asyncio.sleep(0.25)
        gate.muted = True
        await asyncio.wait_for(turn, timeout=3)
        return sup

    sup = asyncio.run(scenario())
    assert sup.cancel_events and sup.cancel_events[0].is_set()  # generation is cancelled, not left running


def test_an_unmuted_turn_runs_to_completion():
    async def scenario():
        sup = _Supervisor(work_sec=0.2)
        session = _session(_Gate(), sup)
        return await session._run_turn(_utterance()), sup

    completed, sup = asyncio.run(scenario())
    assert completed is True and sup.calls == 1


def test_request_processing_duration_is_logged(caplog):
    async def scenario():
        session = _session(_Gate(), _Supervisor())
        return await session._run_turn(_utterance(), "T0001")

    with caplog.at_level(logging.INFO, logger="veda"):
        assert asyncio.run(scenario()) is True

    assert "TIME TAKEN TO PROCESS THE REQUEST :" in caplog.text
    assert "turn=T0001 status=turn_complete" in caplog.text
    assert "elapsed_ms=" in caplog.text


def test_request_processing_duration_is_logged_after_tts_playback(caplog):
    session = VoiceSession(
        audio=object(), collector=_Collector(), stt=_STT("what are my tasks"), tts=_FakeTTS(),
        speaker=_PlayingSpeaker(), supervisor=_Supervisor(), capture_gate=_Gate(), speak_replies=True,
    )

    with caplog.at_level(logging.INFO, logger="veda"):
        assert asyncio.run(session._run_turn(_utterance(), "T0002")) is True

    messages = [record.getMessage() for record in caplog.records]
    playback_done = next(i for i, message in enumerate(messages) if "stage=playback_wait" in message)
    duration_logged = next(
        i for i, message in enumerate(messages)
        if "TIME TAKEN TO PROCESS THE REQUEST :" in message
    )
    assert playback_done < duration_logged


@pytest.mark.parametrize(
    ("elapsed_ms", "expected"),
    [(999, "999 ms"), (60_000, "1.00 mins"), (3_600_000, "1.00 hr")],
)
def test_request_duration_uses_readable_units(elapsed_ms, expected):
    assert _format_request_duration(elapsed_ms) == expected


def test_turn_discards_audio_queued_while_assistant_is_processing(caplog):
    class Audio:
        def __init__(self):
            self.drain_calls = 0

        def drain(self):
            self.drain_calls += 1
            return 1

    async def scenario():
        audio = Audio()
        gate = _Gate()
        processing_started = asyncio.Event()
        session = None

        class PauseAwareSupervisor(_Supervisor):
            async def execute(self, ctx):
                assert gate.paused
                assert session is not None
                assert session.status()["processing"] is True
                assert session.status()["listening"] is False
                processing_started.set()
                return await super().execute(ctx)

        session = VoiceSession(
            audio=audio, collector=_Collector(), stt=_STT("what are my tasks"), tts=None,
            speaker=_Speaker(), supervisor=PauseAwareSupervisor(work_sec=0.25),
            capture_gate=gate, speak_replies=False,
        )
        turn = asyncio.create_task(session._run_turn(_utterance(), "T0001"))
        await processing_started.wait()
        completed = await turn
        return completed, audio, gate, session

    with caplog.at_level(logging.INFO, logger="veda"):
        completed, audio, gate, session = asyncio.run(scenario())

    assert completed
    assert audio.drain_calls >= 2  # before processing and before capture resumes
    assert gate.capture_events == [
        ("pause", "voice_turn:T0001"),
        ("resume", "voice_turn:T0001"),
    ]
    assert session.status()["processing"] is False
    assert "stage=turn_audio_discarded" in caplog.text
    assert "reason=audio_queued_at_turn_boundary" in caplog.text
    assert "stage=input_listening_state turn=T0001 listening=false microphone_stream=paused" in caplog.text
    assert "stage=input_listening_state turn=T0001 listening=true" in caplog.text


class _FakeTTS:
    sample_rate = 22050

    async def synthesize(self, text, voice=None):
        yield b"\x00\x00" * 10


class _PlayingSpeaker(_Speaker):
    def play_pcm(self, pcm, rate):
        pass

    def wait_done(self, timeout):
        return True


# ---- SpeakerPlayback.interrupt ----------------------------------------------------

def test_playback_interrupt_drops_queued_audio_and_goes_idle(monkeypatch):
    import sys
    import types

    from tpa.audio.playback import SpeakerPlayback

    class Stream:   # never calls the callback: chunks just queue up
        latency = 0.0

        def __init__(self, **_kw):
            pass

        def start(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=Stream))
    sp = SpeakerPlayback()
    for _ in range(3):
        sp.play_pcm(b"\x00\x00", 16000)
    assert sp.is_playing
    sp.interrupt()
    assert not sp.is_playing and not sp._chunks


def test_playback_plays_queued_audio_in_order_then_silence_and_goes_idle(monkeypatch):
    import sys
    import types

    import numpy as np

    from tpa.audio.playback import SpeakerPlayback

    class Stream:
        latency = 0.0

        def __init__(self, callback, **_kw):
            self.callback = callback

        def start(self):
            pass

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=Stream))
    sp = SpeakerPlayback()
    sp.play_pcm(np.array([1, 2, 3], dtype=np.int16).tobytes(), 16000)
    sp.play_pcm(np.array([4, 5], dtype=np.int16).tobytes(), 16000)
    out = np.full((4, 1), 9, dtype=np.int16)
    sp._stream.callback(out, 4, None, None)
    assert out[:, 0].tolist() == [1, 2, 3, 4] and sp.is_playing   # spans chunks; one sample still queued
    sp._stream.callback(out, 4, None, None)
    assert out[:, 0].tolist() == [5, 0, 0, 0]   # the rest, then silence: the stream never runs dry
    assert not sp.is_playing and sp.wait_done(timeout=1)


# ---- wake word fails closed ---------------------------------------------------------

def test_a_configured_wake_word_that_cannot_start_disables_voice_instead_of_listening_always(monkeypatch):
    from bootstrap.builders import voice as voice_builder
    from core.config import load_full_config

    cfg = load_full_config().model_copy(deep=True)
    cfg.audio.voice_enabled, cfg.audio.wake_word_enabled, cfg.audio.wake_engine = True, True, "openwakeword"
    for name in ("build_vad", "build_stt", "build_tts", "build_speaker"):
        monkeypatch.setattr(voice_builder, name, lambda *_a, **_k: object())
    monkeypatch.setattr(voice_builder, "build_wake_word", lambda _cfg: None)  # the model failed to load
    session = voice_builder.build_voice_session(cfg, audio=object(), capture_gate=_Gate(), supervisor=object(), event_bus=object())
    assert session is None
