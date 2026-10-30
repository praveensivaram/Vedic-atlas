"""Pydantic configuration section models.

Donor: veda/config.py (337 real lines, not the 264 FILE_MAP estimated —
read in full). Split per migration rule 3: this module holds only the
declarative Pydantic shapes; loading/caching lives in core/config.py, the
actual values live in config/*.yaml (one file per section below).

Section-by-section changes from the donor:
  - `VedaConfig` -> `AppSectionConfig`: dropped `model`/`voice` (a cloud
    model name and a Windows SAPI voice name — both superseded by
    `InferenceConfig`/`AudioConfig`).
  - `BrainConfig` -> `InferenceConfig`: dropped `backend` cloud values
    (`claude`/`copilot`/`hybrid_cloud` — see tpa/inference/factory.py's
    hard reject), all `copilot_*` fields, all `warm_pool_*` fields (no CLI
    cold-start to hide — Ollama's own `keep_alive` covers this). Added
    `model_path`, `n_ctx`, `n_threads` for the llama.cpp backend.
  - `VoiceConfig` split into `AudioConfig` (the voice I/O stack: wake
    word, STT, TTS, mic/speaker device selection) and `SensingConfig`
    (ambient-dispatch tuning: proactivity default, debounce/rate-limit
    windows — the constructor args `SupervisorAgent`/`Debouncer`/
    `RateLimiter` already take).
  - `VisionConfig` / `TeamsConfig`: dropped entirely — both out of scope.
  - `AgentsConfig`/`AgentEntry`: dropped `research`/`coder`/`communicator`
    entries — none of those agents exist in this build (no web-search
    agent, no code-runner, no Teams). `chat` renamed `responder` to match
    the actual registered agent name (service/agent/responder.py).
  - `SkillsConfig`/`SkillEntry`: dropped `vision`/`vscode`/`pycharm`/
    `intellij`/`teams` entries (skills that don't exist); `web_search`
    dropped too — there's no skill wrapping `service/lookup/` yet.
  - `GovernanceConfig`: flattened the donor's nested `sre:` sub-section
    into `circuit_breaker_threshold`/`circuit_breaker_timeout` directly,
    matching `service/governance/factory.py`'s `getattr(config, ...)`
    calls. `policies_dir` default now points at `tpa/governance/policies`.
  - `GuardrailsConfig`: unchanged in shape — still real, still used by
    `service/guardrails/*`.
  - `HooksConfig` dropped entirely: the donor's version was a list of
    `"builtin:x"` name strings with no dynamic loader behind it — there
    is exactly one fixed set of built-in hooks
    (`service/hooks/dispatcher.py`), and `bootstrap/builders/safety.py` wires them directly.
    A config file whose only possible values are the same five hardcoded
    strings adds a config surface with no real flexibility behind it.
  - New: `PrivacyConfig` (`mute_gpio_pin`, `indicator_gpio_pin` for the Pi
    hardware profile; `online.allowlist` for the egress allow-list shared
    by `tpa/online/` and `service/lookup/`) — none of this existed in the
    donor, which had no network allow-list or GPIO concept at all.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MultilingualConfig(BaseModel):
    """app.multilingual: reply in the user's language, or say you did not understand."""

    enabled: bool = True
    # When neither the deterministic detector nor the control model can name the
    # language, answer with `unknown_reply` instead of guessing English.
    reject_unknown: bool = True
    # Only reject messages that plausibly ARE language: at least two words and
    # this many letters. Keeps "2+2" and "ok" from being called unreadable.
    min_letters: int = 8
    unknown_reply: str = "Sorry, I didn't understand that. Could you say it again?"


class AppSectionConfig(BaseModel):
    """app: section — process-level basics."""

    name: str = "Veda"
    port: int = 8000
    max_history: int = 30
    # How many compressed earlier-session summaries go into every chat prompt. Each is ~100-250 tokens, and
    # prompt prefill dominates CPU latency, so small devices set this to 0-1.
    chat_summaries: int = 3
    # Persona system prompt for plain chat: "full" (persona.py, ~560 tokens) or "compact" (config/prompts/persona_chat.md,
    # ~250 tokens). The persona is re-read on every chat turn, so small CPU devices use the compact one.
    chat_persona: Literal["full", "compact"] = "full"
    log_level: str = "INFO"
    # docs/12: the model may set `advise` / `also_chat` in its control decision. False = both are ignored and every
    # turn behaves exactly as before.
    adaptive_reply: bool = False
    # docs/12 P7 (needs adaptive_reply too): `also_chat` runs the chat half beside the tool call.
    mixed_intent: bool = False
    # docs/12 P8: calls whose permission level is `auto` (lookups) run at the same time; everything else stays serial.
    parallel_tool_calls: bool = False
    multilingual: MultilingualConfig = MultilingualConfig()


class InferenceConfig(BaseModel):
    """inference: section — local LLM backend selection (ADR-001)."""

    backend: str = "ollama"  # "ollama" | "llama_cpp"
    ollama_host: str = "http://127.0.0.1:11434"
    model_alias: str = "qwen-wire"
    model_path: str | None = None  # required when backend == "llama_cpp"
    n_ctx: int = 4096
    n_threads: int | None = None
    num_batch: int | None = None  # Ollama prompt-eval batch size (default 512)
    # llama_cpp only: RAM (MB) for a cache of evaluated prompt prefixes, shared across chat / routing / tool-call
    # prompts. 0 = off. Each cached prefix of ~500 tokens costs roughly 70 MB.
    prompt_cache_mb: int = 0
    # llama_cpp only: memory-map the GGUF file. On CPUs where llama.cpp repacks the weights (ARM NEON/dotprod, AVX2)
    # the repacked copy lives in anonymous memory while the mapped file stays resident too, so the model is held twice
    # (measured on the Pi 5 with Qwen3-4B Q4_K_M: 5.5 GB RSS with mmap, 3.4 GB without, same speed).
    use_mmap: bool = True
    timeout: int = 120

    # Generation budget. `num_predict` is the strongest latency control on
    # CPU-only hardware — without it Ollama generates until the model
    # decides to stop, which a base model may never do.
    num_predict: int = 512
    temperature: float = 0.7
    keep_alive: str = "30m"

    # Reasoning-model thinking blocks. Measured on Qwen3-8B: 16x slower
    # with thinking on, for identical visible output. Ignored by models
    # that don't support it.
    think: bool = False

    # Shorter budgets for non-conversational calls — routing and
    # summarising need a handful of tokens, not a full reply.
    routing_num_predict: int = 64
    summary_num_predict: int = 256

    # Validate the model is importable at boot rather than 404-ing on the
    # first user turn. Disable only for tests with no Ollama running.
    verify_on_boot: bool = True
    warmup_on_boot: bool = True


class EmbeddingConfig(BaseModel):
    """embedding: section — local embedding model used for context retrieval."""

    enabled: bool = False
    model_id: str = "BAAI/bge-small-en-v1.5"
    model_path: str | None = None
    cache_dir: str | None = "data/embeddings"
    pooling: Literal["cls", "mean"] = "cls"  # "cls" for BGE; "mean" for all-MiniLM / e5 / gte style models
    max_tokens: int = 512        # the model's context limit; longer text is truncated
    top_k: int = 5
    min_score: float = 0.3  # minimum cosine similarity to inject a memory
    margin: float | None = None  # also keep hits within this of the best one (up to top_k); None = no margin rule
    max_hit_chars: int = 280     # each recalled memory is cut to this
    min_score_by_source: dict[str, float] = {}  # a stricter floor for one source, e.g. exchange: short notes score higher
    conversation_intent_soft_min: float = 0.76  # a looser match counts only instead of a clarifying question or a refusal
    conversation_intent_min: float | None = None  # "what are we discussing" is answered from the newest conversations (None = off)
    max_total_chars: int = 800   # and the whole recalled block to this (prompt tokens are Raspberry Pi prefill time)
    sources: list[str] = ["fact", "summary"]  # which memory sources recall searches


class VadConfig(BaseModel):
    """audio.vad: sub-section — voice activity detection tuning."""

    # "energy" needs only numpy and works everywhere (default).
    # "webrtc" is more precise but needs a C toolchain to build on 3.13+.
    engine: str = "energy"

    # webrtc only: 0-3, higher filters more aggressively.
    aggressiveness: int = 2
    # WebRTC accepts only 10/20/30 ms frames; energy accepts any.
    frame_duration_ms: int = 30

    # energy only: absolute floor, and how far above ambient counts as speech.
    min_rms: float = 300.0
    speech_ratio: float = 3.0
    noise_adapt: float = 0.05

    # Consecutive speech frames before an utterance opens — rejects clicks.
    start_frames: int = 3
    # Trailing silence that ends an utterance.
    silence_ms: int = 700
    # Hard stop so background noise can't hold the pipeline open forever.
    max_utterance_sec: float = 15.0
    # Frames kept before speech is confirmed, so the first syllable survives.
    pre_roll_frames: int = 5


class SpeakerIdConfig(BaseModel):
    """audio.speaker_id: sub-section — multi-speaker voice recognition.

    backend=resemblyzer (default): local PyTorch embedding model, no API
    key — see docs/voice/open-source-wake-speaker-design.md.
    backend=eagle: Picovoice Eagle, needs PICOVOICE_ACCESS_KEY; kept
    available, not installed/used by default.
    """

    enabled: bool = False
    backend: str = "resemblyzer"  # resemblyzer | eagle
    profiles_dir: str = "data/speaker_profiles"
    # Minimum similarity score to count an utterance as a known speaker.
    # Scale depends on backend — Eagle's own score and Resemblyzer's
    # cosine similarity are NOT the same scale; re-tune when switching.
    match_threshold: float = 0.6
    # When true, utterances from an unrecognized voice are dropped before
    # reaching the agent — the stray-talk filter half of this feature.
    # False by default: speaker ID only tags/personalizes, never blocks,
    # until profiles have actually been enrolled.
    require_known_speaker: bool = False
    # resemblyzer only: seconds of audio required before enroll_finish()
    # will accept a profile.
    min_enroll_seconds: float = 12.0
    # resemblyzer only: seconds of buffered audio per recognition window
    # (Resemblyzer has no per-frame API — see ResemblyzerSpeakerRecognizer).
    score_window_sec: float = 2.0


class AudioConfig(BaseModel):
    """audio: section — wake-word / STT / TTS / device selection."""

    wake_word_enabled: bool = True
    wake_engine: str = "openwakeword"  # none | hotkey | openwakeword | porcupine
    wake_hotkey: str = "<f9>"  # separate from the mute hotkey below
    wake_window_sec: float = 8.0
    wake_keyword_path: str | None = None  # .ppn file, porcupine only
    # openwakeword only. "Hey Veda" is a custom-trained model (see
    # docs/voice/open-source-wake-speaker-design.md §4.8) — not one of
    # openwakeword's bundled pretrained words — so wake_oww_model_path is
    # required in practice; there's no bundled fallback for this phrase.
    wake_oww_model: str = "hey_veda"
    wake_oww_model_path: str = "data/models/openwakeword/hey_veda.onnx"
    wake_threshold: float = 0.5
    daemon_trigger: str = "auto"  # "auto" | "wake_word" | "hotkey"
    hotkey: str = "<f8>"

    # Local STT model directory (faster-whisper format). No auto-download.
    stt_model: str = "base.en"
    stt_model_path: str | None = None
    # "faster_whisper" (in-process, laptop) | "whisper_cpp" (Pi; adapter lands in a later phase).
    stt_backend: str = "faster_whisper"
    # "auto" lets a multilingual model detect the language; a code pins it ("en", "hi", "te", "de", "ja", "kn").
    # An English-only (.en) model must stay pinned to "en".
    stt_language: str = "auto"
    # Short utterances misdetect. Below this duration the sticky hint (last successful language) is trusted.
    stt_auto_min_sec: float = 2.5
    stt_threads: int | None = None   # faster-whisper cpu_threads; None = library default
    stt_timeout: float = 120.0       # hard cap for one transcription, seconds
    # Per-utterance spectral gate before Whisper. Best for steady room noise.
    noise_suppression_enabled: bool = True
    noise_suppression_strength: float = Field(default=0.75, ge=0.0, le=1.0)

    tts_engine: str = "auto"  # "auto" (pyttsx3 on Windows, Piper elsewhere) | "pyttsx3" | "piper"
    tts_voice: str = "zira"
    tts_model_path: str | None = None  # required when tts_engine == "piper"; the default voice
    # language (code or English name) -> Piper voice .onnx path. A missing language speaks with the default voice.
    tts_voice_map: dict[str, str] = Field(default_factory=dict)

    mic_device_index: int | None = None
    speaker_device_index: int | None = None

    # Start the always-on loop at boot. The capture gate still governs
    # whether the mic actually opens, so this is safe to leave on.
    voice_enabled: bool = True
    speak_replies: bool = True
    # Ignore transcripts shorter than this — filters STT noise artefacts.
    min_transcript_chars: int = 2

    # Max wait for playback to finish before reopening the mic. A stuck audio
    # device must not wedge the loop forever.
    speak_timeout_sec: float = 60.0
    # Quiet gap after playback before the mic buffer is discarded, so the tail
    # of our own voice doesn't survive the drain.
    post_speak_settle_sec: float = 0.4
    # Drop transcripts that are mostly our own last reply (self-echo).
    echo_guard: bool = True

    vad: VadConfig = VadConfig()
    speaker_id: SpeakerIdConfig = SpeakerIdConfig()


class RoomSensorConfig(BaseModel):
    """sensing.room: the sensor in the user's own room."""

    enabled: bool = False
    driver: Literal["auto", "mock", "dht11"] = "mock"
    gpio_pin: int = Field(default=4, ge=0)
    poll_interval_sec: float = Field(default=300.0, gt=0)
    max_age_sec: float = Field(default=1800.0, gt=0)
    mock_temperature_c: float = 28.0   # what the mock driver reports (a laptop has no sensor)
    mock_humidity_pct: float = 55.0


class MusicConfig(BaseModel):
    """music: section. Plays a song by name through mpv, found with yt-dlp (docs/13). OFF by default."""

    enabled: bool = False
    max_minutes: float = Field(default=5.0, gt=0, le=60)   # a backstop: playback always stops after this long
    search_results: int = Field(default=5, ge=1, le=10)
    search_timeout_sec: float = Field(default=12.0, gt=0)
    mpv_path: str = "mpv"
    ytdlp_path: str = "yt-dlp"


class SensingConfig(BaseModel):
    """sensing: section — ambient-dispatch tuning (Supervisor / debounce / rate-limit)."""

    proactivity: str = "medium"  # "conservative" | "medium" | "chatty"
    debounce_window_sec: float = 30.0
    rate_limit_max: int = 6
    rate_limit_window_sec: float = 60.0
    room: RoomSensorConfig = RoomSensorConfig()


class RemindersConfig(BaseModel):
    """reminders: section — spoken reminders for calendar events and due tasks (no model involved)."""

    enabled: bool = True
    lead_minutes: list[int] = [15]
    at_start: bool = True
    skip_all_day: bool = True
    include_tasks: bool = True
    poll_sec: float = 300.0
    horizon_hours: float = 24.0
    tick_sec: float = 5.0
    idle_settle_sec: float = 1.5

    @field_validator("lead_minutes")
    @classmethod
    def _valid_leads(cls, v: list[int]) -> list[int]:
        from domain.policies.reminder_policy import validate_leads
        return list(validate_leads(v))

    @field_validator("poll_sec", "horizon_hours", "tick_sec")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class OnlineConfig(BaseModel):
    """privacy.online: sub-section — the network egress allow-list."""

    enabled: bool = True  # False = fully offline: no online tools are registered
    allowlist: list[str] = [
        "api.open-meteo.com", "geocoding-api.open-meteo.com", "api.frankfurter.dev", "open.er-api.com",
        "api.tavily.com", "oauth2.googleapis.com", "gmail.googleapis.com", "www.googleapis.com",
    ]
    default_place: str = "Hyderabad"  # used when the user asks for weather without naming a place
    search_api_key_env: str = "TAVILY_API_KEY"  # env var holding the Tavily key (never stored in config)
    google_client_id_env: str = "GOOGLE_API_CLIENT_ID"   # env vars holding the Google OAuth secrets (never stored in config)
    google_client_secret_env: str = "GOOGLE_CLIENT_SECRET"
    google_refresh_token_env: str = "GOOGLE_REFRESH_TOKEN"


class PrivacyConfig(BaseModel):
    """privacy: section — mute/indicator hardware + network egress."""

    mute_switch: str = "software"  # software | keyboard | hid | gpio
    start_muted: bool = True  # boot muted; explicit opt-in required to listen
    mute_gpio_pin: int = 17
    mute_gpio_muted_level: str = "low"  # pin level that means MUTED (low | high)
    indicator_gpio_pin: int = 27
    online: OnlineConfig = OnlineConfig()


class GovernanceAuditConfig(BaseModel):
    """governance.audit: sub-section."""

    backend: str = "sqlite"  # "sqlite" | "null"
    db_path: str = "data/governance_audit.db"


class GovernanceConfig(BaseModel):
    """governance: section — plug-and-play policy enforcement."""

    enabled: bool = False
    provider: str = "builtin"  # "builtin" | "null" ("agt" is a hard reject)
    policies_dir: str = "tpa/governance/policies"
    audit: GovernanceAuditConfig = GovernanceAuditConfig()
    circuit_breaker_threshold: int = 3
    circuit_breaker_timeout: float = 60.0


class PermissionsConfig(BaseModel):
    default_level: str = "notify"
    approval_timeout: int = 60


class ValidatorsConfig(BaseModel):
    block_pii_in_output: bool = True
    block_credentials_in_output: bool = True
    max_input_length: int = 10000
    max_output_length: int = 50000


class RateLimitingConfig(BaseModel):
    enabled: bool = True
    global_rpm: int = 30
    per_skill_rpm: int = 10


class AuditConfig(BaseModel):
    enabled: bool = True
    log_dir: str = "data/audit"
    log_inputs: bool = True
    log_outputs: bool = True
    retention_days: int = 30


class GuardrailsConfig(BaseModel):
    """guardrails: section."""

    permissions: PermissionsConfig = PermissionsConfig()
    validators: ValidatorsConfig = ValidatorsConfig()
    rate_limiting: RateLimitingConfig = RateLimitingConfig()
    audit: AuditConfig = AuditConfig()


class AgentEntry(BaseModel):
    """Single agent definition under agents: section."""

    enabled: bool = True
    # None = inherit inference.model_alias. Per-agent models only make sense
    # once more than one model is imported; ADR-008 specifies a single SLM.
    model: str | None = None
    description: str = ""
    skills: list[str] = []


class AgentsConfig(BaseModel):
    """agents: section. Only agents that actually exist in this build."""

    supervisor: AgentEntry = AgentEntry(description="Routes requests to specialist agents")
    responder: AgentEntry = AgentEntry(description="General conversation, Q&A, brainstorming")
    system: AgentEntry = AgentEntry(
        description="Device actions: open/close apps, volume, running processes",
        skills=["terminal", "file_ops"],
    )
    tasks: AgentEntry = AgentEntry(description="The user's to-do list: add, list, complete, delete tasks")
    lookup: AgentEntry = AgentEntry(description="Live data from the web: weather, currency rates, news, current facts")
    memory: AgentEntry = AgentEntry(description="Saves lasting facts the user states about themselves: favourites, allergies, names, routines")
    session_ttl_sec: int = 900  # session working state older than this is ignored (docs/10)


class SkillEntry(BaseModel):
    """Single skill definition under skills: section."""

    enabled: bool = True
    permission_level: str = "approve"
    extra: dict[str, Any] = {}


class SkillsConfig(BaseModel):
    """skills: section. Only skills that actually exist in this build."""

    terminal: SkillEntry = SkillEntry(
        permission_level="approve",
        extra={
            "working_directory": "~",
            "timeout": 30,
            "blocked_commands": ["rm -rf /", "format", "del /s /q", "shutdown", ":(){:|:&};:"],
            "blocked_patterns": [r"rm\s+-rf\s+/"],
        },
    )
    file_ops: SkillEntry = SkillEntry(
        permission_level="notify",
        extra={
            "allowed_paths": ["~", "."],
            "blocked_paths": ["C:\\Windows", "C:\\Program Files", "/etc", "/usr"],
        },
    )
    tasks: SkillEntry = SkillEntry(permission_level="notify")
    custom_skills_dir: str = "data/custom_skills"


class PromptingConfig(BaseModel):
    """prompting: section (config/prompting.yaml). Tunable prompt budgets; a missing key keeps the dataclass default."""

    budgets: dict[str, int] = {}
    control_history_exchanges: int = Field(default=2, ge=0, le=6)
    turn_chars: int = Field(default=200, ge=40)
    user_message_chars: int = Field(default=500, ge=40)

    @field_validator("budgets")
    @classmethod
    def _known_positive_budgets(cls, budgets: dict[str, int]) -> dict[str, int]:
        from dataclasses import fields

        from domain.policies.token_budget_policy import PromptBudgets

        known = {f.name for f in fields(PromptBudgets)}
        unknown = sorted(set(budgets) - known)
        if unknown:
            raise ValueError(f"unknown prompt budget(s) {unknown}; valid: {sorted(known)}")
        bad = sorted(k for k, v in budgets.items() if v < 0)
        if bad:
            raise ValueError(f"prompt budget(s) must be >= 0: {bad}")
        return budgets


class DevicesConfig(BaseModel):
    """devices: section - phone pairing and the LAN access gate (docs/15-vision-mobile-design.md section 9)."""

    enabled: bool = True   # False = no phone can ever authenticate; non-loopback requests are all refused
    lan_enabled: bool = False   # default bind of `veda server` (True = 0.0.0.0). Binding never grants access: tokens do
    advertise_host: str | None = None   # host/IP put in the pairing QR; None = detect the LAN address
    pairing_code_ttl_s: int = 120
    pairing_max_attempts: int = 5   # wrong codes per source address before it is locked out
    pairing_lock_s: int = 300
    pairing_global_per_hour: int = 20   # redemption attempts per hour across all addresses
    default_scopes: list[str] = ["vision", "chat", "control"]

    @field_validator("default_scopes")
    @classmethod
    def _known_scopes(cls, v: list[str]) -> list[str]:
        bad = sorted(set(v) - {"vision", "chat", "control", "approve"})
        if bad:
            raise ValueError(f"unknown scope(s) {bad}")
        return v

    @field_validator("pairing_code_ttl_s", "pairing_max_attempts", "pairing_lock_s", "pairing_global_per_hour")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v


class VisionConfig(BaseModel):
    """vision: section - phone camera to a local vision model (docs/15-vision-mobile-design.md section 10)."""

    enabled: bool = False
    binary_path: str | None = None   # llama-server; per machine, set in config/hardware/<device>.yaml
    model_path: str = "data/models/vision/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    mmproj_path: str = "data/models/vision/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    model_sha256: str | None = None
    mmproj_sha256: str | None = None
    host: str = "127.0.0.1"
    port: int = 8081
    n_ctx: int = 4096
    n_threads: int | None = None
    # false = `--load-mode none`: read the model into RAM instead of mapping it, so llama.cpp's repacked copy is the only
    # one resident (measured on the Pi 5: 3.3 GB -> 2.5 GB after an image, same speed). Needs a llama-server new enough
    # to have --load-mode (b11716 does; older builds spelled it --no-mmap).
    use_mmap: bool = True
    image_max_tokens: int = 512
    startup_timeout_s: int = 120
    image_max_edge_px: int = 1024
    upload_max_bytes: int = 1_048_576
    image_cache_max: int = 3
    image_cache_ttl_s: int = 300
    idle_unload_seconds: int = 300
    min_headroom_mb: int = 3000
    request_timeout_s: int = 120
    max_new_tokens: int = 200
    retain_images: bool = False

    @field_validator("host")
    @classmethod
    def _loopback_only(cls, v: str) -> str:
        import ipaddress
        try:
            ok = ipaddress.ip_address(v).is_loopback
        except ValueError:
            ok = False   # a hostname could resolve anywhere
        if not ok:
            raise ValueError("vision.host must be a loopback IP address (127.0.0.1 or ::1): the model server is never exposed")
        return v

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("must be 1-65535")
        return v

    @field_validator(
        "n_ctx", "image_max_tokens", "startup_timeout_s", "image_max_edge_px", "upload_max_bytes", "image_cache_max",
        "image_cache_ttl_s", "idle_unload_seconds", "request_timeout_s", "max_new_tokens",
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @field_validator("min_headroom_mb")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("n_threads")
    @classmethod
    def _threads(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("must be >= 1 (or null for the default)")
        return v


class AppConfig(BaseModel):
    """Complete application configuration combining all sections."""

    app: AppSectionConfig = AppSectionConfig()
    inference: InferenceConfig = InferenceConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    audio: AudioConfig = AudioConfig()
    sensing: SensingConfig = SensingConfig()
    reminders: RemindersConfig = RemindersConfig()
    music: MusicConfig = MusicConfig()
    privacy: PrivacyConfig = PrivacyConfig()
    governance: GovernanceConfig = GovernanceConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    agents: AgentsConfig = AgentsConfig()
    prompting: PromptingConfig = PromptingConfig()
    skills: SkillsConfig = SkillsConfig()
    devices: DevicesConfig = DevicesConfig()
    vision: VisionConfig = VisionConfig()