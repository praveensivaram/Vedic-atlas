# Configuration, Constants, and Logging

## Config model: YAML-per-section, not a single file or env-var-driven

`src/core/config.py` loads configuration from **10 separate YAML files**
under `config/`, one per top-level section — not a single monolithic file,
and not primarily env-var-driven (env vars are reserved for secrets, e.g.
`PORCUPINE_ACCESS_KEY`, read directly where needed rather than flowing
through this config system).

```python
_SECTION_FILES = ("app", "inference", "embedding", "audio", "sensing",
                   "privacy", "governance", "guardrails", "agents", "skills")
```

Each section maps to `config/{name}.yaml`. `_read_section(name)` reads via
`yaml.safe_load`, returning `{}` if the file is absent — missing files
aren't an error, they just mean "use the Pydantic defaults for this
section." `load_full_config()` is memoized in a module-level
`_full_config_cache` (call `reload_full_config()` to force a re-read).
Empty/falsy values in a loaded section are filtered out before construction
so they don't override the Pydantic field defaults with `None`:

```python
AppConfig(**{k: v for k, v in raw.items() if v})
```

`ensure_dirs()` creates `DATA_DIR`, `TEMP_DIR`, `AUDIT_DIR` on boot.

**Note**: `core/config.py` imports `yaml` at module level, so `pyyaml` is a
hard runtime dependency of config loading — but it is not listed in
`pyproject.toml`'s dependencies (neither are `sqlalchemy` or `psutil`,
both also required at runtime). See the setup notes in
[00-overview.md](00-overview.md) if you're installing from scratch.

## `AppConfig` — the full schema (`schemas/config_schemas.py`)

| Section | Key fields (with defaults) |
|---|---|
| `app: AppSectionConfig` | `name="Veda"`, `port=8000`, `max_history=30`, `log_level="INFO"` |
| `inference: InferenceConfig` | `backend="ollama"`, `ollama_host`, `model_alias="qwen-wire"`, `model_path=None`, `n_ctx=4096`, `n_threads=None`, `num_batch=None`, `timeout=120`, `num_predict=512`, `temperature=0.7`, `keep_alive="30m"`, `think=False`, `routing_num_predict=64`, `summary_num_predict=256`, `verify_on_boot=True`, `warmup_on_boot=True` |
| `embedding: EmbeddingConfig` | `enabled=False`, `model_id="BAAI/bge-small-en-v1.5"`, `model_path=None`, `cache_dir="data/embeddings"`, `top_k=5`, `min_score=0.3`, `sources=["fact","summary"]` (**config/embedding.yaml sets `enabled=true`, `model_path=data/bge-small-en-v1.5`, `top_k=3`, `min_score=0.5`**; `cache_dir` is no longer used) |
| `audio: AudioConfig` | `wake_word_enabled=True`, `wake_engine="openwakeword"` (**config/audio.yaml sets `none` for v1**: no wake word, everything is transcribed while unmuted), `wake_hotkey="<f9>"`, `wake_window_sec=8.0`, `wake_keyword_path=None`, `wake_oww_model="hey_veda"`, `wake_oww_model_path="data/models/openwakeword/hey_veda.onnx"`, `wake_threshold=0.5`, `daemon_trigger="auto"`, `hotkey="<f8>"` (mute hotkey — separate binding from the wake hotkey), `stt_model="base.en"` (config/audio.yaml sets `small`), `stt_model_path=None` (config points at `data/models/whisper-small`), `stt_backend="faster_whisper"` (`whisper_cpp` until the Pi adapter lands), `stt_language="auto"` (detect the language; clips shorter than `stt_auto_min_sec` reuse the last confidently detected language; switch at runtime with `/language en|hi|...|auto`; an `.en` model is pinned to `en`), `stt_auto_min_sec=2.5`, `stt_threads=None`, `stt_timeout=120.0`, `tts_engine="auto"` (`auto` = `pyttsx3` on Windows, `piper` on Linux/Pi; the repo sets `piper`)`, `tts_voice="zira"` (pyttsx3 hint), `tts_model_path` (default Piper voice), `tts_voice_map` (language code -> Piper `.onnx`; missing language falls back to the default voice; a missing voice or package stops voice at startup with a clear error), `mic_device_index=None`, `speaker_device_index=None`, `voice_enabled=True`, `speak_replies=True`, `min_transcript_chars=2`, `speak_timeout_sec=60.0`, `post_speak_settle_sec=0.4`, `echo_guard=True`, `vad: VadConfig` |
| `audio.vad: VadConfig` | `engine="energy"`, `aggressiveness=2`, `frame_duration_ms=30`, `min_rms=300.0`, `speech_ratio=3.0`, `noise_adapt=0.05`, `start_frames=3`, `silence_ms=700`, `max_utterance_sec=15.0`, `pre_roll_frames=5` |
| `audio.speaker_id: SpeakerIdConfig` | `enabled=False`, `backend="resemblyzer"`, `profiles_dir="data/speaker_profiles"`, `match_threshold=0.6`, `require_known_speaker=False`, `min_enroll_seconds=12.0`, `score_window_sec=2.0` — multi-speaker voice recognition; off and purely additive by default. `backend="eagle"` (Picovoice) remains available — see `docs/voice/open-source-wake-speaker-design.md` |
| `sensing: SensingConfig` | `proactivity="medium"`, `debounce_window_sec=30.0`, `rate_limit_max=6`, `rate_limit_window_sec=60.0` |
| `audio.wake_*` | `wake_word_enabled=True`, `wake_engine` (`openwakeword` \| `none` \| `hotkey` \| `porcupine`; **config/audio.yaml sets `openwakeword`**), `wake_oww_model="hey_veda"`, `wake_oww_model_path=data/models/openwakeword/hey_veda.onnx`, `wake_threshold=0.5` (tune with `python scripts/wake_test.py`), `wake_window_sec=8.0`. A configured engine that fails to start disables voice instead of listening continuously |
| `privacy: PrivacyConfig` | `mute_switch="software"`, `start_muted=True`, `mute_gpio_pin=17`, `indicator_gpio_pin=27`, `online: OnlineConfig` |
| `privacy.online: OnlineConfig` | `enabled=True` (False = fully offline, no online tools), `allowlist=[open-meteo, geocoding-api.open-meteo, api.frankfurter.dev, open.er-api.com, api.tavily.com]`, `default_place="Hyderabad"`, `search_api_key_env="TAVILY_API_KEY"` (the *name* of the env var; the key lives in `.env`) |
| `governance: GovernanceConfig` | `enabled=False`, `provider="builtin"`, `policies_dir="tpa/governance/policies"`, `audit: GovernanceAuditConfig`, `circuit_breaker_threshold=3`, `circuit_breaker_timeout=60.0` |
| `governance.audit: GovernanceAuditConfig` | `backend="sqlite"`, `db_path="data/governance_audit.db"` |
| `guardrails: GuardrailsConfig` | `permissions: PermissionsConfig`, `validators: ValidatorsConfig`, `rate_limiting: RateLimitingConfig`, `audit: AuditConfig` |
| `guardrails.permissions` | `default_level="notify"`, `approval_timeout=60` |
| `guardrails.validators` | `block_pii_in_output=True`, `block_credentials_in_output=True`, `max_input_length=10000`, `max_output_length=50000` |
| `guardrails.rate_limiting` | `enabled=True`, `global_rpm=30`, `per_skill_rpm=10` |
| `guardrails.audit` | `enabled=True`, `log_dir="data/audit"`, `log_inputs=True`, `log_outputs=True`, `retention_days=30` |
| `agents: AgentsConfig` | `supervisor/responder/system: AgentEntry` (each `enabled=True`, `model=None` → inherits `inference.model_alias`, `description=""`, `skills=[]`), `tasks` / `lookup` / `memory: AgentEntry` (descriptions only), `session_ttl_sec=900` (session working state older than this is ignored). The `llm_routing`, `tools_enabled` and `orchestrator_enabled` flags no longer exist: there is no routing and the orchestrator is the only path |
| `prompting: PromptingConfig` (config/prompting.yaml) | `budgets` (per-stage prompt token budgets; a missing key keeps the `PromptBudgets` default: `control=3000`, `narrate=500`, `chat=1500`, `observation_max=320`, `chat_history_turns=8`), `control_history_exchanges=2` (complete user+reply exchanges shown to the control decode), `turn_chars=200`, `user_message_chars=500` |
| `skills: SkillsConfig` | `terminal` (`permission_level="approve"`, extra: `working_directory`, `timeout`, `blocked_commands`, `blocked_patterns`), `file_ops` (`permission_level="notify"`, extra: `allowed_paths`, `blocked_paths`), `tasks` (`permission_level="notify"`), `custom_skills_dir="data/custom_skills"` |
| `app.multilingual: MultilingualConfig` | `enabled=true`, `reject_unknown=true`, `min_letters=8`, `unknown_reply="Sorry, I didn't understand that. Could you say it again?"` — see "Multilingual replies" below |

## Multilingual replies

Every turn names its language once, in the control decode. `ControlDecision.language` carries the
model's own label (`"German"`, `"Telugu"`, …) or `"unknown"`; `domain/policies/language_policy.py`
overrides it for the scripts it reads exactly (Telugu, Devanagari/Hindi) and for German, so those three
can never be mislabelled.

The name is then stated explicitly in each reply prompt (`The user wrote in German. Write your entire
reply in German.`) — far more reliable on a small model than the abstract "reply in the user's language"
rule. The same line goes into the narrate stage, so `reply_mode: llm` tool results are phrased in the
user's language rather than the provider's English.

Tools with `reply_mode: template` (weather, tasks, currency, clock, calendar, gmail, …) still build their
sentence in English inside the adapter — deliberately, so the tool's own numbers and names are spoken
verbatim. When the turn's language is not English, that finished sentence is translated once at reply
assembly (`assistant_orchestrator._localize` → `translate_stage`). The translate prompt carries only the
target language and the sentence — never the user's question, which a small model otherwise repeats
instead of translating. A translation is rejected and the English sentence kept when its numbers are not
in the source sentence (checked with `is_grounded` against the sentence alone) or when it is an echo of
the user's message (`domain/policies/echo_policy.py`, similarity ≥ 0.85). English turns pay no extra cost.

When `app.multilingual.reject_unknown` is on and the model names no readable language for a message that
plausibly is language (≥ 2 words, ≥ `min_letters` letters), the turn answers `unknown_reply` without a
chat completion. Tool and clarification routes are never rejected: picking a tool or asking a question is
proof the message was understood.

Plain chat has its own echo backstop (`domain/policies/echo_policy.py`). The first sentence is held before
it is shown; if it is the user's own message (normalised similarity ≥ 0.85), the stream is cancelled and
the model is asked once more with an explicit "do not repeat the user" instruction. If it echoes again the
turn answers a fixed sentence in the turn's language (`fallback_reply`, English for languages it does not
cover) — the user's own words are never shown back.

Limits: this covers text and the LLM side of voice. `audio.stt_model/…` is still an English-only Whisper
checkpoint and `audio.tts_voice` a single SAPI voice, so spoken input in other languages is transcribed /
spoken with English models until those two are swapped (see docs/voice/).


The repo's actual `config/*.yaml` files override a handful of these
defaults for the demo profile — notably `inference.yaml` sets
`model_alias: qwen3-4b` (not the schema default `qwen-wire`) and
`model_path: data/Qwen3-4B-Q4_K_M.gguf`, with a comment reminding you to
run `ollama create qwen3-4b -f data/models/Modelfile.qwen3-4b` before boot
if using that model.

**Picovoice access key (opt-in backends only)**: the project's defaults
(`audio.wake_engine: openwakeword`, `audio.speaker_id.backend: resemblyzer`)
need no Picovoice key at all — see
`docs/voice/open-source-wake-speaker-design.md` for why. The key is only
needed if you explicitly switch `audio.wake_engine: porcupine` or
`audio.speaker_id.backend: eagle`, read from the `PICOVOICE_ACCESS_KEY`
env var (falling back to the older `PORCUPINE_ACCESS_KEY` name for
compatibility) — Picovoice issues one key valid for every one of their
SDKs, so a single key covers both features if you ever opt into them.
This is the one place in the config system where an env var — not a YAML
file — is the source of truth, consistent with the "env vars are for
secrets" split `core/config.py` otherwise maintains.

## Constants (`core/constants.py`)

```python
PROJECT_NAME = "Veda"
VERSION = "0.1.0"
API_PREFIX = "/api"
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # repo root
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = DATA_DIR / "temp"
MODEL_DIR = DATA_DIR / "models"
AUDIT_DIR = DATA_DIR / "audit"
CONFIG_DIR = PROJECT_ROOT / "config"
```

Every model path, DB path, and log path elsewhere in the codebase is
resolved relative to `PROJECT_ROOT`/`DATA_DIR`, not the process's current
working directory — which is why the app (`bootstrap.app:app`) must be run with `src/`
on `PYTHONPATH` (see [04-controller-layer.md](04-controller-layer.md#running-it))
rather than relying on relative-path assumptions.

## Logging (`core/logging_config.py`)

Request-scoped IDs are propagated via a `contextvars.ContextVar`
(`_request_id`, default `"-"`), not a mutable instance attribute — the
docstring explicitly calls out that this replaces a donor concurrency bug
where a shared mutable attribute raced under asyncio. `set_request_id(id)`
sets it (called by the `request_context` middleware, see
[04-controller-layer.md](04-controller-layer.md)); a `_RequestIdFilter`
injects it into every `LogRecord` as `record.identifier`.

`configure_logging(level="INFO")` builds a logger named `"veda"` with this
format:

```
[%(asctime)s][%(levelname)-7s][%(identifier)s][%(funcName)s][%(module)s:%(filename)s].(%(lineno)d)] : %(message)s
```

A module-level `logger = configure_logging()` default instance exists so
`from core.logging_config import logger` works anywhere before
`bootstrap/lifespan.py`'s `lifespan()` calls `configure_logging(cfg.app.log_level)`
with the real configured level at boot — every module in the codebase logs
through this one shared logger instance.

## Enums (`core/enums.py`)

Besides `ExceptionCode`/`ErrorMessage` (covered in
[05-exception-handling.md](05-exception-handling.md)):

- `SkillCategory(str, Enum)`: `TERMINAL`, `FILE`, `IDE`, `SEARCH`,
  `COMMUNICATION`, `CUSTOM`.
- `HookEvent(str, Enum)`: `PRE_SKILL`, `POST_SKILL`, `ON_ERROR`,
  `ON_AGENT_START`, `ON_AGENT_COMPLETE`, `ON_APPROVAL_NEEDED` — consumed by
  `service/skills/skill_runner.py` and `service/hooks/`.

All enums in this file override `__repr__`/`__str__` to return `.value`.

`inference.prompt_cache_mb` (llama_cpp only, schema default `0`, **config/inference.yaml sets `768`**): RAM for a cache of evaluated prompt prefixes shared across chat, control-decode and content prompts. Without it every call re-reads its whole prompt because the prompts alternate; measured warm: chat 14 s -> 2.4 s, routing 5.5 s -> 2.4 s, tool decision 10 s -> 4.2 s. About 70 MB per cached ~500-token prefix; use `0` on a small device.

## Switching the STT language

`stt_language: auto` detects each utterance and keeps a sticky hint from the last confident detection, used for
clips shorter than `stt_auto_min_sec` (2.5 s by default) where Whisper's language ID is unreliable. A single
language can be pinned at runtime with `/language en|hi|te|de|ja` (or `auto`) in the CLI; the same switch is
`POST /api/voice/language`. The current value is reported by `/api/voice/status` and `/language` with no argument.

## Latin tool arguments

A tool parameter can declare `format: latin` (`config/tools/weather.yaml`, `weather_forecast.yaml`). The
control prompt asks the model to transliterate names (`???????? -> Hyderabad`), and
`domain/policies/argument_policy.py` repairs a non-compliant decode before the call runs:
`place_names.to_latin()` first applies a spelling fix for the names Whisper mangles, then `unidecode` for any
other non-Latin name. A value that still cannot be resolved is left for the tool and ends in a clarification,
not a guessed city. Parameters without the flag (e.g. a task title) stay in the user's language.

## Multilingual speech output

`audio.tts_engine: piper` with `audio.tts_voice_map` selects the voice from the turn's reply language
(`en`, `de`, `hi`, `te`, `ja`; Kannada has no Piper voice and falls back to the default). The staged voices
live in `data/models/piper/`: `en_US-lessac-medium`, `de_DE-thorsten-medium`, `hi_IN-priyamvada-medium`,
`te_IN-padmavathi-medium`, `ja_JP-hi_fi_captain-medium` (each `.onnx` + `.onnx.json`). Voices load lazily and
are cached; Piper runs fully offline. `pyttsx3` remains the fallback when a Piper voice/package is missing.
