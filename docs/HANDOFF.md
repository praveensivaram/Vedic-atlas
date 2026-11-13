# HANDOFF — unified control decode (current state, 2026-10-08)

**Branch:** `feature/unified-orchestrator` (from `release/v1.1` @ `1771944`). Not merged. Read `docs/10-orchestrator-review-and-plan.md` §6d-§6i for the measurements and decisions.

## What exists now
Every user turn goes `SupervisorAgent` -> `AssistantOrchestrator`: ONE grammar-constrained control decode (`{needs_live_data, calls[<=3]}` + optional `clarification`) -> `dispatch_policy` -> tools through `SkillRunner` -> template replies verbatim (no second decode), or one content decode, or the chat `ResponderAgent`. There is **no routing, no router model, no keyword/`required_when` forcing, no `SystemAgent`, no `ToolAgent`, and no feature flag**: rollback is `git revert` of the Phase 3 commit (`9d53bb6`). Design: `docs/unified-assistant-orchestrator.md`; step list: `docs/implementation_guide.md`; architecture docs `docs/00`-`08` are updated to this state.

| Piece | Status |
|---|---|
| Phases 0-6 of the guide | done (see below for what is NOT proven) |
| Memory / recall | Chat sees: saved facts, the last 8 turns of the CURRENT session (summarised or not), and recalled older summaries. Recall (`service/memory/semantic_recall.py`, `config/embedding.yaml`): top 4 by cosine, floor 0.62 (questions about the past score >= 0.66, ordinary ones ~0.56), keep hits within 0.05 of the best, each cut to 280 chars, block capped at 800 chars (~200 tokens), dated, newest first; a short follow-up that finds nothing is retried with the previous question. **"What are we discussing?" mode:** a question that is nearly one of nine example phrasings (bge-small cosine >= 0.82; unseen paraphrases 0.82-1.0, specific questions <= 0.82, ordinary <= 0.77) always goes to chat (`AssistantOrchestrator(about_conversation=...)` overrides whatever the router chose, which used to be a clarifying question or a web search) and is answered from the newest earlier conversations by date (`ConversationManager.recent_summaries_block`, ignores the profile's `chat_summaries: 0`) instead of similarity. Summaries are written only after the 30-minute session gap (or 50 turns plus a 2-minute pause); `scripts/backfill_summaries.py` rebuilds lost ones (dry run by default). Measured on 12 self-written questions over 19 summaries: right summary in the block 10/12 (was 9/12), ordinary questions that wrongly got memory 4/16 (was 9/16 on the old profile, 15/16 on base). **Per-exchange index:** every saved question-and-answer is embedded as a short note (`You: ... | Veda: ...`, the words actually said, no model call; `domain/policies/exchange_note_policy.py`) in the background right after the reply (`ConversationManager.add_turn` hook) and searched together with the summaries; exchanges need their own floor (0.70, `embedding.min_score_by_source`) because short notes score higher; a hit already shown in the recent history is not repeated; at startup `_index_exchanges` indexes what is missing and drops vectors whose turn the 30-day cleanup deleted. Measured on a copy of the real data (14 detail questions): answer in the prompt 13/14 (summaries alone 12/14), ordinary questions that wrongly got memory 4/16 (those 4 are genuine repeats of an earlier question). Known gaps: interrupted replies are not saved; no memory benchmark yet. |
| Reminders | Spoken heads-up (15 min, configurable) and at-start reminders for calendar events and due tasks; a timer, no model; never while Veda speaks or you talk; speaks even when muted; text also printed in the REPL. `docs/11-reminders.md`, `config/reminders.yaml`. Needs the Google sign-in for the calendar part (tasks work without). |
| Tools | `get_weather`, `get_weather_forecast`, `convert_currency`, `web_search`, `tasks`, `remember`, `app_control`, `volume_control`, `device_status`, `gmail_search`, `gmail_read`, `gmail_draft`, `gmail_send`, `calendar_agenda`, `calendar_create`, `current_time` (terminal/file_ops are NOT model-callable, by decision). Gmail/Calendar: `docs/08-tool-harness.md`; link the account with `python scripts/google_auth.py`; **accuracy with the Google tools, measured 2026-10-08 (A1, temp 0.1):** golden 45/57 before and 46/57 after adding `calendar_create` + `current_time` + the prompt changes; 12/13 new cases (calendar writes, clock, sent mail, search follow-ups) pass; held-out 36/46 (A5), of which the 6 new cases all pass and the 40 original cases score 30/40 (the 28/40 above predates the Google tools). Known flip: "should I bring an umbrella?" now picks `get_weather` instead of `get_weather_forecast`. The model still invents a time when none is said, so `calendar_create` trusts a time only if the user's own words contain one (`calendar_policy.message_states_time`); the other write guards are listed under "Write guards" in `docs/08-tool-harness.md` (static prompt now ~2,597 of the 2,600 the tests allow, after dropping duplicate examples to fit `calendar_create` and `current_time`: still no room for another tool without trimming; re-run `python scripts/google_auth.py` once to grant `calendar.events` before `calendar_create` can write) |
| Data plane | `PromptComposer.content_stage(task, document, ...)`, one shared prefix; a `returns: document` result can never reach a control prompt |
| Session state | `session_contexts` table, keyed by tool name, TTL 900 s, never from a destructive call |
| Safety | destructive call must have its target named by the user (pronouns/ordinals/filler don't count); claims backstop on every reply path; unparseable decision fails closed |
| Tests | `pytest -q`: 392 pass; 5 `test_tts_selection` failures only on a machine without pyttsx3/piper |

## What is NOT done / not proven (the real remaining work)
1. **Accuracy gate not cleared.** Qwen3-4B Q4, temp 0: tuned golden set 84% (43/51), held-out set 70% (28/40, was 62% before the optional-clarification change and the wider target veto). Misses: trivia sent to `web_search`; private data ("my meetings", "my messages") mapped onto `tasks`; no-context follow-ups guess a topic/amount; two-intent messages drop the second call (0/4); `the first one` after a clarification (now vetoed -> asks). Options: larger control model (not viable on the Pi), accept/mitigate false calls (most are harmless reads), fix the harmful classes. **Never tune prompts against `tests/eval/orchestrator_heldout.yaml`**; replace burned cases with fresh phrasings.
2. **Latency.** Decision median ~14.6 s / p95 ~24 s on the dev laptop, decode-bound (~0.35 s per output token). The last change (optional `clarification`, `30b82a2`) should save ~2 s and is **unmeasured**; verify llama.cpp accepts the optional key, then re-run both evals and compare with 43/51, 25/40 and the median.
3. **Pi:** nothing measured on the device (latency, prefix-cache hits, `prompt_cache_mb`). Small profiles now use `n_ctx: 3072` + `prompting.budgets.control: 2800` (a test enforces they fit); quality of 0.6B/1.7B on this prompt is unknown.
4. Known gaps: `SkillRunner` permission is per skill, not per argument (`app_control close` cannot need more than `open`); the `pending` clarification state is stored in the table but not rendered (prose history carries it; not needed so far); manifest fields `triggers`/`required_when` have no reader (kept for a possible Plan B); `tpa/governance/policies/agent_routing.yaml` governs the deleted `route_to_agent` action; Gmail/Calendar (guide Phase 7) is intentionally not started.

## How to run things
```bash
pytest -q                                   # no model needed; includes an isolated bootstrap smoke test
python scripts/spike_decision_protocol.py --variants A5                                          # tuned set, REAL model
python scripts/spike_decision_protocol.py --variants A5 --golden tests/eval/orchestrator_heldout.yaml
python scripts/spike_decision_protocol.py --rejudge tests/eval/results/<run>.json [--golden ...] [--phase4]   # re-score saved output, no model
```
Needs `data/Qwen3-4B-Q4_K_M.gguf` (gitignored; path in `config/inference.yaml`) and `llama-cpp-python`. The venv is not in git: `python -m venv .venv && pip install -e .`.

## Rules for the next person
- **Never `rm -rf` the gitignored `data/` (models, `veda.db`, saved facts).** An earlier session did, after a smoke test, and lost the user's DB and saved facts (models were restored from copies). Use `tests/test_bootstrap_smoke.py`'s approach (a temp copy of `src/` + `config/`) for anything that boots the app.
- Layering `controller -> service -> domain <- tpa`; `domain/` pure; concrete adapters only in `src/bootstrap/`; errors via `AppException` + `ExceptionCode` (add a row in `exceptions/handlers.py::_STATUS_BY_CODE`).
- The Bash tool here eats backslashes in heredocs/`python - <<EOF` (`\n` becomes a newline). Write scripts with the Write tool and run them as files; use Edit for anything containing regexes or `\n`.
- `docs/` and `data/` are gitignored; docs here were force-added (`git add -f docs/...`).
- Do not commit: `.idea/`, `vedic-atlas-env/`, `wakeword_training/`, `src/*.egg-info/`, `tests/eval/results/routing-v1-prompt-*.json`.
- Commit trailer: `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.

## Setting up on another laptop (what git does NOT carry)
```bash
git clone https://github.com/praveensivaram/Vedic-atlas.git vedic-atlas && cd vedic-atlas
python -m venv .venv && . .venv/Scripts/activate   # Linux/Pi: source .venv/bin/activate, and run scripts/setup_pi.sh instead of the pip line
pip install -e .                                   # llama-cpp-python compiles from source where no wheel exists (10-30 min on a Pi)
pip install pytest pytest-asyncio                  # not in pyproject
pytest -q                                          # expect all green except test_tts_selection (needs pyttsx3 on Windows / piper elsewhere)
```
**Copy these by hand into `data/` (gitignored; none of it is in the repo):**
| File | Path | Size | Notes |
|---|---|---|---|
| Qwen3-4B Q4_K_M GGUF | `data/Qwen3-4B-Q4_K_M.gguf` | 2.4 GB | path set in `config/inference.yaml`; the exact file used for every measurement here |
| Whisper base.en | `data/models/whisper-base.en/` (config.json, model.bin, tokenizer.json, vocabulary.txt) | 140 MB | only for voice |
| Wake word | `data/models/openwakeword/hey_veda.onnx` + `.onnx.data` | tiny | trained model; only for voice |
| bge-small embeddings | `data/bge-small-en-v1.5/` (`onnx/model.onnx`, `tokenizer.json`) | 130 MB | semantic memory; without it recall is off, not broken |
| Piper voice (optional) | set `audio.tts_model_path` | 60 MB | Linux/Pi TTS |
Secrets: `.env` (gitignored) holds `TAVILY_API_KEY` (web search) and optionally `VEDA_PROFILE=qwen3-4b|qwen3-1.7b|qwen3-0.6b`. Without the Tavily key `web_search` says it isn't set up; everything else works. A fresh `data/veda.db` is created on first run (the previous machine's conversations/tasks/saved facts are NOT transferred).

**First 15 minutes on the new machine**
1. `pytest -q` (no model needed; includes the isolated bootstrap smoke test).
2. `python scripts/spike_decision_protocol.py --variants A5` with NOTHING else running: expect ~43/51 (84%) on the tuned set; note the decision-ms median (laptop: ~12 s; this is the number that matters on the Pi).
3. `python scripts/spike_decision_protocol.py --variants A5 --golden tests/eval/orchestrator_heldout.yaml`: expect ~28/40 (70%). Never tune a prompt against this file.
4. Then run the app (`python -m controller.cli`, typed chat first) and try: weather in Tokyo -> "should I bring an umbrella?"; "I didn't ask about Singapore weather, what's 1+1?"; "what about tomorrow?" in a fresh session (should ask); add/list/"delete it" (should ask which)/delete by name; "open notepad"; "what's on my calendar" (should refuse).
5. On a Raspberry Pi: do step 2 first and write down the decision median, `prompt_cache_mb` use and prefix-cache hits. Small profiles (`VEDA_PROFILE=...`) use `n_ctx: 3072`; 0.6B/1.7B quality on this prompt is unmeasured.

**Reading order for the next agent:** this file -> `docs/10-orchestrator-review-and-plan.md` §6d-§6k (measurements, decisions, why) -> `docs/03-service-layer.md` + `docs/08-tool-harness.md` (current architecture) -> `docs/unified-assistant-orchestrator.md` (original design; Part 1-2 principles still apply) -> `docs/implementation_guide.md` (historical step list; phases are all done).
**Data notes:** `tests/eval/results/spike-protocol-20261008-002323.json` and `...-103421.json` were scored by a stale parser (they show 1/51 and 0/40 INVALID); the raw model output in them is valid — use `--rejudge <file> [--golden ...] --phase4` to score them correctly (43/51 and 28/40).
**Open decisions for the user:** larger control model vs accepting false calls vs fixing only the harmful classes (private data -> `tasks`, no-context guesses); whether to restore the old `veda.db`/`knowledge.json` from `vedic-atlas\data` (older state, 2-4 Oct) on the original laptop; merging to `main` (no flag: merging commits to this design).
