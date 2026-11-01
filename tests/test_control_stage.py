"""Phase 2 (docs/10): the control schema and the control prompt. No model needed."""

import json
import re
from datetime import datetime

import pytest

from domain.entities.conversation import Turn
from domain.policies.token_budget_policy import PromptBudgets
from domain.policies.tool_call_schema import CONTROL_KEYS, MAX_CALLS, build_control_schema
from service.prompting.prompt_composer import PromptComposer
from tpa.filestore.file_prompt_store import FilePromptStore
from tpa.filestore.yaml_tool_manifest_store import YamlToolManifestStore

MANIFESTS = YamlToolManifestStore().load_all()
NAMES = {m.name for m in MANIFESTS}
NOW = datetime(2026, 10, 7, 14, 30)


def _composer(now=NOW, manifests=MANIFESTS, budgets=None):
    return PromptComposer(FilePromptStore(), manifests, budgets=budgets, now=lambda: now)


def _t(role, content):
    return Turn(id=None, session_id="s", role=role, content=content, created_at=NOW)


# ---- schema ------------------------------------------------------------------

def test_control_schema_shape_and_default_key_order():
    s = build_control_schema(MANIFESTS)
    assert list(s["properties"]) == list(CONTROL_KEYS)
    # advise / also_chat / clarification are optional; language is required so the reply can be localised
    assert s["required"] == ["needs_live_data", "calls", "language"]
    assert CONTROL_KEYS[0] == "needs_live_data"            # the flag is committed to before the calls
    assert s["additionalProperties"] is False
    calls = s["properties"]["calls"]
    assert calls["minItems"] == 0 and calls["maxItems"] == MAX_CALLS
    assert {o["properties"]["tool"]["const"] for o in calls["items"]["oneOf"]} == NAMES
    assert s["properties"]["clarification"]["type"] == "string"


def test_control_schema_can_never_force_a_call():
    for manifests in (MANIFESTS, MANIFESTS[:1]):
        assert build_control_schema(manifests)["properties"]["calls"]["minItems"] == 0


def test_control_schema_key_order_is_configurable_but_must_be_a_permutation():
    order = ("calls", "needs_live_data", "advise", "also_chat", "clarification", "language")
    s = build_control_schema(MANIFESTS, key_order=order)
    assert list(s["properties"]) == list(order)
    with pytest.raises(ValueError):
        build_control_schema(MANIFESTS, key_order=("calls", "clarification"))


def test_control_schema_validates_real_decisions():
    jsonschema = pytest.importorskip("jsonschema")
    schema = build_control_schema(MANIFESTS)
    ok = [
        {"needs_live_data": False, "calls": [], "language": "English"},
        {"needs_live_data": True, "calls": [{"tool": "get_weather", "args": {"place": "Tokyo"}}], "language": "English"},
        {"needs_live_data": False, "calls": [], "clarification": "Which task do you mean?", "language": "English"},
    ]
    for doc in ok:
        jsonschema.validate(doc, schema)
    bad = [
        {"needs_live_data": False, "calls": [{"tool": "rm_rf", "args": {}}], "language": "English"},        # unknown tool
        {"needs_live_data": False, "calls": [{"tool": "tasks", "args": {"action": "zap"}}], "language": "English"},
        {"needs_live_data": False, "calls": [], "clarification": "x" * 201, "language": "English"},
        {"calls": [], "clarification": None, "language": "English"},                                         # flag missing
    ]
    for doc in bad:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(doc, schema)


# ---- prompt: prefix stability -----------------------------------------------

def test_system_block_is_byte_identical_across_turns_and_has_no_volatile_text():
    c = _composer()
    a = c.control_stage("what's the weather in Tokyo")
    b = c.control_stage("explain recursion", [_t("user", "hi"), _t("assistant", "hello")], "ACTIVE: tasks | just now")
    later = _composer(now=datetime(2027, 1, 1, 9, 0)).control_stage("anything")
    assert a.system == b.system == later.system
    # The tail's line markers must not appear with real content. `RECENT: none` / `ACTIVE: none` are legitimate in the
    # static examples (they teach the empty case); `RECENT:` followed by turns, a TODAY line or a bare USER line are not.
    lines = a.system.splitlines()
    assert not any(l.startswith("TODAY:") or l.startswith("USER:") for l in lines)
    assert all(l == "RECENT: none" for l in lines if l.startswith("RECENT:"))
    assert all(l == "ACTIVE: none" or l.startswith("ACTIVE: ") for l in lines if l.startswith("ACTIVE:"))
    assert "2026" not in a.system and "Tokyo" not in a.system.split("Examples:")[0]


def test_one_cached_prefix_regardless_of_call_pattern():
    c = _composer()
    for msg in ("a", "b", "c"):
        c.control_stage(msg)
    c.control_stage("d", [_t("user", "x"), _t("assistant", "y")], "ACTIVE: get_weather | place=Pune | just now")
    assert len(c._control_static) == 1


def test_every_tool_signature_is_in_the_static_block():
    system = _composer().control_stage("x").system
    for m in MANIFESTS:
        assert f"- {m.name}(" in system


def test_prompt_examples_only_reference_real_tools_and_args():
    body = FilePromptStore().get("control_stage")
    params = {m.name: {p.name for p in m.params} for m in MANIFESTS}
    seen = 0
    for line in body.splitlines():
        if not line.startswith('{"needs_live_data"'):
            continue
        doc = json.loads(line)
        assert list(doc) == [k for k in CONTROL_KEYS if k in doc] and {"needs_live_data", "calls"} <= set(doc)
        assert doc.get("clarification") is not None or "clarification" not in doc   # never an explicit null (costs tokens)
        if "clarification" in doc:
            assert doc["calls"] == []
        for call in doc["calls"]:
            seen += 1
            assert call["tool"] in NAMES, f"control_stage.md example uses unknown tool {call['tool']}"
            assert set(call["args"]) <= params[call["tool"]], f"unknown arg in example {call}"
    assert seen >= 4


def test_examples_follow_the_schema_key_order():
    for order in (CONTROL_KEYS, ("calls", "needs_live_data", "advise", "also_chat", "clarification", "language")):
        system = _composer().control_stage("x", key_order=order).system
        lines = [l for l in system.splitlines() if l.startswith("{") and "needs_live_data" in l and "calls" in l]
        assert len(lines) >= 8
        for line in lines:
            data = json.loads(line)
            assert list(data) == [k for k in order if k in data]


def test_needs_live_data_definition_is_in_the_prompt_and_saved_facts_are_not_live():
    body = FilePromptStore().get("control_stage")
    assert "needs_live_data = true ONLY if" in body and "needs_live_data = false for everything else" in body
    true_line = next(l for l in body.splitlines() if l.startswith("needs_live_data = true"))
    assert "facts" not in true_line          # saved facts are answered from the chat prompt (review gap G2)
    assert "NOT a request" in body           # the mention-without-request rule


def test_prompt_examples_are_not_copies_of_golden_test_cases():
    """An example that is also a test case measures memorisation, not judgement (it leaked 'Delhi' into a no-state case)."""
    import yaml
    from pathlib import Path
    golden = {
        g["say"].lower()
        for name in ("orchestrator_golden.yaml", "orchestrator_heldout.yaml")
        for g in yaml.safe_load((Path(__file__).parent / "eval" / name).read_text(encoding="utf-8"))
    }
    system = _composer().control_stage("x").system
    shown = {line[len("User: "):].lower() for line in system.splitlines() if line.startswith("User: ")}
    assert shown and not (shown & golden), f"prompt example(s) duplicate golden cases: {shown & golden}"


def test_the_mention_without_request_pattern_and_the_no_fitting_tool_rule_are_present():
    body = FilePromptStore().get("control_stage")
    assert "NOT a request" in body and "NO tool above can provide it" in body
    assert any(l == '{"needs_live_data":false,"calls":[]}' for l in body.splitlines())
    assert '"clarification":null' not in body       # an explicit null is ~5 wasted output tokens on every turn


# ---- prompt: volatile tail ---------------------------------------------------

def test_volatile_tail_order_active_recent_today_user():
    p = _composer().control_stage(
        "should I bring an umbrella?", [_t("user", "weather in Tokyo"), _t("assistant", "18 degrees")],
        "ACTIVE: get_weather | place=Tokyo | 3 min ago",
    )
    lines = p.prompt.splitlines()
    assert lines[0].startswith("ACTIVE:") and lines[1] == "RECENT:"
    assert lines[2] == "User: weather in Tokyo" and lines[3] == "Veda: 18 degrees"
    assert lines[4].startswith("TODAY: Wednesday 07 Oct 2026") and lines[5] == "USER: should I bring an umbrella?"
    assert p.trimmed == []                                   # a normal turn fits the budget with its history intact


def test_active_is_omitted_entirely_when_empty():
    assert "ACTIVE" not in _composer().control_stage("hi").prompt


def test_history_is_whole_exchanges_only():
    c = _composer()
    turns = [_t("assistant", "orphan"), _t("user", "q1"), _t("assistant", "a1"), _t("user", "q2"), _t("assistant", "a2"), _t("user", "unanswered")]
    got = c._exchange_turns(turns, 2)
    assert [t.content for t in got] == ["q1", "a1", "q2", "a2"]
    assert [t.content for t in c._exchange_turns(turns, 1)] == ["q2", "a2"]
    assert c._exchange_turns(turns, 0) == []
    assert c._exchange_turns([_t("user", "only")], 2) == []


def test_over_budget_drops_oldest_exchange_first_and_keeps_active_and_user():
    base = _composer().control_stage("x").tokens
    c = _composer(budgets=PromptBudgets(control=base + 40))
    turns = [_t("user", "old question " * 8), _t("assistant", "old answer " * 8), _t("user", "recent q"), _t("assistant", "recent a")]
    p = c.control_stage("final question", turns, "ACTIVE: get_weather | place=Tokyo | just now")
    assert "history_exchange" in p.trimmed
    assert "recent q" in p.prompt and "recent a" in p.prompt      # the newest pair is the last to go
    assert "old question" not in p.prompt
    assert "ACTIVE: get_weather" in p.prompt and "USER: final question" in p.prompt


def test_only_flagged_examples_are_rendered_one_per_tool_at_most():
    system = _composer().control_stage("x").system
    for m in MANIFESTS:
        assert sum(ex.prompt_example for ex in m.examples) <= 1
        for ex in m.examples:
            assert (f"User: {ex.user}\n" in system) == ex.prompt_example


def test_static_prompt_leaves_room_for_the_volatile_tail_inside_the_budget():
    p = _composer().control_stage("x")
    assert p.sections["static"] <= PromptBudgets().control - 400, "static prompt grew: no room for ACTIVE + RECENT + USER"


def test_token_counts_are_reported():
    p = _composer().control_stage("hi")
    assert p.tokens == p.sections["static"] + _composer()._count(p.prompt) and p.sections["static"] > 100
    assert re.search(r"get_weather", p.system)


# ---- prompt budgets as configuration (guide §2.1b) ---------------------------

def test_shipped_prompting_yaml_loads_and_builds_valid_budgets():
    from core.config import load_full_config
    from domain.policies.token_budget_policy import PromptBudgets

    cfg = load_full_config().prompting
    b = PromptBudgets(**cfg.budgets)
    assert b.control > b.chat and b.observation_max > 0 and cfg.control_history_exchanges == 2
    assert (cfg.turn_chars, cfg.user_message_chars) == (200, 500)


def test_prompting_config_rejects_unknown_or_negative_budgets_and_missing_keys_fall_back():
    from domain.policies.token_budget_policy import PromptBudgets
    from schemas.config_schemas import PromptingConfig

    with pytest.raises(ValueError):
        PromptingConfig(budgets={"controll": 1})
    with pytest.raises(ValueError):
        PromptingConfig(budgets={"control": -5})
    partial = PromptBudgets(**PromptingConfig(budgets={"control": 2500}).budgets)
    assert partial.control == 2500 and partial.narrate == PromptBudgets().narrate   # unlisted keys keep their default


def test_composer_clip_lengths_are_configurable():
    long_turn = "word " * 100
    default = _composer().control_stage("x", [_t("user", long_turn), _t("assistant", long_turn)])
    short = PromptComposer(FilePromptStore(), MANIFESTS, now=lambda: NOW, turn_chars=50, user_message_chars=20).control_stage(
        "y" * 100, [_t("user", long_turn), _t("assistant", long_turn)])
    assert len(default.prompt) > len(short.prompt)
    assert "USER: " + "y" * 19 + "…" in short.prompt


def test_a_decision_without_the_optional_clarification_key_parses():
    from service.agent.control_decoder import parse_decision
    d = parse_decision('{"needs_live_data":false,"calls":[]}')
    assert d.valid and d.clarification is None and d.calls == ()
    asked = parse_decision('{"needs_live_data":false,"calls":[],"clarification":"Which one?"}')
    assert asked.valid and asked.clarification == "Which one?"
    assert not parse_decision('{"calls":[]}').valid          # the flag is still mandatory


# ---- guards (docs/12 P1) -----------------------------------------------------------------------------------------------

# The composer's default counter (estimate_tokens, 3.4 chars/token) deliberately over-counts: the static block measures 2588
# with it today, against ~2,240 measured with the real tokenizer (training/README.md). docs/12's 2,400-token cap is therefore
# ~2,750 in this counter's units. Raised to 3,000 when budgets.control went from 3000 to 3450 (+15%, the 4B model is the
# only supported one) to make room for P3/P5. Now equal to budgets.control (3550) - 400 (the room kept for ACTIVE/RECENT/USER).
STATIC_CONTROL_TOKEN_CAP = 3150


def test_static_control_prompt_stays_under_the_token_cap():
    composer = _composer()
    tokens = composer._count(composer.control_stage("x").system)
    assert tokens <= STATIC_CONTROL_TOKEN_CAP, (
        f"the static control prompt is {tokens} tokens (cap {STATIC_CONTROL_TOKEN_CAP}). control_stage never trims the "
        "static block: past the budget it silently drops CONVERSATION HISTORY, so follow-ups like 'and in Chennai?' "
        "stop working and it looks like a model problem. Shorten tool descriptions or examples before adding more."
    )


def test_every_rendered_example_round_trips_the_key_order_helper_unchanged():
    # _in_key_order silently skips examples whose keys are outside CONTROL_KEYS: such an example would be shown to the
    # model in one order while the grammar forces another.
    system = _composer().control_stage("x").system
    lines = [l for l in system.splitlines() if l.startswith('{"needs_live_data"')]
    assert len(lines) >= 8
    for line in lines:
        assert set(json.loads(line)) <= set(CONTROL_KEYS), f"example uses a key the schema does not know: {line}"
        assert PromptComposer._in_key_order(line, CONTROL_KEYS) == line


# ---- advise / also_chat (docs/12 P3) -------------------------------------------------------------------------------------

def test_advise_and_also_chat_are_optional_booleans_emitted_after_calls_and_before_clarification():
    s = build_control_schema(MANIFESTS)
    assert list(s["properties"])[-3:] == ["advise", "also_chat", "clarification"] and list(s["properties"])[:2] == ["needs_live_data", "calls"]
    assert s["properties"]["advise"] == {"type": "boolean"} and s["properties"]["also_chat"] == {"type": "boolean"}
    assert "advise" not in s["required"] and "also_chat" not in s["required"]   # an always-present key costs ~2 s a turn


def test_flags_default_to_false_and_are_read_strictly():
    from service.agent.control_decoder import parse_decision
    d = parse_decision('{"needs_live_data":true,"calls":[]}')
    assert d.valid and d.advise is False and d.also_chat is False
    d = parse_decision('{"needs_live_data":true,"calls":[],"advise":true,"also_chat":true}')
    assert d.valid and d.advise is True and d.also_chat is True
    for bad in ('"advise":"true"', '"advise":1', '"also_chat":"yes"'):
        assert not parse_decision('{"needs_live_data":true,"calls":[],' + bad + "}").valid


def test_only_the_readout_tools_allow_advice():
    assert {m.name for m in MANIFESTS if m.allow_advice} == {"get_weather", "get_weather_forecast", "calendar_agenda", "device_status", "get_room_temperature"}


def test_the_flags_ride_on_the_resolution_but_the_route_table_is_unchanged():
    from domain.entities.control_decision import ControlDecision
    from domain.policies.dispatch_policy import Route, resolve
    by_name = {m.name: m for m in MANIFESTS}
    call = {"tool": "get_weather", "args": {"place": "Pune"}}
    r = resolve(ControlDecision(calls=(call,), needs_live_data=True, advise=True, also_chat=True), by_name)
    assert r.route is Route.TOOLS and r.advise and r.also_chat
    r = resolve(ControlDecision(calls=(call,), needs_live_data=True), by_name)
    assert r.route is Route.TOOLS and not r.advise and not r.also_chat
    assert resolve(ControlDecision(needs_live_data=False, advise=True, also_chat=True), by_name).route is Route.CHAT
