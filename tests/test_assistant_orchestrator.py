"""Phase 3 (docs/10): the orchestrator's turn flow, with a scripted decoder and fake skills (no model)."""

import asyncio
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from domain.entities.agent_context import AgentContext
from domain.entities.agent_result import AgentResult
from domain.entities.control_decision import ControlDecision
from domain.entities.skill_result import SkillResult
from domain.policies.dispatch_policy import ONE_AT_A_TIME_REPLY, REFUSAL_REPLY, SEPARATELY_REPLY, UNPARSEABLE_REPLY
from domain.policies.reply_policy import UNCONFIRMED_REPLY
from service.agent.assistant_orchestrator import AssistantOrchestrator
from service.agent.base_agent import BaseAgent
from service.agent.control_decoder import ControlDecoder, DecodeResult, parse_decision
from service.agent.supervisor import SupervisorAgent
from service.prompting.prompt_composer import PromptComposer
from service.session.session_state import SessionStateService
from tpa.filestore.file_prompt_store import FilePromptStore
from tpa.filestore.yaml_tool_manifest_store import YamlToolManifestStore
from tpa.persistence.models import session_context, turn_trace  # noqa: F401 (register tables)
from tpa.persistence.repositories.session_context_repository import SqliteSessionContextRepository
from tpa.persistence.repositories.trace_repository import SqliteTraceRepository
from tpa.persistence.session import Base

MANIFESTS = {m.name: m for m in YamlToolManifestStore().load_all()}
NOW = datetime(2026, 10, 7, 14, 30)


def call(tool, **args):
    return {"tool": tool, "args": args}


def dec(calls=(), live=None, clar=None):
    return ControlDecision(calls=tuple(calls), needs_live_data=live, clarification=clar)


class ScriptedDecoder:
    """Returns the next scripted decision and records what it was shown."""

    def __init__(self, *decisions):
        self.decisions, self.seen = list(decisions), []

    async def decide(self, user_message, history=(), active=""):
        self.seen.append({"message": user_message, "active": active, "history": list(history)})
        return DecodeResult(decision=self.decisions.pop(0), prompt_tokens=900, ms=7)


class FakeSkills:
    def __init__(self, outputs):
        self.outputs, self.ran = outputs, []

    async def execute_skill(self, ctx, name, **args):
        self.ran.append((name, args))
        out = self.outputs[name]
        if isinstance(out, Exception):
            raise out
        return out(args) if callable(out) else out


class Convo:
    def __init__(self):
        self.turns, self.added, self.resolved = [], [], 0

    def current_session_id(self):
        self.resolved += 1
        return "sess-1"

    def add_turn(self, role, content, session_id=None):
        self.added.append((role, content, session_id))


class FakeResponder(BaseAgent):
    def __init__(self, text="Chat reply."):
        super().__init__(name="responder", description="chat")
        self.text, self.ran = text, 0

    async def execute(self, ctx):
        self.ran += 1
        return AgentResult(agent_name="responder", response=self.text)

    async def execute_stream(self, ctx, cancel_event=None):
        self.ran += 1
        yield self.text


class FakeClient:
    def __init__(self, text="Narrated."):
        self.text, self.calls = text, 0

    async def complete(self, **kw):
        self.calls += 1
        return self.text


def ok(spoken=None, observation="obs", final=False):
    return SkillResult(skill_name="x", success=True, output=observation, metadata={"spoken": spoken, "final": final})


def fail(error="provider down"):
    return SkillResult(skill_name="x", success=False, error=error)


def env(decisions, skills, *, responder_text="Chat reply.", narration="Narrated.", claims=lambda t: False, conversation_intent=None, adaptive_reply=False):
    engine = create_engine("sqlite:///:memory:", future=True, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, future=True, expire_on_commit=False)
    session_state = SessionStateService(SqliteSessionContextRepository(session_factory=factory), MANIFESTS, now=lambda: NOW)
    traces = SqliteTraceRepository(session_factory=factory)
    decoder, runner, convo = ScriptedDecoder(*decisions), FakeSkills(skills), Convo()
    responder, client = FakeResponder(responder_text), FakeClient(narration)
    orch = AssistantOrchestrator(
        decoder=decoder, composer=PromptComposer(FilePromptStore(), list(MANIFESTS.values())), skill_runner=runner,
        manifests=MANIFESTS, responder=responder, client=client, conversation=convo, claims_action=claims,
        session_state=session_state, traces=traces, grounding_check=lambda *a: True, conversation_intent=conversation_intent,
        adaptive_reply=adaptive_reply,
    )
    return SimpleNamespace(orch=orch, decoder=decoder, runner=runner, convo=convo, responder=responder,
                           client=client, state=session_state, traces=traces)


def turn(e, message, stream=False):
    ctx = AgentContext(user_message=message)
    if stream:
        async def run():
            return "".join([c async for c in e.orch.execute_stream(ctx)])
        return ctx, asyncio.run(run())
    return ctx, asyncio.run(e.orch.execute(ctx))


WEATHER = lambda a: ok(spoken=f"In {a.get('place') or 'Hyderabad'} it's 18 degrees.")


# ---- the two original bugs ---------------------------------------------------

def test_follow_up_reuses_the_active_place_from_the_previous_turn():
    e = env([dec([call("get_weather", place="Tokyo")], live=True), dec([call("get_weather", place="Tokyo")], live=True)],
            {"get_weather": WEATHER})
    turn(e, "what's the weather in Tokyo")
    assert e.decoder.seen[0]["active"] == ""                                   # nothing active on the first turn
    _, result = turn(e, "should I bring an umbrella?")
    assert e.decoder.seen[1]["active"] == "ACTIVE: get_weather | place=Tokyo | just now"
    assert result.response == "In Tokyo it's 18 degrees." and e.responder.ran == 0


def test_mention_without_request_goes_to_chat_and_no_tool_runs():
    e = env([dec(live=False)], {"get_weather": WEATHER}, responder_text="2")
    _, result = turn(e, "I didn't ask about Singapore weather, what's 1+1?")
    assert result.response == "2" and e.runner.ran == [] and e.responder.ran == 1
    assert e.client.calls == 0                                                 # no content decode either


# ---- dispatch routes ---------------------------------------------------------

def test_live_data_with_no_tool_is_refused_and_chat_is_not_consulted():
    e = env([dec(live=True)], {})
    _, result = turn(e, "what's on my calendar")
    assert result.response == REFUSAL_REPLY and e.responder.ran == 0 and e.runner.ran == []


def test_clarification_is_asked_saved_and_runs_nothing():
    e = env([dec(clar="Which task do you mean?")], {})
    ctx, result = turn(e, "delete it")
    assert result.response == "Which task do you mean?" and e.runner.ran == []
    assert ("user", "delete it", "sess-1") in e.convo.added and ("assistant", "Which task do you mean?", "sess-1") in e.convo.added


def test_invalid_decode_fails_closed_instead_of_guessing():
    e = env([ControlDecision(valid=False, note="truncated")], {})
    _, result = turn(e, "anything")
    assert result.response == UNPARSEABLE_REPLY and e.responder.ran == 0 and e.runner.ran == []


def test_destructive_call_in_a_multi_call_is_not_executed():
    e = env([dec([call("get_weather", place="Pune"), call("tasks", action="delete", title="bank")])], {"get_weather": WEATHER})
    _, result = turn(e, "weather in Pune and delete the bank task")
    assert result.response == "In Pune it's 18 degrees. " + SEPARATELY_REPLY    # the weather ran; the delete never did
    assert e.runner.ran == [("get_weather", {"place": "Pune"})]


def test_the_models_own_clarification_is_asked_after_the_tool_answer_and_saved_in_history():
    decision = ControlDecision(calls=(call("get_weather", place="Pune"),), needs_live_data=True, clarification="Which task do you mean?")
    e = env([decision], {"get_weather": WEATHER})
    _, result = turn(e, "weather in Pune and mark it done")
    assert result.response == "In Pune it's 18 degrees. Which task do you mean?"
    assert e.convo.added[-1][1] == result.response
    assert "follow-up question asked" in e.traces.recent(1)[0].notes


def test_orchestration_bug_returns_an_error_reply_not_an_exception():
    class Boom:
        async def decide(self, *a, **k): raise RuntimeError("bug")
    e = env([], {})
    e.orch._decoder = Boom()
    _, result = turn(e, "hi")
    assert "went wrong" in result.response


# ---- tools: reply assembly ---------------------------------------------------

def test_multi_intent_runs_both_in_order_and_joins_template_sentences_verbatim_with_no_model_call():
    e = env([dec([call("get_weather", place="Tokyo"), call("convert_currency", amount=100, **{"from": "USD", "to": "INR"})], live=True)],
            {"get_weather": WEATHER, "convert_currency": ok(spoken="100 USD is about 8300 INR.")})
    _, result = turn(e, "weather in Tokyo and convert 100 USD to INR")
    assert result.response == "In Tokyo it's 18 degrees. 100 USD is about 8300 INR."
    assert [n for n, _ in e.runner.ran] == ["get_weather", "convert_currency"] and e.client.calls == 0


def test_only_results_that_need_narrating_reach_the_model_and_share_one_decode():
    e = env([dec([call("web_search", query="IPL"), call("get_weather", place="Tokyo"), call("web_search", query="ISRO")], live=True)],
            {"web_search": ok(observation="SNIPPET about the topic"), "get_weather": WEATHER}, narration="Here is the news.")
    _, result = turn(e, "search IPL, weather Tokyo, search ISRO")
    assert e.client.calls == 1                                                  # one content decode for both searches
    assert result.response == "Here is the news. In Tokyo it's 18 degrees."     # spliced at the first narrated position


def test_failed_call_gets_plain_failure_text_never_an_invented_result():
    e = env([dec([call("get_weather", place="Tokyo"), call("convert_currency", amount=5, **{"from": "USD", "to": "EUR"})], live=True)],
            {"get_weather": fail("provider down"), "convert_currency": ok(spoken="5 USD is 4.3 EUR.")})
    _, result = turn(e, "x")
    assert result.response == "I couldn't do that: provider down 5 USD is 4.3 EUR."


def test_a_failed_call_that_carries_its_own_sentence_is_spoken_with_that_sentence():
    failed = SkillResult(skill_name="x", success=False, error="no app name", metadata={"spoken": "Which app do you mean?"})
    e = env([dec([call("get_weather", place="Tokyo")], live=True)], {"get_weather": failed})
    _, result = turn(e, "x")
    assert result.response == "Which app do you mean?"


def test_a_skill_that_raises_is_a_failed_call_not_a_crash():
    e = env([dec([call("get_weather", place="Tokyo")], live=True)], {"get_weather": RuntimeError("boom")})
    _, result = turn(e, "x")
    assert result.response == "I couldn't do that: boom"


def test_ungrounded_or_failed_narration_falls_back_to_the_tool_text():
    e = env([dec([call("web_search", query="x")], live=True)], {"web_search": ok(observation="first line\nsecond")}, narration="")
    _, result = turn(e, "x")
    assert result.response == "first line"


def test_narration_input_is_capped_by_the_tools_max_result_tokens():
    seen = {}
    e = env([dec([call("web_search", query="x")], live=True)], {"web_search": ok(observation="word " * 3000)})
    orig = e.orch._composer.narrate_stage
    e.orch._composer.narrate_stage = lambda q, results, language=None: seen.setdefault("n", len(results[0])) and orig(q, results, language=language)
    turn(e, "x")
    assert seen["n"] < 3000 * 5 / 2                                             # cut well below the raw 15000 chars


# ---- backstop ----------------------------------------------------------------

def test_claim_with_no_successful_tool_is_replaced_on_the_clarify_path_too():
    e = env([dec(clar="Task added to your list.")], {}, claims=lambda t: "task added" in t.lower())
    _, result = turn(e, "x")
    assert result.response == UNCONFIRMED_REPLY


def test_claim_backed_by_a_successful_tool_is_allowed():
    e = env([dec([call("tasks", action="add", title="milk")])], {"tasks": ok(spoken="Task added: milk.")},
            claims=lambda t: "task added" in t.lower())
    _, result = turn(e, "add milk")
    assert result.response == "Task added: milk."


# ---- session state, persistence, trace ---------------------------------------

def test_session_id_is_resolved_once_and_state_is_written_only_after_success():
    e = env([dec([call("get_weather", place="Tokyo")], live=True), dec([call("get_weather", place="Oslo")], live=True)],
            {"get_weather": WEATHER})
    ctx, _ = turn(e, "weather in Tokyo")
    assert e.convo.resolved == 1 and ctx.session_id == "sess-1"
    assert e.state.get("sess-1").slots == {"place": "Tokyo"}
    e.runner.outputs["get_weather"] = fail()
    turn(e, "weather in Oslo")
    assert e.state.get("sess-1").slots == {"place": "Tokyo"}                     # a failed call never overwrites


def test_destructive_call_runs_but_never_becomes_state():
    e = env([dec([call("tasks", action="delete", title="bank")])], {"tasks": ok(spoken="Removed.")})
    turn(e, "delete the bank task")
    assert e.state.get("sess-1") is None


def test_trace_records_the_new_fields():
    e = env([dec([call("get_weather", place="Tokyo")], live=True), dec([call("get_weather", place="Tokyo")], live=True)],
            {"get_weather": WEATHER})
    turn(e, "weather in Tokyo")
    turn(e, "umbrella?")
    t = e.traces.recent(1)[0]
    assert (t.agent, t.decided, t.state_used, t.needs_live_data, t.clarified) == ("assistant", "tool", True, True, False)
    assert t.slots_inherited == ["place"] and t.prompt_tokens["control"] == 900 and t.timings_ms["control"] == 7
    turn_clar = env([dec(clar="Which one?")], {})
    turn(turn_clar, "delete it")
    assert turn_clar.traces.recent(1)[0].clarified is True and turn_clar.traces.recent(1)[0].decided == "clarify"


def test_exactly_one_control_decode_per_turn():
    e = env([dec(live=False), dec([call("get_weather", place="Tokyo")], live=True)], {"get_weather": WEATHER})
    turn(e, "chat")
    turn(e, "weather")
    assert len(e.decoder.seen) == 2


# ---- streaming + supervisor delegation ---------------------------------------

def test_stream_of_a_tool_turn_emits_the_reply_once_and_chat_streams_through_the_responder():
    e = env([dec([call("get_weather", place="Tokyo")], live=True), dec(live=False)], {"get_weather": WEATHER}, responder_text="Hi!")
    _, tool_text = turn(e, "weather", stream=True)
    _, chat_text = turn(e, "hello", stream=True)
    assert tool_text == "In Tokyo it's 18 degrees." and chat_text == "Hi!"


def test_supervisor_delegates_every_turn_to_the_orchestrator():
    e = env([dec(live=False)], {}, responder_text="Chat!")
    sup = SupervisorAgent(orchestrator=e.orch)
    result = asyncio.run(sup.execute(AgentContext(user_message="hello")))
    assert result.response == "Chat!" and result.delegated_to == "responder"

    e2 = env([dec([call("get_weather", place="Tokyo")], live=True)], {"get_weather": WEATHER})
    sup2 = SupervisorAgent(orchestrator=e2.orch)

    async def run():
        return "".join([c async for c in sup2.execute_stream(AgentContext(user_message="weather"))])

    assert asyncio.run(run()) == "In Tokyo it's 18 degrees."


# ---- decoder -----------------------------------------------------------------

def test_parse_decision_is_strict():
    good = '{"needs_live_data": true, "calls": [{"tool": "get_weather", "args": {}}], "clarification": null}'
    d = parse_decision(good)
    assert d.valid and d.needs_live_data is True and d.calls[0]["tool"] == "get_weather"
    for bad in ("", "not json", '{"calls": []}', '{"needs_live_data": "yes", "calls": [], "clarification": null}',
                '{"needs_live_data": true, "calls": {}, "clarification": null}', '{"needs_live_data": true, "calls": [], "clarification": 3}'):
        assert not parse_decision(bad).valid


def test_decoder_sends_the_constrained_schema_and_never_raises():
    class Client:
        def __init__(self, out): self.out, self.kw = out, None
        async def complete(self, **kw):
            self.kw = kw
            if isinstance(self.out, Exception):
                raise self.out
            return self.out

    composer = PromptComposer(FilePromptStore(), list(MANIFESTS.values()))
    good = Client('{"needs_live_data": false, "calls": [], "clarification": null}')
    r = asyncio.run(ControlDecoder(client=good, composer=composer, manifests=MANIFESTS).decide("hi"))
    assert r.decision.valid and list(good.kw["json_schema"]["properties"])[0] == "needs_live_data" and r.prompt_tokens > 0
    assert good.kw["system"] == composer.control_stage("other").system         # the cached prefix

    boom = Client(RuntimeError("model down"))
    r = asyncio.run(ControlDecoder(client=boom, composer=composer, manifests=MANIFESTS).decide("hi"))
    assert not r.decision.valid and "model down" in r.decision.note


def test_decoder_defaults_are_the_measured_winners():
    """Spike A5: temperature 0 and explicit `ACTIVE: none` / `RECENT: none` scored best (docs/10 §6c)."""
    class Client:
        kw = None
        async def complete(self, **kw):
            self.kw = kw
            return '{"needs_live_data": false, "calls": [], "clarification": null}'

    c = Client()
    composer = PromptComposer(FilePromptStore(), list(MANIFESTS.values()))
    asyncio.run(ControlDecoder(client=c, composer=composer, manifests=MANIFESTS).decide("hello there"))
    assert c.kw["temperature"] == 0.0
    assert "ACTIVE: none" in c.kw["prompt"] and "RECENT: none" in c.kw["prompt"]


# ---- data plane: a document never reaches a control prompt (docs/10 Phase 5) -----------------------------

def test_the_control_prompt_has_no_input_for_a_tool_result():
    import inspect
    params = set(inspect.signature(PromptComposer.control_stage).parameters) - {"self"}
    assert params == {"user_message", "history", "active", "exchanges", "key_order", "show_empty"}


def test_a_document_cannot_leak_through_history_or_state_when_narration_fails():
    doc = "SECRETDOC " * 3000                                  # one huge line, as an email body would be
    e = env([dec([call("web_search", query="x")], live=True)], {"web_search": ok(observation=doc)}, narration="")
    _, result = turn(e, "search x")
    saved = e.convo.added[-1][1]
    assert saved == result.response
    assert len(saved) < 400 and saved != doc                    # capped fallback, not the document
    assert "SECRETDOC" not in e.state.active_line("sess-1")     # session state only ever holds validated ARGUMENTS


def test_a_document_result_is_capped_before_the_content_decode():
    seen = {}
    e = env([dec([call("web_search", query="x")], live=True)], {"web_search": ok(observation="word " * 3000)})
    orig = e.orch._composer.content_stage
    e.orch._composer.content_stage = lambda task, doc, q, **kw: seen.setdefault("task", task) and orig(task, doc, q, **kw)
    turn(e, "x")
    assert seen["task"] == "narrate"


# ---- questions about the conversation itself always reach chat (where the history and memory are) ---------------------------------------

def about(answer=True, score=None):
    """A fake conversation-intent scorer: strong (0.9) or none (0.1), or an exact score."""
    async def check(message):
        return score if score is not None else (0.9 if answer else 0.1)
    return check


def test_a_question_about_the_conversation_goes_to_chat_even_if_the_router_asked_a_question():
    e = env([dec(clar="What would you like to know about the conversation so far?")], {}, responder_text="We talked about plans.",
            conversation_intent=about())
    _, result = turn(e, "what are we discussing")
    assert result.response == "We talked about plans." and e.responder.ran == 1


def test_a_question_about_the_conversation_never_runs_a_tool_the_router_picked():
    e = env([dec([call("web_search", query="what areas we talked")], live=True)], {"web_search": ok(spoken="web!")},
            responder_text="We talked about plans.", conversation_intent=about())
    _, result = turn(e, "what areas have we talked about")
    assert result.response == "We talked about plans." and e.responder.ran == 1


def test_other_questions_keep_the_routers_decision():
    e = env([dec([call("get_weather", place="Tokyo")], live=True)], {"get_weather": WEATHER}, conversation_intent=about(False))
    _, result = turn(e, "weather in Tokyo")
    assert result.response == "In Tokyo it's 18 degrees." and e.responder.ran == 0


def test_a_failing_intent_check_does_not_break_the_turn():
    async def boom(message):
        raise RuntimeError("embedding down")

    e = env([dec([call("get_weather", place="Tokyo")], live=True)], {"get_weather": WEATHER}, conversation_intent=boom)
    _, result = turn(e, "weather in Tokyo")
    assert result.response == "In Tokyo it's 18 degrees."


def test_the_streamed_path_also_reaches_chat():
    e = env([dec(clar="What do you mean?")], {}, responder_text="We talked about plans.", conversation_intent=about())
    _, text = turn(e, "what are we discussing", stream=True)
    assert text == "We talked about plans." and e.responder.ran == 1


def test_a_softer_match_replaces_a_clarifying_question_with_chat_and_tells_chat_why():
    e = env([dec(clar="What would you like to know about our conversation?")], {}, responder_text="We talked about plans.",
            conversation_intent=about(score=0.795))                     # "can you tell me what are we discusing lately"
    ctx, result = turn(e, "can you tell me what are we discusing lately")
    assert result.response == "We talked about plans." and e.responder.ran == 1 and ctx.metadata["about_conversation"] is True


def test_a_softer_match_replaces_a_refusal_too():
    e = env([dec(live=True)], {}, responder_text="We talked about plans.", conversation_intent=about(score=0.78))
    _, result = turn(e, "what have we been on about lately")
    assert result.response == "We talked about plans."


def test_a_softer_match_never_overrides_a_tool_the_router_chose():
    e = env([dec([call("get_weather", place="Tokyo")], live=True)], {"get_weather": WEATHER}, conversation_intent=about(score=0.78))
    ctx, result = turn(e, "can you tell me the weather")
    assert result.response == "In Tokyo it's 18 degrees." and e.responder.ran == 0 and "about_conversation" not in ctx.metadata


def test_a_score_below_the_soft_threshold_changes_nothing():
    e = env([dec(clar="Which one?")], {}, conversation_intent=about(score=0.70))
    _, result = turn(e, "which one")
    assert result.response == "Which one?" and e.responder.ran == 0


def test_the_conversation_intent_override_only_ever_produces_chat():
    cases = [
        dec([call("get_weather", place="Tokyo")], live=True),   # tools -> chat
        dec(clar="What do you mean?"),                          # clarify -> chat
        dec(live=True),                                         # refuse -> chat
        dec(),                                                  # chat stays chat
        ControlDecision(valid=False, note="bad"),               # fail-closed -> chat
    ]
    for decision in cases:
        e = env([decision], {"get_weather": WEATHER}, responder_text="Chat.", conversation_intent=about())
        _, result = turn(e, "what are we discussing")
        assert result.response == "Chat." and e.responder.ran == 1 and e.runner.ran == []


def test_the_trace_records_reply_mode_and_also_chat_only_when_adaptive_reply_is_on():
    flagged = ControlDecision(calls=(call("get_weather", place="Tokyo"),), needs_live_data=True, also_chat=True)
    for adaptive in (False, True):
        e = env([flagged], {"get_weather": WEATHER}, adaptive_reply=adaptive)
        turn(e, "weather in Tokyo")
        trace = e.traces.recent(1)[0]
        assert trace.reply_mode == "template" and trace.also_chat is adaptive


# ---- advise (docs/12 P4) -------------------------------------------------------------------------------------------------

from domain.policies.reply_policy import resolve_reply_mode   # noqa: E402

FORECAST = ok(spoken="Tomorrow in Pune: rain, 24 degrees.")
ADVISE_CALL = lambda: ControlDecision(calls=(call("get_weather_forecast", place="Pune", date_offset=1),), needs_live_data=True, advise=True)


class RecordingClient(FakeClient):
    def __init__(self, text):
        super().__init__(text)
        self.prompts = []

    async def complete(self, **kw):
        self.prompts.append(kw)
        return await super().complete(**kw)


def advise_env(decision, skills, text, *, adaptive=True, real_grounding=True):
    e = env([decision], skills, narration=text, adaptive_reply=adaptive)
    e.orch._client = e.client = RecordingClient(text)
    if real_grounding:
        e.orch._grounding_check = AssistantOrchestrator._default_grounding
    return e


def test_resolve_reply_mode_needs_the_model_and_every_tool_to_agree():
    assert resolve_reply_mode(True, [True]) == "advise" and resolve_reply_mode(True, [True, True]) == "advise"
    assert resolve_reply_mode(True, [True, False]) == "template"    # one literal-readback tool keeps the whole turn on templates
    assert resolve_reply_mode(False, [True]) == "template" and resolve_reply_mode(True, []) == "template"


def test_advise_runs_one_content_decode_over_all_results_and_is_traced():
    e = advise_env(ADVISE_CALL(), {"get_weather_forecast": FORECAST}, "Take an umbrella: rain is forecast for tomorrow.")
    _, result = turn(e, "will it rain tomorrow in Pune, do I need an umbrella")
    assert result.response == "Take an umbrella: rain is forecast for tomorrow."
    assert e.client.calls == 1
    prompt = e.client.prompts[0]["prompt"]
    assert "Do not calculate new numbers" in prompt and "VERIFIED" in prompt and "Tomorrow in Pune: rain, 24 degrees." in prompt
    assert e.traces.recent(1)[0].reply_mode == "advise"


def test_advise_is_ignored_when_adaptive_reply_is_off():
    e = advise_env(ADVISE_CALL(), {"get_weather_forecast": FORECAST}, "Take an umbrella.", adaptive=False)
    _, result = turn(e, "umbrella?")
    assert result.response == "Tomorrow in Pune: rain, 24 degrees." and e.client.calls == 0


def test_advise_on_a_tool_that_does_not_allow_it_is_clamped_to_the_template():
    two = ControlDecision(calls=(call("get_weather_forecast", place="Pune", date_offset=1), call("tasks", action="add", title="milk")),
                          needs_live_data=True, advise=True)
    e = advise_env(two, {"get_weather_forecast": FORECAST, "tasks": ok(spoken="Added milk to your list.")}, "x")
    _, result = turn(e, "umbrella? and add milk")
    assert result.response == "Tomorrow in Pune: rain, 24 degrees. Added milk to your list." and e.client.calls == 0


def test_ungrounded_advice_falls_back_to_the_template_after_exactly_one_decode():
    e = advise_env(ADVISE_CALL(), {"get_weather_forecast": FORECAST}, "Take an umbrella, it will be 77 degrees.")
    _, result = turn(e, "umbrella?")
    assert result.response == "Tomorrow in Pune: rain, 24 degrees." and e.client.calls == 1
    assert e.traces.recent(1)[0].reply_mode == "template"


def test_an_empty_narration_is_retried_once_but_an_ungrounded_one_is_not():
    e = env([dec([call("web_search", query="x")], live=True)], {"web_search": ok(observation="first line")}, narration="")
    e.orch._client = e.client = RecordingClient("")
    turn(e, "x")
    assert e.client.calls == 2
    e = env([dec([call("web_search", query="x")], live=True)], {"web_search": ok(observation="It is 20 degrees")}, narration="It is 99 degrees")
    e.orch._grounding_check = AssistantOrchestrator._default_grounding
    _, result = turn(e, "x")
    assert e.client.calls == 1 and result.response == "It is 20 degrees"


def test_grounding_never_runs_when_there_are_no_tool_results():
    assert AssistantOrchestrator._default_grounding("There are 7 continents.", [], "how many continents are there")
    assert not AssistantOrchestrator._default_grounding("It is 77 degrees.", ["It is 31 degrees."], "weather")


def test_the_advise_task_shares_the_content_prefix_and_leaves_narrate_alone():
    c = PromptComposer(FilePromptStore(), list(MANIFESTS.values()))
    a, n = c.content_stage("advise", "doc", "q"), c.content_stage("narrate", "doc", "q")
    assert a.system == n.system and "TASK:" in a.prompt and a.prompt != n.prompt
    assert "advice" in c._prompts.get("narrate")   # narrate.md still forbids advice, untouched


# ---- room temperature (docs/12 P5) ---------------------------------------------------------------------------------------

from datetime import timedelta   # noqa: E402

from domain.entities.room_snapshot import RoomSnapshot   # noqa: E402
from domain.policies.room_policy import room_line   # noqa: E402
from service.sensing.room_sensor import RoomSensor   # noqa: E402
from service.sensing.room_state import RoomState   # noqa: E402
from service.skills.builtin.room import UNAVAILABLE, RoomTemperatureSkill   # noqa: E402
from tpa.sensors.room_drivers import MockRoomDriver   # noqa: E402


def test_room_line_rounds_and_a_stale_or_missing_reading_is_absent():
    snap = RoomSnapshot(30.6, 47.5, NOW - timedelta(minutes=4))
    assert room_line(snap, NOW, 1800) == "ROOM: 31°C, 48% humidity (4 min ago)."
    assert room_line(RoomSnapshot(30.6, 47.5, NOW - timedelta(minutes=31)), NOW, 1800) == ""
    assert room_line(None, NOW, 1800) == ""


def test_the_sensor_stores_the_latest_reading_and_a_failed_read_never_raises():
    state = RoomState()
    sensor = RoomSensor(MockRoomDriver(28.0, 55.0), state, now=lambda: NOW)
    asyncio.run(sensor._poll_and_publish())
    assert state.get() == RoomSnapshot(28.0, 55.0, NOW)

    class Broken:
        def read(self):
            raise RuntimeError("checksum")

    state2 = RoomState()
    asyncio.run(RoomSensor(Broken(), state2, now=lambda: NOW)._poll_and_publish())
    assert state2.get() is None


def test_the_room_tool_reads_the_snapshot_and_fails_cleanly_when_it_is_stale():
    manifest = MANIFESTS["get_room_temperature"]
    state = RoomState()
    skill = RoomTemperatureSkill(manifest, state, 1800, now=lambda: NOW)
    result = asyncio.run(skill.execute(AgentContext(user_message="x")))
    assert not result.success and result.metadata["spoken"] == UNAVAILABLE
    state.set(RoomSnapshot(30.6, 47.5, NOW - timedelta(minutes=4)))
    result = asyncio.run(skill.execute(AgentContext(user_message="x")))
    assert result.success and result.metadata["spoken"] == "It's 31 degrees in here, with 48 percent humidity."
    state.set(RoomSnapshot(30.6, 47.5, NOW - timedelta(minutes=40)))
    assert not asyncio.run(skill.execute(AgentContext(user_message="x"))).success


def test_the_room_line_is_appended_to_system_context_never_assigned():
    e = env([dec()], {})
    e.orch._room_context = lambda: "ROOM: 31°C, 48% humidity (4 min ago)."
    ctx = AgentContext(user_message="I'm shivering", system_context="CALLER NOTE")
    asyncio.run(e.orch.execute(ctx))
    assert ctx.system_context == "CALLER NOTE\nROOM: 31°C, 48% humidity (4 min ago)."
    e = env([dec()], {})
    e.orch._room_context = lambda: ""
    ctx = AgentContext(user_message="hi", system_context="CALLER NOTE")
    asyncio.run(e.orch.execute(ctx))
    assert ctx.system_context == "CALLER NOTE"


def test_the_control_prompt_separates_the_room_sensor_from_the_weather():
    system = PromptComposer(FilePromptStore(), list(MANIFESTS.values())).control_stage("x").system
    assert "get_room_temperature(" in system and "indoors only" in system and "In here" in system


# ---- mixed intent (docs/12 P7) -------------------------------------------------------------------------------------------

from domain.policies.grounding_policy import states_a_measurement   # noqa: E402

PUNE = lambda a: ok(spoken="In Pune it's 24 degrees.")


def mixed_env(decision, skills, *, responder_text="That's 56.", mixed=True, adaptive=True):
    e = env([decision], skills, responder_text=responder_text, adaptive_reply=adaptive)
    e.orch._mixed_intent = adaptive and mixed
    seen = []
    orig = e.responder.execute

    async def spy(ctx):
        seen.append(ctx)
        return await orig(ctx)

    e.responder.execute = spy
    e.seen = seen
    return e


def chat_plus_weather(**kw):
    return ControlDecision(calls=(call("get_weather", place="Pune"),), needs_live_data=True, also_chat=True, **kw)


def test_mixed_turn_answers_both_halves_chat_first_tool_last_with_no_extra_decode():
    e = mixed_env(chat_plus_weather(), {"get_weather": PUNE})
    ctx, result = turn(e, "what is 7 times 8, and the weather in Pune")
    assert result.response == "That's 56. In Pune it's 24 degrees."
    assert e.responder.ran == 1 and e.client.calls == 0
    chat_ctx = e.seen[0]
    assert chat_ctx.metadata["tool_half_pending"] and chat_ctx.metadata["skip_persist"]
    assert "tool_half_pending" not in ctx.metadata                      # the caller's context is untouched
    assert e.convo.added == [("user", "what is 7 times 8, and the weather in Pune", "sess-1"),
                             ("assistant", "That's 56. In Pune it's 24 degrees.", "sess-1")]   # ONE combined turn
    assert e.traces.recent(1)[0].reply_mode == "mixed" and e.traces.recent(1)[0].also_chat is True


def test_without_the_mixed_intent_flag_the_tool_half_replies_alone():
    for kwargs in ({"mixed": False}, {"adaptive": False}):
        e = mixed_env(chat_plus_weather(), {"get_weather": PUNE}, **kwargs)
        _, result = turn(e, "x")
        assert result.response == "In Pune it's 24 degrees." and e.responder.ran == 0


def test_a_chat_half_that_states_a_measurement_is_dropped_and_the_verified_sentence_stands_alone():
    for invented in ("It's probably 77 degrees there.", "Around 60% humidity.", UNCONFIRMED_REPLY):
        e = mixed_env(chat_plus_weather(), {"get_weather": PUNE}, responder_text=invented)
        _, result = turn(e, "x")
        assert result.response == "In Pune it's 24 degrees."
    assert not states_a_measurement("7 times 8 is 56") and states_a_measurement("34 degrees")


def test_a_failing_chat_half_still_lets_the_tool_half_reply():
    e = mixed_env(chat_plus_weather(), {"get_weather": PUNE})

    async def boom(ctx):
        raise RuntimeError("model down")

    e.responder.execute = boom
    _, result = turn(e, "x")
    assert result.response == "In Pune it's 24 degrees."


def test_when_advise_applies_it_wins_and_the_chat_half_is_dropped():
    both = ControlDecision(calls=(call("get_weather_forecast", place="Pune", date_offset=1),), needs_live_data=True,
                           advise=True, also_chat=True)
    e = mixed_env(both, {"get_weather_forecast": FORECAST}, responder_text="That's 56.")
    e.orch._client = e.client = RecordingClient("Take an umbrella: rain is forecast for tomorrow.")
    _, result = turn(e, "x")
    assert result.response == "Take an umbrella: rain is forecast for tomorrow." and e.responder.ran == 0 and e.client.calls == 1


def test_a_tool_that_needs_its_own_narration_never_joins_a_mixed_turn():
    both = ControlDecision(calls=(call("web_search", query="x"),), needs_live_data=True, also_chat=True)
    e = mixed_env(both, {"web_search": ok(observation="first line")})
    _, result = turn(e, "x")
    assert e.responder.ran == 0          # control + narrate + chat would be three decodes


def test_the_streamed_path_also_answers_both_halves():
    e = mixed_env(chat_plus_weather(), {"get_weather": PUNE})
    _, text = turn(e, "x", stream=True)
    assert text == "That's 56. In Pune it's 24 degrees."


def test_the_real_responder_caps_output_adds_the_tail_line_and_saves_nothing_itself():
    from service.agent.responder import ResponderAgent
    convo = SimpleNamespace(turns=[], get_summaries_block=lambda: "", added=[], add_turn=lambda *a, **k: convo.added.append(a))
    knowledge = SimpleNamespace(get_context=lambda: "")
    responder = ResponderAgent(client=None, conversation=convo, knowledge=knowledge)
    plain, mixed = AgentContext(user_message="hi"), AgentContext(user_message="hi", metadata={"tool_half_pending": True, "skip_persist": True})
    assert responder._budget(plain) == {} and responder._budget(mixed) == {"num_predict": 40}
    assert "handled separately" not in responder.build_prompt(plain)
    prompt = responder.build_prompt(mixed)
    assert "handled separately" in prompt and prompt.index("handled separately") < prompt.index("USER: hi")
    asyncio.run(responder.on_completion(mixed, "That's 56."))
    assert convo.added == []
    asyncio.run(responder.on_completion(plain, "Hello."))
    assert len(convo.added) == 2


# ---- parallel tool calls (docs/12 P8) ------------------------------------------------------------------------------------

import time as _time   # noqa: E402


class TimedSkills:
    """Like SkillRunner: appends to ctx.skill_results when a call FINISHES; tracks how many calls overlap."""

    def __init__(self, delays, levels):
        self.delays, self.levels, self.running, self.peak = delays, levels, 0, 0
        self.skill_registry = SimpleNamespace(get=lambda name: SimpleNamespace(permission_level=self.levels[name]))

    async def execute_skill(self, ctx, name, **args):
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(self.delays[args.get("place") or args.get("title") or name])
        self.running -= 1
        ctx.skill_results.append({"skill": name, "args": args})
        return ok(spoken=f"{name}:{args.get('place') or args.get('title')}")


def parallel_env(decision, runner, *, parallel=True):
    e = env([decision], {})
    e.orch._skill_runner = runner
    e.orch._parallel_tool_calls = parallel
    return e


TWO_CITIES = ControlDecision(calls=(call("get_weather", place="Slow"), call("get_weather", place="Fast")), needs_live_data=True)


def test_auto_calls_run_at_the_same_time_and_results_stay_in_call_order():
    runner = TimedSkills({"Slow": 0.3, "Fast": 0.05}, {"get_weather": "auto"})
    e = parallel_env(TWO_CITIES, runner)
    started = _time.perf_counter()
    ctx, result = turn(e, "weather in Slow and Fast")
    assert _time.perf_counter() - started < 0.5 and runner.peak == 2          # ~max(t1, t2), not t1 + t2
    assert result.response == "get_weather:Slow get_weather:Fast"              # the slow one finished last but is first
    assert [c["args"]["place"] for c in ctx.skill_results] == ["Slow", "Fast"]


def test_without_the_flag_calls_stay_serial():
    runner = TimedSkills({"Slow": 0.05, "Fast": 0.05}, {"get_weather": "auto"})
    turn(parallel_env(TWO_CITIES, runner, parallel=False), "x")
    assert runner.peak == 1


def test_calls_that_are_not_auto_never_run_concurrently_and_keep_call_order():
    mixed = ControlDecision(calls=(call("tasks", action="add", title="milk"), call("get_weather", place="Slow"),
                                   call("get_weather", place="Fast")), needs_live_data=True)
    runner = TimedSkills({"milk": 0.1, "Slow": 0.2, "Fast": 0.05}, {"tasks": "notify", "get_weather": "auto"})
    e = parallel_env(mixed, runner)
    ctx, result = turn(e, "x")
    assert result.response == "tasks:milk get_weather:Slow get_weather:Fast"
    assert [c["skill"] for c in ctx.skill_results] == ["tasks", "get_weather", "get_weather"]
    assert runner.peak == 2                      # the two lookups overlap; the task never joins them
    approve = TimedSkills({"milk": 0.05, "Slow": 0.05, "Fast": 0.05}, {"tasks": "approve", "get_weather": "approve"})
    turn(parallel_env(mixed, approve), "x")
    assert approve.peak == 1                     # approval prompts are one at a time


def test_an_unreadable_permission_level_counts_as_not_auto():
    runner = TimedSkills({"Slow": 0.05, "Fast": 0.05}, {})                     # registry lookup raises KeyError
    turn(parallel_env(TWO_CITIES, runner), "x")
    assert runner.peak == 1
