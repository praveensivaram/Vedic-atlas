"""AssistantOrchestrator — one context-aware decision per turn, then execute (docs/10 §3.1).

    1  resolve the session id ONCE, read session state -> `ACTIVE:` line
    2  ONE control decode  ->  ControlDecision                      (ControlDecoder)
    3  dispatch_policy.resolve:  TOOLS | CLARIFY | CHAT | REFUSE | FAIL_CLOSED
    4  TOOLS   -> each call through SkillRunner (permission / rate-limit / validation / audit all apply)
                  -> reply_policy: template sentences verbatim, ONE content decode only for results that need narrating
       CHAT    -> ResponderAgent (persona, saved facts, summaries, recall; its reply veto is the claims backstop)
       others  -> a fixed sentence, no further model call
    5  claims backstop, persist the turn, write session state, record the TurnTrace

No regex reads the user's message here (principle P1); every decision above came from the model or from policy.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from datetime import datetime
from typing import AsyncIterator, Callable, Mapping, Sequence, Awaitable

from core.logging_config import logger, log_text
from domain.entities.agent_context import AgentContext
from domain.entities.agent_result import AgentResult
from domain.entities.control_decision import ControlDecision
from domain.entities.session_context import SessionContext
from domain.entities.tool_manifest import REPLY_TEMPLATE, RETURNS_DOCUMENT, ToolManifest
from domain.entities.turn_trace import TurnTrace
from domain.policies.argument_policy import normalize_latin_arguments
from domain.policies.dispatch_policy import Resolution, Route, resolve
from domain.policies.grounding_policy import is_grounded, states_a_measurement
from domain.policies.echo_policy import is_echo
from domain.policies.grounding_policy import is_grounded
from domain.policies.language_policy import language_of, looks_like_text, normalise_language
from domain.policies.reply_policy import (
    ADVISE_MODE, FAILED, NARRATE, SPOKEN, UNCONFIRMED_REPLY, classify_call, failure_text, resolve_reply_mode, spoken_text,
)
from domain.ports.memory_repository_port import MemoryRepositoryPort
from domain.ports.trace_repository_port import TraceRepositoryPort
from service.agent.base_agent import BaseAgent
from service.agent.control_decoder import ControlDecoder, DecodeResult
from service.agent.tool_execution import ExecutedCall, TurnOutcome, execute_call, narrate_with_grounding
from service.prompting.prompt_composer import PromptComposer
from service.session.session_state import SessionStateService
from service.skills.skill_runner import SkillRunner

_ERROR_REPLY = "Something went wrong on my end. Could you try again?"
_DOCUMENT_FALLBACK_TOKENS = 60  # about two spoken sentences: all a failed narration of a document may fall back to
_ADVISE_HEADER = "VERIFIED - quote the numbers, names and dates below exactly; do not alter them:"
_DECIDED = {
    Route.TOOLS: "tool", Route.CHAT: "chat", Route.CLARIFY: "clarify", Route.REFUSE: "refuse", Route.FAIL_CLOSED: "fail",
}


class AssistantOrchestrator(BaseAgent):
    def __init__(
        self,
        *,
        decoder: ControlDecoder,
        composer: PromptComposer,
        skill_runner: SkillRunner,
        manifests: Mapping[str, ToolManifest],
        responder: BaseAgent,
        client,
        conversation=None,
        claims_action: Callable[[str], bool] | None = None,
        session_state: SessionStateService | None = None,
        traces: TraceRepositoryPort | None = None,
        memory: MemoryRepositoryPort | None = None,
        model: str | None = None,
        narrate_num_predict: int = 90,
        narrate_temperature: float = 0.2,
        grounding_check: Callable[[str, Sequence[str], str], bool] | None = None,
        conversation_intent: Callable[[str], Awaitable[float]] | None = None,
        conversation_intent_min: float = 0.82,
        conversation_intent_soft_min: float = 0.76,
        adaptive_reply: bool = False,
        mixed_intent: bool = False,
        parallel_tool_calls: bool = False,
        room_context: Callable[[], str] | None = None,
        multilingual: bool = True,
        reject_unknown_language: bool = True,
        unknown_language_reply: str = "Sorry, I didn't understand that. Could you say it again?",
        min_language_letters: int = 8,
    ):
        super().__init__(
            name="assistant", model=model,
            description="Decides each turn once: use a tool, ask a question, chat, or say it cannot look something up.",
        )
        self._decoder = decoder
        self._composer = composer
        self._skill_runner = skill_runner
        self._manifests = manifests
        self._responder = responder
        self._client = client
        self._conversation = conversation
        self._claims_action = claims_action
        self._session_state = session_state
        self._traces = traces
        self._memory = memory
        self._narrate_num_predict = narrate_num_predict
        self._narrate_temperature = narrate_temperature
        self._grounding_check = grounding_check or self._default_grounding
        # "What are we discussing?" is about the conversation itself, which only chat (with its history and memory) can
        # answer. The routing model has answered it with a clarifying question and even a web search. A question that is
        # nearly one of the example phrasings (by embedding, not keywords) goes to chat whatever the router said; a softer
        # match (a typo, a polite wrapper: "can you tell me what are we discusing lately") goes to chat only INSTEAD OF a
        # clarifying question or a refusal, because asking "what do you mean?" is never the right answer to it.
        self._conversation_intent = conversation_intent
        self._intent_min = conversation_intent_min
        self._intent_soft_min = conversation_intent_soft_min
        # docs/12: False = the control decision's `advise` / `also_chat` are ignored entirely.
        self._adaptive_reply = adaptive_reply
        self._mixed_intent = adaptive_reply and mixed_intent
        self._parallel_tool_calls = parallel_tool_calls
        self._room_context = room_context   # () -> the ROOM line, or "" when there is no fresh reading
        # Multilingual: the control decode names the language; the deterministic detector overrides it for the
        # scripts it knows. An unnamed language on a message that reads as language gets a fixed "I did not
        # understand" instead of a guessed English answer.
        self._multilingual = multilingual
        self._reject_unknown_language = reject_unknown_language
        self._unknown_language_reply = unknown_language_reply
        self._min_language_letters = min_language_letters

    @staticmethod
    def _default_grounding(reply: str, results: Sequence[str], user_message: str) -> bool:
        if not results:
            # No tool data to check against: general knowledge ("there are 7 continents") is not a claim about tool
            # data, and the allowed-number pool would otherwise be only the question and today's date.
            return True
        today = datetime.now().strftime("%A %d %B %Y %H:%M")
        return is_grounded(reply, [*results, user_message, today])

    # ---- turn preparation ---------------------------------------------------

    def _add_room_context(self, ctx: AgentContext) -> None:
        """Append (never assign: the API caller may have set system_context) the room reading for the chat prompt."""
        if self._room_context is None:
            return
        try:
            line = self._room_context()
        except Exception as e:   # context is an extra; never let it take the turn down
            logger.warning(f"[assistant] room context unavailable: {type(e).__name__}")
            return
        ctx.system_context = "\n".join(part for part in (ctx.system_context, line) if part)

    def _begin(self, ctx: AgentContext) -> None:
        ctx.current_agent = self.name
        if not ctx.agent_chain or ctx.agent_chain[-1] != self.name:
            ctx.agent_chain.append(self.name)
        if ctx.session_id is None and self._conversation is not None:   # resolved exactly once per turn
            try:
                ctx.session_id = self._conversation.current_session_id()
            except Exception as e:
                logger.warning(f"[assistant] session id unavailable: {e}")

    async def _decide(self, ctx: AgentContext) -> tuple[DecodeResult, Resolution, SessionContext | None, str, TurnOutcome]:
        state = None
        if self._session_state is not None and ctx.session_id is not None:
            state = self._session_state.get(ctx.session_id, ctx.speaker_id)
        active = self._session_state.render_active(state) if (self._session_state and state) else ""
        history = self._conversation.turns if self._conversation is not None else []
        logger.info(f"[turn] user={log_text(ctx.user_message)} voice={ctx.from_voice}")
        result = await self._decoder.decide(ctx.user_message, history, active)
        normalized, arg_notes = normalize_latin_arguments(result.decision, self._manifests)
        if arg_notes:
            result = replace(result, decision=normalized)
            logger.info(f"[control] normalized Latin tool args: {arg_notes}")
        logger.info(
            f"[control] raw={log_text(result.raw)} -> calls={len(result.decision.calls)} advise={result.decision.advise} "
            f"also_chat={result.decision.also_chat} needs_live_data={result.decision.needs_live_data} "
            f"flags: adaptive_reply={self._adaptive_reply} mixed_intent={self._mixed_intent} parallel={self._parallel_tool_calls}"
        )
        resolution = resolve(result.decision, self._manifests, user_message=ctx.user_message)
        if self._conversation_intent is not None:
            try:
                score = await self._conversation_intent(ctx.user_message)
                strong = score >= self._intent_min
                soft = score >= self._intent_soft_min and resolution.route in (Route.CLARIFY, Route.REFUSE)
                if strong or soft:
                    ctx.metadata["about_conversation"] = True   # chat then answers from the newest conversations
                    if resolution.route is not Route.CHAT:
                        logger.info(f"[control] question about the conversation itself (score {score:.2f}): route "
                                    f"{resolution.route.value} -> chat")
                        resolution = Resolution(Route.CHAT)
                    # The fence: this override may only ever move a turn TO chat, never to a tool route.
                    assert resolution.route is Route.CHAT, "conversation-intent override produced a non-chat route"
            except Exception as e:   # the check is an extra; never let it take the turn down
                logger.warning(f"[control] conversation-intent check failed: {type(e).__name__}")
        outcome = TurnOutcome(reply=None)
        outcome.notes.extend(arg_notes)
        outcome.timings_ms["control"] = result.ms
        outcome.prompt_tokens["control"] = result.prompt_tokens
        if result.trimmed:
            outcome.notes.append(f"control prompt trimmed: {list(result.trimmed)}")
        if not result.decision.valid:
            outcome.notes.append(f"control decision invalid: {result.decision.note}")
        # The model's own label for the turn, with the deterministic detector overriding it for the scripts it
        # reads exactly (so Telugu/Hindi/German can never be mislabelled).
        language = language_of(ctx.user_message) or normalise_language(result.decision.language)
        if self._multilingual:
            if language:
                ctx.metadata["language"] = language
            elif result.decision.language is not None:
                # The model carried a language key but could not name a language ("unknown"). Only an explicit
                # answer can trigger the "I didn't understand" reply: an older/absent key is permissive.
                ctx.metadata["language_unknown"] = True
        logger.info(
            f"[control] route={resolution.route.value} live={result.decision.needs_live_data} "
            f"calls={[c.get('tool') for c in resolution.calls]} ms={result.ms} tokens={result.prompt_tokens} "
            f"lang={log_text(language)} active={log_text(active)} user={log_text(ctx.user_message)}"
        )
        return result, resolution, state, active, outcome

    # ---- execution ----------------------------------------------------------

    def _is_auto(self, tool: str) -> bool:
        """Does this call proceed without asking the user? Read from the skill actually registered (configuration can
        override a manifest's level); an unknown or unreadable level counts as NOT auto, so it stays serial."""
        try:
            registry = getattr(self._skill_runner, "skill_registry", None)
            level = registry.get(tool).permission_level if registry is not None else self._manifests[tool].permission_level
            return str(getattr(level, "value", level)) == "auto"
        except Exception:
            return False

    async def _execute_calls(self, ctx: AgentContext, calls: Sequence[dict], outcome: TurnOutcome) -> None:
        """Run the calls. Normally one after another. With parallel_tool_calls, the `auto` ones (lookups) run at the same
        time and everything else stays serial: the approval broker waits up to 60 s on one user decision, so two
        approval prompts at once would be unusable, especially by voice. Results are always recorded in CALL order."""
        auto = [i for i, c in enumerate(calls) if self._is_auto(c["tool"])] if self._parallel_tool_calls else []
        if len(auto) < 2:
            for call in calls:
                outcome.calls.append(await execute_call(self._skill_runner, ctx, call, private=self._manifests[call["tool"]].private))
            return

        executed: dict[int, "ExecutedCall"] = {}
        entries: dict[int, list[dict]] = {}   # the skill_results entries each call added, to be put back in call order
        base = len(ctx.skill_results)

        async def run_isolated(i: int) -> None:   # its own skill_results list: the runner appends on completion
            own = replace(ctx, skill_results=[])
            call = calls[i]
            executed[i] = await execute_call(self._skill_runner, own, call, private=self._manifests[call["tool"]].private)
            entries[i] = own.skill_results

        await asyncio.gather(*(run_isolated(i) for i in auto))
        for i, call in enumerate(calls):
            if i in executed:
                continue
            before = len(ctx.skill_results)   # serial: the real context, so approvals and their bookkeeping work as ever
            executed[i] = await execute_call(self._skill_runner, ctx, call, private=self._manifests[call["tool"]].private)
            entries[i] = ctx.skill_results[before:]
        ctx.skill_results[base:] = [e for i in range(len(calls)) for e in entries[i]]
        outcome.calls.extend(executed[i] for i in range(len(calls)))
        outcome.notes.append(f"{len(auto)} calls ran in parallel")

    async def _run_tools(
        self, ctx: AgentContext, calls: Sequence[dict], outcome: TurnOutcome, *, advise: bool = False,
    ) -> str:
        t0 = time.perf_counter()
        await self._execute_calls(ctx, calls, outcome)
        outcome.timings_ms["execute"] = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        reply = await self._assemble_reply(ctx, outcome, advise=advise)
        outcome.timings_ms["reply"] = int((time.perf_counter() - t1) * 1000)
        return reply

    def _runs_mixed(self, res: Resolution, outcome: TurnOutcome) -> bool:
        """`also_chat` runs the chat half beside the tools only when it costs no extra decode and loses no guarantee:
        - not when `advise` applies: that is one decode over the results, and advice cannot answer free knowledge, so
          `advise` wins and the chat half is dropped;
        - not when a called tool needs a narration decode (llm reply mode): that would make three decodes."""
        if not (self._mixed_intent and res.also_chat):
            return False
        manifests = [self._manifests[c["tool"]] for c in res.calls]
        if res.advise and all(m.allow_advice for m in manifests):
            outcome.notes.append("also_chat dropped: advise applies")
            return False
        if any(m.reply_mode != REPLY_TEMPLATE for m in manifests):
            outcome.notes.append("also_chat dropped: a tool needs its own narration decode")
            return False
        return True

    async def _run_tools_and_chat(self, ctx: AgentContext, res: Resolution, outcome: TurnOutcome) -> str:
        """The tool calls and the chat half at the same time (the flag asserts they are independent), so the network
        fetch hides under the chat decode. Three protections, because the chat half starts before any tool result:
        a ~40-token cap, a prompt line telling it to leave the other part alone, and the verified tool sentence last."""
        chat_ctx = replace(
            ctx, metadata={**ctx.metadata, "tool_half_pending": True, "skip_persist": True},
            agent_chain=list(ctx.agent_chain), skill_results=[],
        )

        async def chat_half():
            started = time.perf_counter()
            try:
                return await self._responder.execute(chat_ctx)
            finally:
                outcome.timings_ms["chat"] = int((time.perf_counter() - started) * 1000)

        tool_reply, chat = await asyncio.gather(
            self._run_tools(ctx, res.calls, outcome), chat_half(), return_exceptions=True,
        )
        if isinstance(tool_reply, BaseException):
            raise tool_reply
        chat_text = ""
        if isinstance(chat, BaseException):
            logger.warning(f"[assistant] chat half failed, replying with the tool half only: {type(chat).__name__}")
            outcome.notes.append("chat half failed")
        else:
            chat_text = (chat.response or "").strip()
            if chat_text == UNCONFIRMED_REPLY or states_a_measurement(chat_text):
                # A claim of an action, or a measurement: the chat half started before the tool result existed, so
                # either one can only be invented. The verified tool sentence stands alone.
                logger.warning(f"[tool-guard] chat half of a mixed turn dropped: {log_text(chat_text, 160)}")
                outcome.notes.append("chat half dropped")
                chat_text = ""
        private = any(self._manifests[c.tool].private for c in outcome.calls)
        logger.info(f"[mixed] chat half={'<private>' if private else repr(chat_text)} kept={bool(chat_text)}")
        if chat_text:
            outcome.reply_mode = "mixed"
        return " ".join(part for part in (chat_text, tool_reply) if part)

    async def _assemble_reply(self, ctx: AgentContext, outcome: TurnOutcome, *, advise: bool = False) -> str:
        """Template sentences are kept verbatim and never reach the model; only results that need narrating share
        ONE content decode, spliced in at the position of the first of them.

        ADVISE (the model asked, adaptive_reply is on, and every successful call's tool allows it): all successful
        results, template sentences included, go into that one decode as the `advise` task. If it fails or is
        ungrounded the reply is exactly what TEMPLATE would have said."""
        calls = outcome.calls
        kinds = [
            classify_call(ok=c.ok, reply_mode=self._manifests[c.tool].reply_mode, final=c.final) for c in calls
        ]
        mode = resolve_reply_mode(
            advise and self._adaptive_reply, [self._manifests[c.tool].allow_advice for c in calls if c.ok],
        )
        narration: str | None = None
        task, header = "narrate", ""
        if mode == ADVISE_MODE:
            task, header = "advise", _ADVISE_HEADER
            kinds = [FAILED if k == FAILED else NARRATE for k in kinds]
        to_narrate = [c for c, k in zip(calls, kinds) if k == NARRATE]
        if to_narrate:
            if mode == ADVISE_MODE:   # each result is already a short spoken sentence; cap it by its manifest as ever
                results = [self._composer.cap_tokens(spoken_text(c.spoken, c.observation), self._manifests[c.tool].max_result_tokens) for c in to_narrate]
            else:
                results = [self._composer.cap_tokens(c.observation, self._manifests[c.tool].max_result_tokens) for c in to_narrate]
            narration = await narrate_with_grounding(
                client=self._client, composer=self._composer, grounding_check=self._grounding_check, ctx=ctx,
                results=results, model=self.model, num_predict=self._narrate_num_predict,
                temperature=self._narrate_temperature, outcome=outcome, task=task, header=header,
                private=any(self._manifests[c.tool].private for c in to_narrate),
            )
            if mode == ADVISE_MODE:
                outcome.reply_mode = ADVISE_MODE if narration is not None else ""
                if narration is None:
                    outcome.notes.append("advise fell back to the tool sentences")
        pieces: list[str] = []
        templated: list[int] = []   # positions holding an English template/fallback sentence
        narration_placed = False
        for c, kind in zip(calls, kinds):
            if kind == FAILED:
                # The skill's own sentence when it gave one ("Which app do you mean?", "I can't check your email yet because
                # it isn't set up"); the generic text with the raw error only for a failure that came with no sentence.
                pieces.append(c.spoken or failure_text(c.error))
                templated.append(len(pieces) - 1)
            elif kind == SPOKEN:
                pieces.append(spoken_text(c.spoken, c.observation))
                templated.append(len(pieces) - 1)
            elif narration is not None:
                if not narration_placed:
                    pieces.append(narration)
                    narration_placed = True
            else:  # narration failed or was ungrounded: say plainly what the tool returned
                text = spoken_text(c.spoken, c.observation)
                if self._manifests[c.tool].returns == RETURNS_DOCUMENT:
                    # A document's first line can be the whole document. This text is spoken AND saved into history,
                    # which feeds later prompts, so it is cut: a document must not leak into any control prompt.
                    text = self._composer.cap_tokens(text, _DOCUMENT_FALLBACK_TOKENS)
                pieces.append(text)
                templated.append(len(pieces) - 1)
        language = ctx.metadata.get("language")
        if templated and language and language.lower() != "english":
            # Template sentences are English by construction (the adapter builds them). Translate them only now,
            # as one finished sentence, so the verbs/names/numbers stay the tool's and English turns pay nothing.
            joined = " ".join(pieces[i] for i in templated)
            translated = await self._localize(joined, language, ctx, outcome)
            if translated:
                for i in templated:
                    pieces[i] = ""
                pieces[templated[0]] = translated
        return " ".join(piece for piece in pieces if piece)

    async def _localize(self, text: str, language: str, ctx: AgentContext, outcome: TurnOutcome) -> str | None:
        """Translate one finished template sentence into the turn's language.

        The prompt carries the sentence only (never the user's question). A reply that repeats the user's
        message, or whose numbers are not in the sentence, is rejected: the English sentence is kept."""
        stage = self._composer.translate_stage(text, language)
        outcome.prompt_tokens["translate"] = stage.tokens
        for attempt in range(2):
            try:
                translated = (await self._client.complete(
                    prompt=stage.prompt, system=stage.system, model=self.model,
                    temperature=self._narrate_temperature if attempt == 0 else 0.6,
                    num_predict=max(self._narrate_num_predict, 120),
                )).strip()
            except Exception as e:
                logger.warning(f"[language] template translation failed: {e}")
                return None
            if not translated:
                continue
            if is_echo(translated, ctx.user_message):
                logger.warning(f"[language] template translation echoed the user: {log_text(translated, 120)}")
                outcome.notes.append("template translation echoed the user")
                continue
            if not is_grounded(translated, [text]):
                logger.warning(f"[language] rejected ungrounded template translation: {log_text(translated, 120)}")
                outcome.notes.append("template translation rejected (ungrounded)")
                return None
            outcome.notes.append(f"template reply translated to {language}")
            return translated
        return None

    def _backstop(self, reply: str, outcome: TurnOutcome) -> str:
        """claims_action on EVERY reply path. A claim is only legitimate if a tool actually succeeded this turn."""
        if self._claims_action is not None and not any(c.ok for c in outcome.calls) and self._claims_action(reply):
            logger.warning(f"[tool-guard] reply claimed an action with no successful tool call: {log_text(reply, 160)}")
            outcome.notes.append("reply claimed an action; refused")
            return UNCONFIRMED_REPLY
        return reply

    async def _tools_reply(self, ctx: AgentContext, res: Resolution, outcome: TurnOutcome) -> str:
        names = [c["tool"] for c in res.calls]
        permits = [self._manifests[n].allow_advice for n in names]
        mixed = self._runs_mixed(res, outcome)
        logger.info(
            f"[plan] tools={names} multi_call={len(names) > 1} model_advise={res.advise} tools_allow_advice={permits} "
            f"-> reply_mode={'advise' if (res.advise and self._adaptive_reply and all(permits)) else 'template'} "
            f"also_chat={res.also_chat} -> mixed_path={mixed}"
        )
        if mixed:
            reply = await self._run_tools_and_chat(ctx, res, outcome)
        else:
            reply = await self._run_tools(ctx, res.calls, outcome, advise=res.advise)
        if res.follow_up:   # a held call or the model's own question: asked AFTER what could be answered
            logger.info(f"[plan] follow-up question after the tool answer: {log_text(res.follow_up)}")
            outcome.notes.append("follow-up question asked")
            reply = f"{reply} {res.follow_up}".strip()
        return reply

    def _log_reply(self, res: Resolution, outcome: TurnOutcome, reply: str, started: float) -> None:
        private = any(self._manifests[c.tool].private for c in outcome.calls)
        logger.info(
            f"[reply] route={res.route.value} mode={outcome.reply_mode or 'template'} decodes={sorted(k for k in outcome.timings_ms if k in ('control', 'narrate', 'chat'))} "
            f"timings_ms={outcome.timings_ms} total_ms={int((time.perf_counter() - started) * 1000)} notes={outcome.notes} "
            f"reply={'<private>' if private else log_text(reply, 300)}"
        )

    def _unknown_language(self, ctx: AgentContext, resolution: Resolution, outcome: TurnOutcome) -> str | None:
        """The fixed "I didn't understand" reply, or None when this turn may go ahead.

        Only a plain chat turn can be rejected: if the model picked a tool or asked a clarification, it
        demonstrably understood the message regardless of whether it could name the language. The
        `looks_like_text` guard keeps "2+2" and "ok" from being called unreadable."""
        if not (self._multilingual and self._reject_unknown_language):
            return None
        if resolution.route is not Route.CHAT or not ctx.metadata.get("language_unknown"):
            return None
        if not looks_like_text(ctx.user_message, self._min_language_letters):
            return None
        logger.info(f"[language] no language named for {log_text(ctx.user_message, 80)}; asking the user to repeat")
        outcome.notes.append("unknown language: asked the user to repeat")
        return self._unknown_language_reply

    # ---- persistence & trace ------------------------------------------------

    def _persist(self, ctx: AgentContext, reply: str) -> None:
        if self._conversation is not None:
            self._conversation.add_turn("user", ctx.user_message, session_id=ctx.session_id)
            self._conversation.add_turn("assistant", reply, session_id=ctx.session_id)
        if self._memory is not None:
            try:
                self._memory.record(agent_name=self.name, action="turn", context={}, user_message=ctx.user_message)
            except Exception as e:
                logger.debug(f"[assistant] memory record failed: {e}")

    def _record_trace(
        self, ctx: AgentContext, route: Route, decision: ControlDecision, outcome: TurnOutcome, reply: str,
        started: float, state: SessionContext | None, active: str,
    ) -> None:
        if self._traces is None:
            return
        reply_mode = outcome.reply_mode or ("template" if route is Route.TOOLS else "")
        inherited = sorted({
            k for c in outcome.calls for k, v in c.args.items() if state is not None and state.slots.get(k) == v
        })
        try:
            self._traces.record(TurnTrace(
                request_id=ctx.request_id, agent=self.name, user_message=ctx.user_message[:300], reply=reply[:400],
                decided=_DECIDED[route], narrated=outcome.narrated, total_ms=int((time.perf_counter() - started) * 1000),
                calls=[
                    {"tool": c.tool, "args": c.args, "ok": c.ok, "ms": c.ms, "result": "" if self._manifests[c.tool].private else (c.observation or "")[:300], "error": c.error}
                    for c in outcome.calls
                ],
                prompt_tokens=outcome.prompt_tokens, timings_ms=outcome.timings_ms, notes=list(outcome.notes),
                state_used=bool(active), slots_inherited=inherited, needs_live_data=decision.needs_live_data,
                clarified=route is Route.CLARIFY, reply_mode=reply_mode,
                also_chat=self._adaptive_reply and decision.also_chat,
            ))
        except Exception as e:  # tracing must never break a turn
            logger.warning(f"[assistant] trace record failed: {e}")

    def _write_state(self, ctx: AgentContext, outcome: TurnOutcome) -> None:
        if self._session_state is not None and ctx.session_id is not None and outcome.calls:
            self._session_state.record_success(ctx.session_id, ctx.speaker_id, outcome.calls)

    # ---- public -------------------------------------------------------------

    async def warmup(self) -> None:
        """Throw-away decodes so each fixed prompt prefix (control, chat, narrate) is already in the prompt cache for
        the first real turn. Each is warmed on its own: one failing does not skip the others."""
        try:
            await self._decoder.decide("hello", (), "")
            logger.info("[assistant] control prefix warmed")
        except Exception as e:
            logger.warning(f"[assistant] warmup skipped: {e}")
        # Chat and narrate prompts straight to the client, one token each: nothing is saved to the conversation.
        chat_ctx = AgentContext(user_message="hello", from_voice=True)
        narrate = self._composer.narrate_stage("hello", ["ok"])
        for stage, system, prompt in (
            ("chat", self._responder.build_system_prompt(chat_ctx), self._responder.build_prompt(chat_ctx)),
            ("narrate", narrate.system, narrate.prompt),
        ):
            try:
                await self._client.complete(prompt=prompt, system=system, model=self.model, num_predict=1)
                logger.info(f"[assistant] {stage} prefix warmed")
            except Exception as e:
                logger.warning(f"[assistant] {stage} warmup skipped: {e}")


    async def execute(self, ctx: AgentContext) -> AgentResult:
        self._begin(ctx)
        started = time.perf_counter()
        try:
            result, res, state, active, outcome = await self._decide(ctx)
            unknown = self._unknown_language(ctx, res, outcome)
            if unknown is not None:
                self._persist(ctx, unknown)
                self._write_state(ctx, outcome)
                self._record_trace(ctx, res.route, result.decision, outcome, unknown, started, state, active)
                return AgentResult(agent_name=self.name, response=unknown)
            if res.route is Route.CHAT:
                self._add_room_context(ctx)
                chat = await self._responder.execute(ctx)
                logger.info(f"[reply] route=chat room_context={'ROOM' in ctx.system_context} reply={log_text(chat.response, 300)}")
                self._record_trace(ctx, res.route, result.decision, outcome, chat.response, started, state, active)
                chat.delegated_to = self._responder.name
                return chat
            reply = await self._tools_reply(ctx, res, outcome) if res.route is Route.TOOLS else res.text
        except Exception as e:  # never let an orchestration bug take the turn down
            logger.exception(f"[assistant] turn failed: {e}")
            return AgentResult(agent_name=self.name, response=_ERROR_REPLY)

        reply = self._backstop(reply, outcome)
        self._log_reply(res, outcome, reply, started)
        self._persist(ctx, reply)
        self._write_state(ctx, outcome)
        self._record_trace(ctx, res.route, result.decision, outcome, reply, started, state, active)
        return AgentResult(agent_name=self.name, response=reply, skill_calls=list(ctx.skill_results))

    async def execute_stream(self, ctx: AgentContext, cancel_event: asyncio.Event | None = None) -> AsyncIterator[str]:
        self._begin(ctx)
        started = time.perf_counter()
        try:
            result, res, state, active, outcome = await self._decide(ctx)
            unknown = self._unknown_language(ctx, res, outcome)
            if unknown is not None:
                self._persist(ctx, unknown)
                self._write_state(ctx, outcome)
                self._record_trace(ctx, res.route, result.decision, outcome, unknown, started, state, active)
                yield unknown
                return
            if res.route is Route.CHAT:
                self._add_room_context(ctx)
                async for chunk in self._responder.execute_stream(ctx, cancel_event=cancel_event):
                    yield chunk
                self._record_trace(ctx, res.route, result.decision, outcome, "", started, state, active)
                return
            reply = await self._tools_reply(ctx, res, outcome) if res.route is Route.TOOLS else res.text
        except Exception as e:
            logger.exception(f"[assistant] turn failed: {e}")
            yield _ERROR_REPLY
            return

        # A tool, clarification, refusal or failure has no model text to stream: the reply exists only now.
        reply = self._backstop(reply, outcome)
        self._log_reply(res, outcome, reply, started)
        self._persist(ctx, reply)
        self._write_state(ctx, outcome)
        self._record_trace(ctx, res.route, result.decision, outcome, reply, started, state, active)
        yield reply
