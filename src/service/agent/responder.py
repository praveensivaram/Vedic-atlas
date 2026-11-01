"""ResponderAgent — default conversational agent (persona + history + knowledge).

Donor: veda/agents/responder.py, read in full and adapted:
  - DROPPED the `ctx.recognized_user` branch entirely — that's face
    recognition, out of scope. The donor's greeting-by-name behaviour
    only existed because vision could identify who was talking.
  - DROPPED the "AGENT_BOUNDARY" paragraph about the code agent handling
    project/file changes — the code agent is out of scope.
  - `ctx.metadata.get("from_voice")` replaced with `ctx.from_voice` — the
    donor stashed this in an untyped metadata dict; Batch 2 promoted it
    to an explicit field on AgentContext specifically so this check
    doesn't rely on a magic string key.
  - Depends on ConversationManager (service layer) and KnowledgeStorePort
    (domain layer) rather than concrete donor classes.
"""

import asyncio
from datetime import datetime
from typing import AsyncIterator, Callable

from domain.entities.agent_context import AgentContext
from domain.entities.agent_result import AgentResult
from domain.ports.inference_port import InferencePort
from domain.ports.knowledge_store_port import KnowledgeStorePort
from domain.ports.memory_repository_port import MemoryRepositoryPort
from core.logging_config import logger, log_text
from domain.policies.claim_guard_policy import SentenceClaimGuard
from domain.policies.echo_policy import fallback_reply, is_echo
from domain.policies.language_policy import language_of
from domain.policies.reply_filler_policy import FillerStreamFilter, strip_filler
from domain.policies.reply_policy import UNCONFIRMED_REPLY
from domain.policies.token_budget_policy import estimate_tokens
from service.agent.base_llm_agent import LLMAgent
from service.agent.persona import VEDA_SYSTEM_PROMPT
from service.conversation.conversation_manager import ConversationManager
from service.memory.semantic_recall import SemanticRecall


_FOLLOW_UP_WORDS = 6   # a message this short probably leans on the one before it


class ResponderAgent(LLMAgent):
    """General conversational agent wired to Veda's persona + persistent memory."""

    def __init__(
        self,
        client: InferencePort,
        conversation: ConversationManager,
        knowledge: KnowledgeStorePort,
        model: str | None = None,
        max_history_turns: int = 20,
        memory: MemoryRepositoryPort | None = None,
        recall: "SemanticRecall | None" = None,
        persona: str | None = None,
        reply_veto: Callable[[str], bool] | None = None,
        chat_budget_tokens: int | None = None,
        history_turn_chars: int = 200,
    ):
        super().__init__(
            name="responder",
            description=(
                "General conversation, persona replies, Q&A, small talk, and anything "
                "the user wants to chat about. The safe default when no specialist fits."
            ),
            system_prompt=persona or VEDA_SYSTEM_PROMPT,
            client=client,
            model=model,
            memory=memory,
        )
        self.conversation = conversation
        self.knowledge = knowledge
        self.max_history_turns = max_history_turns
        self._recall = recall
        # True = this reply asserts a tool action that did not happen. Only set where the chat path is reached with
        # no tool having run (the orchestrator), so every claim here is unbacked. See docs/10 §4.7.
        self._reply_veto = reply_veto
        # Chat prompt budget (PromptBudgets.chat) and the per-turn clip. History is the only part that grows, so it is
        # what gets trimmed (oldest first) to fit; None = unbounded, the previous behaviour.
        self._chat_budget = chat_budget_tokens
        self._turn_chars = history_turn_chars

    _TOOL_HALF_PREDICT = 40   # mixed turn: no room to editorialise about the tool half, and it finishes sooner

    def _budget(self, ctx: AgentContext) -> dict:
        if ctx.metadata.get("tool_half_pending"):
            return {"num_predict": self._TOOL_HALF_PREDICT}
        return super()._budget(ctx)

    def build_system_prompt(self, ctx: AgentContext) -> str:
        # Persona only: byte-identical every turn so the backend's KV prefix
        # cache is reused. Volatile context lives in build_prompt — PERF_BRIEF §5.2.
        return self.system_prompt

    @staticmethod
    def _time_context() -> str:
        now = datetime.now()
        hour = now.hour
        if hour < 5:
            part = "late night"
        elif hour < 12:
            part = "morning"
        elif hour < 17:
            part = "afternoon"
        elif hour < 21:
            part = "evening"
        else:
            part = "night"
        return f"CURRENT TIME: {now.strftime('%H:%M on %A')} ({part})."

    def build_prompt(self, ctx: AgentContext) -> str:
        parts: list[str] = []
        # Least-volatile context first so the cacheable prefix extends as far as
        # possible; every-turn blocks go last, right before the question.
        # Saved facts (the `remember` tool) are always included; recalled summaries are added when relevant.
        for block in (self.knowledge.get_context(), ctx.metadata.get("semantic_knowledge_context")):
            if block:
                parts.append(block)
        if ctx.system_context:
            parts.append(f"CURRENT CONTEXT:\n{ctx.system_context}")
        # Deliberately no cross-agent activity block here: it only holds internal
        # routing records (supervisor/responder), which the model mistook for user
        # tasks and echoed back. Activity is still recorded via _record_to_memory.
        tail: list[str] = [self._time_context()]
        if ctx.from_voice:
            tail.append(
                "DELIVERY: this request arrived via voice — keep the reply to one or two "
                "short sentences, zero lists, nothing the user wouldn't hear in one breath."
            )
        if ctx.metadata.get("tool_half_pending"):
            tail.append(
                "NOTE: another part of this message is being handled separately. Answer only the rest. "
                "Do not mention that other part and do not answer it yourself."
            )
        tail.append(f"USER: {ctx.user_message}")
        # The control decode names the language for any language; the detector is the safety override for the
        # scripts it knows exactly. Naming it beats "reply in the user's language": a small model follows a
        # concrete instruction, and the abstract rule is easy to ignore in a long English persona.
        language = ctx.metadata.get("language") or language_of(ctx.user_message)
        if language:
            tail.append(f"The user wrote in {language}. Write your entire reply in {language}.")
        if ctx.metadata.get("no_echo"):
            tail.append(
                "Your previous attempt repeated the user's message. Answer it now in your own words; "
                "do not repeat or restate it."
            )
        tail.append(
            "Respond directly to the user in the same language as USER's message "
            "(English only if USER wrote English). Answer it; never repeat or restate the user's message. "
            "Be concise and conversational."
        )
        fits = None
        if self._chat_budget is not None:
            system_tokens = estimate_tokens(self.system_prompt)

            def fits(history: str) -> bool:  # measures the assembled prompt, so separators and rounding are counted
                return system_tokens + estimate_tokens("\n\n".join(parts + ([history] if history else []) + tail)) <= self._chat_budget

        history_block = self._render_history(fits)
        if history_block:
            parts.append(history_block)
        parts.extend(tail)
        return "\n\n".join(parts)

    async def _prepare_semantic_context(self, ctx: AgentContext) -> None:
        if self._recall is None or not self._recall.enabled:
            return
        shown = {str(t.id) for t in self.conversation.turns if getattr(t, "id", None) is not None}

        async def recalled(query: str):
            hits = await self._recall.recall(query)
            # An exchange already in the recent history above would only be repeated.
            return [h for h in hits if not (h.source == "exchange" and h.ref_id in shown)]

        if ctx.metadata.get("about_conversation") or await self._recall.asks_about_conversation(ctx.user_message):
            block = self.conversation.recent_summaries_block()
            if block:
                ctx.metadata["semantic_knowledge_context"] = block
                return
        hits = await recalled(ctx.user_message)
        if not hits and len(ctx.user_message.split()) <= _FOLLOW_UP_WORDS:
            # "what did he say?" names nothing: search it together with the question before it. Only tried when the
            # message alone found nothing, so an unrelated short question does not drag the last topic back in.
            previous = next((t.content for t in reversed(self.conversation.turns) if t.role == "user"), "")
            if previous:
                hits = await recalled(f"{previous} {ctx.user_message}")
        block = self._recall.as_context_block(hits)
        if block:
            ctx.metadata["semantic_knowledge_context"] = block

    async def execute(self, ctx: AgentContext) -> AgentResult:
        await self._prepare_semantic_context(ctx)
        result = await super().execute(ctx)
        result.response = self._vet(strip_filler(result.response))
        return result

    def _vet(self, response: str) -> str:
        if self._reply_veto is not None and response and self._reply_veto(response):
            logger.warning(f"[tool-guard] chat reply claimed an action with no tool call: {log_text(response, 160)}")
            return UNCONFIRMED_REPLY
        return response

    async def execute_stream(self, ctx: AgentContext, cancel_event=None) -> AsyncIterator[str]:
        await self._prepare_semantic_context(ctx)
        # Stock "let me know if you need anything else" sentences are dropped as they stream, so they are
        # neither spoken nor left in the text the user sees.
        if self._reply_veto is None:
            filler_filter = FillerStreamFilter()
            async for chunk in super().execute_stream(ctx, cancel_event=cancel_event):
                text = filler_filter.feed(chunk)
                if text:
                    yield text
            tail = filler_filter.finish()
            if tail:
                yield tail
            return

        # Vetted path: text is released only in whole sentences, each checked first. On a violation the model is
        # stopped through a PRIVATE cancel event (the caller's event also decides whether [DONE] is sent).
        for attempt in range(2):
            filler_filter = FillerStreamFilter()
            guard = SentenceClaimGuard(self._reply_veto)
            spoken: list[str] = []
            inner_cancel = asyncio.Event()

            async def _relay() -> None:  # a caller cancel (e.g. mute) must still reach the model
                if cancel_event is not None:
                    await cancel_event.wait()
                    inner_cancel.set()

            relay = asyncio.create_task(_relay())
            stream = super().execute_stream(ctx, cancel_event=inner_cancel)
            base_finished = False   # the base stream ran to its end, so it already saved the turn itself
            echoed = False
            first_sentence = True
            try:
                async for chunk in stream:
                    text = filler_filter.feed(chunk)
                    for sentence in guard.feed(text) if text else []:
                        if first_sentence:
                            first_sentence = False
                            if is_echo(sentence, ctx.user_message):
                                # The model handed the user's own message back. Nothing has been released yet,
                                # so stop it here and answer again instead of showing the echo.
                                echoed = True
                                inner_cancel.set()
                                break
                        spoken.append(sentence)
                        yield sentence
                    if echoed or guard.tripped:
                        inner_cancel.set()
                        break
                else:
                    base_finished = True
                if not echoed and not guard.tripped and not (cancel_event is not None and cancel_event.is_set()):
                    for sentence in guard.feed(filler_filter.finish()) + guard.finish():
                        if first_sentence:
                            first_sentence = False
                            if is_echo(sentence, ctx.user_message):
                                echoed = True
                                break
                        spoken.append(sentence)
                        yield sentence
            finally:
                relay.cancel()
                await stream.aclose()

            if guard.tripped:
                logger.warning("[tool-guard] streamed chat reply claimed an action with no tool call; stopped")
                if not guard.released:
                    yield UNCONFIRMED_REPLY
                if not base_finished:
                    # The base stream was cut short, so it never saved the turn: save what the user was actually given.
                    # (If it did finish it saved the full text through on_completion, which vets it.)
                    await self.on_completion(ctx, "".join(spoken).strip() or UNCONFIRMED_REPLY)
                return
            if not echoed:
                return
            logger.warning(f"[language] chat reply echoed the user: {log_text(''.join(spoken), 120)}")
            if attempt == 0:
                ctx.metadata["no_echo"] = True   # build_prompt asks for a real answer on the retry
                continue
            answer = fallback_reply(ctx.metadata.get("language"))
            yield answer
            await self.on_completion(ctx, answer)
            return

    def _render_history(self, fits: Callable[[str], bool] | None = None) -> str:
        blocks: list[str] = []
        summaries = self.conversation.get_summaries_block()
        if summaries:
            blocks.append(summaries)
        recent = self.conversation.turns[-self.max_history_turns:]
        lines = []
        for turn in recent:
            content = (turn.content or "").strip()
            if turn.role != "user":
                content = strip_filler(content)  # older saved replies may still end with the stock offer
            if not content:
                continue
            if len(content) > self._turn_chars:
                content = content[: self._turn_chars - 1] + "…"
            prefix = "User" if turn.role == "user" else "Veda"
            lines.append(f"{prefix}: {content}")
        if fits is not None:
            dropped = 0
            while lines and not fits("\n\n".join(blocks + ["RECENT CONVERSATION:\n" + "\n".join(lines)])):
                lines.pop(0)
                dropped += 1
            if dropped:
                logger.info(f"[chat] history trimmed to the {self._chat_budget}-token chat budget: dropped {dropped} oldest turn(s)")
        if lines:
            blocks.append("RECENT CONVERSATION:\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    async def on_completion(self, ctx: AgentContext, response: str) -> None:
        if ctx.metadata.get("skip_persist"):   # the orchestrator saves ONE combined turn for a mixed reply
            return
        # Saved history is replayed into later prompts; a filler line left in it gets copied into every reply.
        response = self._vet(strip_filler(response))
        if not response:
            return
        self.conversation.add_turn("user", ctx.user_message, session_id=ctx.session_id)
        self.conversation.add_turn("assistant", response, session_id=ctx.session_id)
        self._record_to_memory(action="chat", context={}, user_message=ctx.user_message)