"""PromptComposer — assembles each stage's prompt under a hard token budget.

The model is a small quantized one on CPU, so prompts are built from the
*minimum* the stage needs:

  Stage A (call):     persona_lite + only this agent's tools + their examples
                      + last 2 turns + the user message
  Stage B (narrate):  persona_lite + narrate rules + the question + the (capped)
                      tool result. No tools, no history.

Static text (persona, tool block, examples) always comes first and is
byte-identical turn to turn, so the backend's KV prefix cache is reused; only
the volatile tail (date, history, user message, tool result) changes.

If a stage is over budget the composer trims in a fixed order — oldest
history first, then the tool result / user message — and reports what it
trimmed so the trace can show it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

from domain.entities.conversation import Turn
from domain.entities.tool_manifest import ToolManifest, ToolParam
from domain.policies.language_policy import language_of
from domain.policies.tool_call_schema import CONTROL_KEYS
from domain.policies.token_budget_policy import PromptBudgets, estimate_tokens
from domain.ports.tool_manifest_store_port import PromptStorePort

_TURN_CHARS = 200        # a history turn is cut to this many characters
_USER_MSG_CHARS = 500

# Content-stage tasks: task -> (prompt file holding its instruction, label the document carries in the prompt).
# "narrate" keeps the label "TOOL RESULT" it has always had.
CONTENT_TASKS = {
    "narrate": ("narrate", "TOOL RESULT"),
    "summarise": ("content_summarise", "DOCUMENT"),
    "draft_reply": ("content_draft_reply", "DOCUMENT"),
    "extract": ("content_extract", "DOCUMENT"),
    "advise": ("content_advise", "TOOL RESULT"),   # a reasoned answer from verified results; narrate itself forbids advice
}


@dataclass
class ComposedPrompt:
    system: str
    prompt: str
    tokens: int
    sections: dict[str, int] = field(default_factory=dict)
    trimmed: list[str] = field(default_factory=list)


class PromptComposer:
    def __init__(
        self,
        prompts: PromptStorePort,
        manifests: Sequence[ToolManifest],
        budgets: PromptBudgets | None = None,
        count_tokens: Callable[[str], int] = estimate_tokens,
        now: Callable[[], datetime] = datetime.now,
        turn_chars: int = _TURN_CHARS,
        user_message_chars: int = _USER_MSG_CHARS,
    ):
        self._prompts = prompts
        self._manifests = {m.name: m for m in manifests}
        self.budgets = budgets or PromptBudgets()
        self._count = count_tokens
        self._now = now
        self._turn_chars = turn_chars
        self._user_msg_chars = user_message_chars
        self._control_static: dict[tuple[str, ...], str] = {}  # keyed by the JSON key order only, never by a tool subset

    # ---- lookups ----------------------------------------------------------

    # ---- rendering helpers -----------------------------------------------

    @staticmethod
    def _signature(m: ToolManifest) -> str:
        def one(p: ToolParam) -> str:
            kind = "|".join(p.enum) if p.enum else p.type
            return f"{p.name}{'' if p.required else '?'}: {kind}"

        return f"- {m.name}({', '.join(one(p) for p in m.params)}) - {m.description}"

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        text = " ".join((text or "").split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _static_control_text(self, key_order: tuple[str, ...]) -> str:
        """persona_lite + control_stage.md with EVERY tool signature. Byte-identical every turn: it holds no date,
        history, state or user text, and exactly one tool set exists, so there is exactly one cached prefix."""
        if key_order not in self._control_static:
            tools = "\n".join(self._signature(m) for m in self._manifests.values())
            body = self._prompts.get("control_stage").replace("<<tools>>", tools).replace(
                "<<tool_examples>>", self._flagged_examples()
            )
            body = "\n".join(self._in_key_order(line, key_order) for line in body.splitlines())
            self._control_static[key_order] = f"{self._prompts.get('persona_lite')}\n\n{body}"
        return self._control_static[key_order]

    def _flagged_examples(self) -> str:
        """The ONE example per tool that opts in with `prompt_example: true`, as control decisions. Every other
        example stays in the manifest for tests and docs only, so prompt size grows by one line pair per tool at
        most, never by every example. (A call always sets needs_live_data true, as the rules state.)"""
        lines: list[str] = []
        for m in self._manifests.values():
            for ex in m.examples:
                if ex.prompt_example:
                    decision = {
                        "needs_live_data": True,
                        "calls": list(ex.calls),
                        "language": language_of(ex.user) or "English",
                    }
                    lines.append(f"User: {ex.user}\n" + json.dumps(decision, separators=(",", ":"), ensure_ascii=False))
        return "\n".join(lines)

    @staticmethod
    def _in_key_order(line: str, key_order: tuple[str, ...]) -> str:
        """Re-serialise an example decision line so its keys follow the schema's emission order; the grammar and the
        examples must agree or the model is shown one order and forced into another."""
        if not line.startswith('{"needs_live_data"'):
            return line
        try:
            data = json.loads(line)
        except ValueError:
            return line
        if not set(data) <= set(CONTROL_KEYS):
            return line
        return json.dumps({k: data[k] for k in key_order if k in data}, separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def _exchange_turns(history: Sequence[Turn], exchanges: int) -> list[Turn]:
        """The last `exchanges` complete user->assistant pairs, oldest first. Whole pairs only: a lone user turn
        or a leading assistant turn is dropped, so RECENT never starts mid-conversation."""
        pairs: list[tuple[Turn, Turn]] = []
        turns = list(history)
        i = len(turns) - 1
        while i > 0 and len(pairs) < max(exchanges, 0):
            if turns[i].role != "user" and turns[i - 1].role == "user":
                pairs.append((turns[i - 1], turns[i]))
                i -= 2
            else:
                i -= 1
        return [t for pair in reversed(pairs) for t in pair]

    def control_stage(
        self,
        user_message: str,
        history: Sequence[Turn] = (),
        active: str = "",
        *,
        exchanges: int = 2,
        key_order: tuple[str, ...] = CONTROL_KEYS,
        show_empty: bool = False,
    ) -> ComposedPrompt:
        """The one decision prompt. SYSTEM is static; the volatile tail is `ACTIVE + RECENT + TODAY + USER`.
        Over budget, the oldest exchange goes first, then ACTIVE is kept (it is ~25 tokens and the most informative)."""
        system = self._static_control_text(key_order)
        today = f"TODAY: {self._now().strftime('%A %d %b %Y, %H:%M')}."
        user_line = f"USER: {self._clip(user_message, self._user_msg_chars)}"
        trimmed: list[str] = []
        keep = exchanges
        while True:
            lines = [
                f"{'User' if t.role == 'user' else 'Veda'}: {self._clip(t.content, self._turn_chars)}"
                for t in self._exchange_turns(history, keep)
                if self._clip(t.content, self._turn_chars)
            ]
            # show_empty: say "none" explicitly so the model SEES that there is nothing to continue, instead of
            # having to infer it from an absent line (omitting is the default and the design doc's layout).
            head = [active] if active else (["ACTIVE: none"] if show_empty else [])
            recent = [("RECENT:\n" + "\n".join(lines))] if lines else (["RECENT: none"] if show_empty else [])
            volatile = "\n".join(head + recent + [today, user_line])
            total = self._count(system) + self._count(volatile)
            if total <= self.budgets.control or keep == 0:
                break
            keep -= 1
            trimmed.append("history_exchange")
        sections = {
            "static": self._count(system),
            "active": self._count(active) if active else 0,
            "history": self._count("\n".join(lines)) if lines else 0,
            "user": self._count(user_line),
        }
        return ComposedPrompt(system=system, prompt=volatile, tokens=total, sections=sections, trimmed=trimmed)

    def narrate_stage(self, user_message: str, tool_results: Sequence[str], language: str | None = None) -> ComposedPrompt:
        """Phrase tool results for the user: the `narrate` task of the content stage."""
        result = "\n".join(r.strip() for r in tool_results if r and r.strip())
        return self.content_stage("narrate", result, user_message, language=language)

    def translate_stage(self, text: str, language: str) -> ComposedPrompt:
        """Translate ONE finished template sentence into `language`, facts untouched.

        Deliberately carries no QUESTION / user text: a weak model that sees the user's own sentence beside
        "translate into <language>" repeats the question instead of translating TEXT (the echo bug)."""
        system = self._prompts.get("translate")
        clipped = self.cap_tokens(text, self.budgets.observation_max)
        prompt = f"TARGET LANGUAGE: {language}\nTEXT:\n{clipped}\n\nTRANSLATION:"
        return ComposedPrompt(
            system=system,
            prompt=prompt,
            tokens=self._count(system) + self._count(prompt),
            sections={"static": self._count(system), "text": self._count(clipped)},
        )

    def content_stage(
        self, task: str, document: str, user_message: str, *, max_tokens: int = 0, language: str | None = None
    ) -> ComposedPrompt:
        """The data plane: reason over ONE capped document. No tool list, no history, ever.

        SYSTEM is the same for every task (persona + preamble), so all tasks share one cached prefix; the task
        instruction rides in the volatile tail. `max_tokens` is the producing tool's `max_result_tokens`, applied
        before anything else; the stage's own `observation_max` / `narrate` budgets still apply after it.
        `language` is the turn's language (from the control decode or the deterministic detector); when it is
        None the detector runs on the question itself."""
        if task not in CONTENT_TASKS:
            raise ValueError(f"unknown content task {task!r}; expected one of {sorted(CONTENT_TASKS)}")
        system = f"{self._prompts.get('persona_lite')}\n\n{self._prompts.get('content_stage')}"
        instruction = self._prompts.get(CONTENT_TASKS[task][0]).strip()
        label = CONTENT_TASKS[task][1]
        result = self.cap_tokens(document, max_tokens)
        trimmed: list[str] = []
        if result != document:
            trimmed.append("tool_result")
        cap = self.budgets.observation_max
        if self._count(result) > cap:
            result = self._truncate_tokens(result, cap)
            if "tool_result" not in trimmed:
                trimmed.append("tool_result")
        question = self._clip(user_message, self._user_msg_chars)
        today = f"TODAY: {self._now().strftime('%d %b %Y')}"
        language = language or language_of(user_message)
        # Tool results are often English (a provider's numbers/headers); naming the language keeps the
        # narration in the user's, not the tool's, language.
        language_line = f"\nLANGUAGE: the user wrote in {language}; answer in {language}." if language else ""

        def build(doc: str) -> str:
            return f"TASK: {instruction}\n{today}\nQUESTION: {question}\n{label}:\n{doc}{language_line}\n\nAnswer:"

        prompt = build(result)
        total = self._count(system) + self._count(prompt)
        if total > self.budgets.narrate:
            if "tool_result" not in trimmed:
                trimmed.append("tool_result")
            target = max(50, self._count(result) - (total - self.budgets.narrate))
            while True:  # the ellipsis suffix can cost a token, so re-check until it fits
                result = self._truncate_tokens(result, target)
                prompt = build(result)
                total = self._count(system) + self._count(prompt)
                if total <= self.budgets.narrate or target <= 50:
                    break
                target -= 8
        sections = {"static": self._count(system), "tool_result": self._count(result)}
        return ComposedPrompt(system=system, prompt=prompt, tokens=total, sections=sections, trimmed=trimmed)

    def cap_tokens(self, text: str, max_tokens: int) -> str:
        """Cut a tool result to `max_tokens` before it can reach any prompt (a tool's own `max_result_tokens`)."""
        return self._truncate_tokens(text, max_tokens) if max_tokens > 0 else text

    def _truncate_tokens(self, text: str, max_tokens: int) -> str:
        """Cut `text` so it fits `max_tokens` (binary search on characters)."""
        if self._count(text) <= max_tokens:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._count(text[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo].rstrip() + " …"
