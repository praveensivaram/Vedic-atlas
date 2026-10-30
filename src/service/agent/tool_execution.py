"""Tool-call execution shared by the orchestrator: run one call through SkillRunner, and phrase tool results.

    execute_call            one call through SkillRunner (permission / rate-limit / validators / audit hooks all apply)
    narrate_with_grounding  the content stage's `narrate` task: one short completion over capped tool results, then a
                            check that every number in it came from the results, the question or today's date
    ExecutedCall / TurnOutcome  what a turn did, for the reply, the session state and the trace
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from core.logging_config import logger, log_text
from domain.entities.agent_context import AgentContext
from domain.ports.inference_port import InferencePort
from service.prompting.prompt_composer import PromptComposer
from service.skills.skill_runner import SkillRunner


@dataclass
class ExecutedCall:
    tool: str
    args: dict[str, Any]
    ok: bool
    observation: str   # compact text the narrate stage may use
    spoken: str | None  # the tool's own user-facing sentence (template mode)
    final: bool         # the tool says `spoken` is ready as-is (skip narration)
    error: str | None
    ms: int


@dataclass
class TurnOutcome:
    reply: str | None  # None => no tool needed, hand to plain chat
    calls: list[ExecutedCall] = field(default_factory=list)
    narrated: bool = False
    prompt_tokens: dict[str, int] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    reply_mode: str = ""   # "template" | "advise": what shaped the reply (trace only)


async def execute_call(
    skill_runner: SkillRunner, ctx: AgentContext, call: dict[str, Any], *, private: bool = False,
) -> ExecutedCall:
    """Run one call through SkillRunner (permission / rate-limit / validators / audit hooks all apply) and log it.
    `private` (the tool's manifest says so) keeps the result text out of the log line."""
    tool, args = call["tool"], call["args"]
    started = time.perf_counter()
    logger.info(f"[tool-call] start tool={tool} args={json.dumps(args, default=str)}")
    try:
        result = await skill_runner.execute_skill(ctx, tool, **args)
        ok = bool(result.success)
        observation = str(result.output) if ok and result.output is not None else ""
        error = None if ok else (result.error or "blocked")
        spoken = (result.metadata or {}).get("spoken")
        final = bool((result.metadata or {}).get("final"))
    except Exception as e:
        ok, observation, error, spoken, final = False, "", str(e), None, False
    ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        f"[tool-call] agent={ctx.current_agent} tool={tool} args={json.dumps(args, default=str)} "
        f"ok={ok} ms={ms} result={('<private>' if private and ok else (observation or error or '')[:200])!r}"
    )
    return ExecutedCall(tool, args, ok, observation, spoken, final, error, ms)


async def narrate_with_grounding(
    *,
    client: InferencePort,
    composer: PromptComposer,
    grounding_check: Callable[[str, Sequence[str], str], bool],
    ctx: AgentContext,
    results: Sequence[str],
    model: str | None,
    num_predict: int,
    temperature: float,
    outcome: "TurnOutcome",
    task: str = "narrate",
    header: str = "",
    private: bool = False,
) -> str | None:
    """The narrate/content stage: one short completion over capped tool results, checked for ungrounded numbers.
    An EMPTY completion is retried once (transient); an UNGROUNDED one is not: the retry would use the same prompt at
    near-zero temperature with no hint of what was wrong, and the caller's deterministic reply is already correct, so
    a retry costs seconds for low odds and would break the two-model-calls-per-turn ceiling.
    Returns the text, or None; the caller then falls back to the deterministic reply.
    `task` picks the content task ("narrate" or "advise"); `header` is a line placed above the results in the prompt."""
    language = ctx.metadata.get("language")
    if task == "narrate" and not header:
        stage_b = composer.narrate_stage(ctx.user_message, results, language=language)
    else:
        body = "\n".join(r.strip() for r in results if r and r.strip())
        document = f"{header}\n{body}" if header else body
        stage_b = composer.content_stage(task, document, ctx.user_message, language=language)
    outcome.prompt_tokens[task] = stage_b.tokens
    shown = (lambda t: "<private>" if private else log_text(t, 300))
    logger.info(f"[{task}] decoding over {len(results)} result(s), prompt ~{stage_b.tokens} tokens...")
    for attempt in range(2):
        try:
            text = (await client.complete(
                prompt=stage_b.prompt, system=stage_b.system, model=model,
                temperature=temperature, num_predict=num_predict,
            )).strip()
        except Exception as e:
            logger.error(f"[tool-turn] narrate stage inference failed: {e}")
            return None
        if not text:
            logger.info(f"[{task}] empty output (attempt {attempt + 1})")
            continue
        grounded = grounding_check(text, results, ctx.user_message)
        logger.info(f"[{task}] output={shown(text)} grounded={grounded}")
        if grounded:
            outcome.narrated = True
            return text
        logger.warning(f"[tool-guard] ungrounded narration (attempt {attempt + 1}): {shown(text)}")
        outcome.notes.append("ungrounded narration rejected")
        return None   # not retried: see the docstring
    return None
