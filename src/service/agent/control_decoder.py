"""ControlDecoder — the ONE model call that decides what a turn does (docs/10 §4.1).

Prompt: persona_lite + every tool + rules (static, one cached prefix) / ACTIVE + RECENT + TODAY + USER (volatile).
Output: grammar-constrained `{"needs_live_data", "calls"}` + optional `"clarification"`, so the model cannot emit malformed
JSON, an unknown tool or an invalid argument. The decoder never raises: a failed call or unusable output becomes
an INVALID decision, which `dispatch_policy.resolve` turns into a fail-closed reply.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

from core.logging_config import logger
from domain.entities.control_decision import ControlDecision
from domain.entities.conversation import Turn
from domain.entities.tool_manifest import ToolManifest
from domain.policies.tool_call_schema import CONTROL_KEYS, build_control_schema
from domain.ports.inference_port import InferencePort
from service.prompting.prompt_composer import PromptComposer


@dataclass
class DecodeResult:
    decision: ControlDecision
    prompt_tokens: int = 0
    ms: int = 0
    trimmed: tuple[str, ...] = ()
    raw: str = ""                       # the model's raw output, for the control log


def parse_decision(raw: str) -> ControlDecision:
    """Strict parse: anything off-shape is INVALID rather than guessed at (a truncated reply is the usual cause)."""
    try:
        data = json.loads(raw)
        calls, live, clar = data["calls"], data["needs_live_data"], data.get("clarification")
        language = data.get("language")
        advise, also_chat = data.get("advise", False), data.get("also_chat", False)
    except (TypeError, ValueError, KeyError):
        return ControlDecision(valid=False, note=f"unparseable: {str(raw)[:120]!r}")
    if not isinstance(calls, list) or not isinstance(live, bool) or not (clar is None or isinstance(clar, str)):
        return ControlDecision(valid=False, note=f"wrong types: {str(raw)[:120]!r}")
    if not isinstance(advise, bool) or not isinstance(also_chat, bool):   # strict: never coerce "true" / 1
        return ControlDecision(valid=False, note=f"wrong types: {str(raw)[:120]!r}")
    if not (language is None or isinstance(language, str)):
        return ControlDecision(valid=False, note=f"wrong types: {str(raw)[:120]!r}")
    return ControlDecision(
        calls=tuple(calls), needs_live_data=live, clarification=clar,
        language=language, advise=advise, also_chat=also_chat,
    )


class ControlDecoder:
    def __init__(
        self,
        *,
        client: InferencePort,
        composer: PromptComposer,
        manifests: Mapping[str, ToolManifest],
        model: str | None = None,
        num_predict: int = 200,
        temperature: float = 0.0,
        show_empty: bool = True,
        exchanges: int = 2,
        key_order: tuple[str, ...] = CONTROL_KEYS,
    ):
        self._client = client
        self._composer = composer
        self._model = model
        self._num_predict = num_predict
        self._temperature = temperature
        self._exchanges = exchanges
        self._key_order = key_order
        self._show_empty = show_empty
        self._schema = build_control_schema(list(manifests.values()), key_order=key_order)

    async def decide(self, user_message: str, history: Sequence[Turn] = (), active: str = "") -> DecodeResult:
        prompt = self._composer.control_stage(
            user_message, history, active, exchanges=self._exchanges, key_order=self._key_order,
            show_empty=self._show_empty,
        )
        started = time.perf_counter()
        raw = ""
        logger.info(f"[control] deciding (prompt ~{prompt.tokens} tokens, history trimmed={list(prompt.trimmed)})...")
        try:
            raw = await self._client.complete(
                prompt=prompt.prompt, system=prompt.system, model=self._model, json_schema=self._schema,
                temperature=self._temperature, num_predict=self._num_predict,
            )
        except Exception as e:  # inference down, timed out, ... -> fail closed, never an exception into the turn
            logger.error(f"[control] decode failed: {e}")
            decision = ControlDecision(valid=False, note=f"inference failed: {e}")
        else:
            decision = parse_decision(raw)
            if not decision.valid:
                logger.warning(f"[control] {decision.note}")
        return DecodeResult(
            decision=decision, prompt_tokens=prompt.tokens, ms=int((time.perf_counter() - started) * 1000),
            trimmed=tuple(prompt.trimmed), raw=raw if isinstance(raw, str) else "",
        )
