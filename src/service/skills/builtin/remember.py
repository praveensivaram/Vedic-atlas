"""RememberSkill — save, list and forget lasting facts about the user.

The chat prompt always carries the saved facts (see ResponderAgent), so a preference stated once
("my favourite sweet is gulab jamun") is known in every later session without depending on a
similarity search over old summaries. Facts are keyed by topic: a new value for the same topic
replaces the old one.

Name, description, argument schema and examples come from config/tools/remember.yaml; this class only
holds behaviour. Each result carries a `spoken` sentence, which is the user-facing reply (reply_mode:
template), so the confirmation is always backed by what was really written.
"""

from __future__ import annotations

from core.logging_config import logger, log_text
from domain.entities.agent_context import AgentContext
from domain.entities.skill_result import SkillResult
from domain.entities.tool_manifest import ToolManifest
from domain.policies.fact_policy import (
    MAX_FACTS, MAX_TOPIC_CHARS, MAX_VALUE_CHARS, is_sensitive, normalise_topic, split_fact,
)
from service.memory.knowledge_base import KnowledgeBase
from service.skills.manifest_skill import ManifestSkill

VALID_ACTIONS = ("save", "forget", "list")
_MAX_SPOKEN_FACTS = 8


class RememberSkill(ManifestSkill):
    """Keep lasting facts about the user: preferences, allergies, names, routines."""

    what = "save that"

    def __init__(
        self, knowledge: KnowledgeBase, manifest: ToolManifest,
        permission_level: str | None = None, enabled: bool = True,
    ):
        super().__init__(manifest, permission_level=permission_level, enabled=enabled)
        self._knowledge = knowledge

    async def run(self, ctx: AgentContext, **params) -> SkillResult:
        action = str(params.get("action") or "").lower()
        if action not in VALID_ACTIONS:
            return self._fail(f"Invalid action '{action}'. Use save, forget or list.")
        if action == "save":
            return self._save(params)
        if action == "forget":
            return self._forget(params)
        return self._list()

    # ---- actions --------------------------------------------------------

    def _save(self, params: dict) -> SkillResult:
        topic = " ".join(str(params.get("topic") or "").split())
        value = " ".join(str(params.get("value") or "").split())
        if not normalise_topic(topic) or not value:
            return self._fail("topic and value are required for save", "What should I remember?")
        if len(topic) > MAX_TOPIC_CHARS or len(value) > MAX_VALUE_CHARS:
            return self._fail("fact too long", "That's too long for me to keep. Can you say it shorter?")
        if is_sensitive(topic, value):
            return self._fail(
                "refused: sensitive data", "I won't keep passwords, PINs or card and account numbers."
            )
        is_new_topic = not any(
            (parts := split_fact(f)) and parts[0] == normalise_topic(topic) for f in self._knowledge.list_facts()
        )
        if is_new_topic and len(self._knowledge.list_facts()) >= MAX_FACTS:
            return self._fail(
                f"memory full ({MAX_FACTS} facts)",
                "I'm already keeping as much as I can. Tell me something to forget first.",
            )
        stored, previous = self._knowledge.remember(topic, value)
        name = normalise_topic(topic)
        logger.info(f"[remember] saved {log_text(stored)}" + (f" (replaced {log_text(previous)})" if previous else ""))
        if previous is None:
            return self._ok(f"Saved: {stored}", f"Got it, I'll remember that your {name} is {value}.")
        if previous.lower() == value.lower():
            return self._ok(f"Already saved: {stored}", f"I already have your {name} as {value}.")
        return self._ok(
            f"Updated: {stored} (was {previous})", f"Updated. Your {name} is now {value}, not {previous}."
        )

    def _forget(self, params: dict) -> SkillResult:
        topic = " ".join(str(params.get("topic") or "").split())
        if not normalise_topic(topic):
            return self._fail("topic is required for forget", "What should I forget?")
        removed = self._knowledge.forget(topic)
        if removed is None:
            return self._ok(
                f"Nothing saved about '{topic}'. Nothing was changed.",
                f"I don't have anything saved about {normalise_topic(topic)}.",
            )
        logger.info(f"[remember] forgot {log_text(removed)}")
        return self._ok(f"Forgot: {removed}", f"Okay, I've forgotten your {normalise_topic(topic)}.")

    def _list(self) -> SkillResult:
        facts = self._knowledge.list_facts()
        if not facts:
            return self._ok("No facts are saved.", "I haven't saved anything about you yet.")
        shown = facts[:_MAX_SPOKEN_FACTS]
        more = len(facts) - len(shown)
        spoken = "Here's what I remember: " + "; ".join(shown) + (f"; and {more} more." if more else ".")
        return self._ok(f"{len(facts)} fact(s):\n" + "\n".join(f"- {f}" for f in facts), spoken)
