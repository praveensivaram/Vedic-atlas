"""AuditLogger — structured JSONL audit trail.

Donor: veda/guardrails/audit.py, read in full and ported verbatim. This
is a separate, simpler audit trail from the governance layer's
AuditSinkPort (Batch 3) — that one is the tamper-evident, hash-chained
sink for governance decisions; this one is a plain per-day JSONL log of
skill/agent/permission/guardrail events. Both can coexist.
"""

import json
from datetime import date, datetime
from pathlib import Path

from domain.entities.agent_context import AgentContext
from core.constants import AUDIT_DIR
from core.logging_config import logger


class AuditLogger:
    """Writes structured audit events to daily JSONL log files."""

    def __init__(
        self,
        log_dir: str | Path | None = None,
        enabled: bool = True,
        log_inputs: bool = True,
        log_outputs: bool = True,
    ):
        self.log_dir = Path(log_dir) if log_dir else AUDIT_DIR
        self.enabled = enabled
        self.log_inputs = log_inputs
        self.log_outputs = log_outputs

    def _get_log_file(self) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self.log_dir / f"{date.today().isoformat()}.jsonl"

    def _write_event(self, event: dict) -> None:
        if not self.enabled:
            return
        try:
            with open(self._get_log_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")

    def log_skill_execution(
        self, ctx: AgentContext, skill_name: str, params: dict, success: bool,
        output: str | None = None, error: str | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now().isoformat(), "event_type": "skill_execution",
            "request_id": ctx.request_id, "agent": ctx.current_agent,
            "skill": skill_name, "success": success,
        }
        if self.log_inputs:
            event["params"] = params
        if self.log_outputs and output:
            event["output"] = output[:500]
        if error:
            event["error"] = error
        self._write_event(event)

    def log_agent_event(self, ctx: AgentContext, agent_name: str, event_type: str, detail: str = "") -> None:
        event = {
            "timestamp": datetime.now().isoformat(), "event_type": f"agent_{event_type}",
            "request_id": ctx.request_id, "agent": agent_name, "agent_chain": ctx.agent_chain,
            "user_message": ctx.user_message[:200] if self.log_inputs else "(redacted)",
        }
        if detail:
            event["detail"] = detail
        self._write_event(event)

    def log_permission_event(
        self, ctx: AgentContext, skill_name: str, level: str, approved: bool, request_id: str = "",
    ) -> None:
        event = {
            "timestamp": datetime.now().isoformat(), "event_type": "permission_check",
            "request_id": ctx.request_id, "skill": skill_name, "permission_level": level, "approved": approved,
        }
        if request_id:
            event["approval_request_id"] = request_id
        self._write_event(event)

    def log_guardrail_event(self, ctx: AgentContext, guardrail_name: str, passed: bool, reason: str = "") -> None:
        event = {
            "timestamp": datetime.now().isoformat(), "event_type": "guardrail_check",
            "request_id": ctx.request_id, "guardrail": guardrail_name, "passed": passed,
        }
        if reason:
            event["reason"] = reason
        self._write_event(event)