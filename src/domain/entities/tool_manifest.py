"""ToolManifest — the single declarative source of truth for one model-callable tool.

Everything that used to be scattered (description in the skill class, regexes
in the guard, examples in the prompt, hosts in privacy.yaml) is declared once
here, and the rest of the harness is *generated* from it: the prompt block,
the JSON-schema the model is constrained to, the router's trigger patterns,
the guard's claim/intent rules, and the reply mode.

Pure data — no I/O, no regex compilation. Loaded from config/tools/*.yaml by
a ToolManifestStorePort adapter.
"""

from dataclasses import dataclass, field
from typing import Any

REPLY_TEMPLATE = "template"  # reply is built from the tool's own `spoken` text — no 2nd model call
REPLY_LLM = "llm"            # a short narrate-stage model call phrases the result

RETURNS_VALUE = "value"        # a small fact; may appear in a control prompt
RETURNS_DIGEST = "digest"      # a short list of references/snippets; may appear in a control prompt
RETURNS_DOCUMENT = "document"  # large content; may reach only the data plane, never a control prompt


@dataclass(frozen=True)
class ToolParam:
    """One flat argument of a tool."""

    name: str
    type: str  # "string" | "integer" | "number" | "boolean"
    required: bool = False
    enum: tuple[str, ...] = ()
    description: str = ""
    format: str | None = None   # "latin": a proper name (place/query) the model must transliterate to Latin


@dataclass(frozen=True)
class ToolExample:
    """A worked example: user phrase -> the exact `calls` the model should emit."""

    user: str
    calls: tuple[dict[str, Any], ...]  # each {"tool": name, "args": {...}}
    prompt_example: bool = False       # the one example this tool may contribute to a prompt (all others are docs/tests)


@dataclass(frozen=True)
class ToolManifest:
    name: str                      # tool name the model emits, e.g. "get_weather"
    agent: str                     # owning specialist agent, e.g. "lookup"
    description: str               # one short line — it is paid for in every prompt
    params: tuple[ToolParam, ...]
    examples: tuple[ToolExample, ...] = ()
    triggers: tuple[str, ...] = ()  # regexes: user text that should route to this tool's agent
    required_when: tuple[str, ...] = ()  # regexes: user text for which a call is MANDATORY (facts must come from the tool)
    claims: tuple[str, ...] = ()    # regexes: reply text that asserts this tool's action happened
    reply_mode: str = REPLY_LLM
    cache_ttl_sec: int = 0
    permission_level: str = "notify"
    hosts: tuple[str, ...] = field(default_factory=tuple)  # egress hosts the tool may touch
    requires_online: bool = False
    returns: str = RETURNS_VALUE   # what kind of result the tool yields (see RETURNS_*)
    max_result_tokens: int = 0     # hard cap on the result before it reaches any prompt; 0 = use the stage budget
    destructive: bool = False      # the whole tool is destructive: never inherits session slots, never writes session state
    # Per-argument destructiveness for tools whose enum mixes safe and destructive actions,
    # e.g. (("action", ("delete",)),). A tuple of pairs keeps the dataclass hashable.
    destructive_when: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # The argument(s) that say WHAT a destructive call acts on (tasks: title). The user must have named it in their
    # own message; otherwise the call is not run and the turn asks. Empty = the tool has no such rule.
    target_params: tuple[str, ...] = ()
    # The result is the user's private content (mail, calendar): it is spoken to them but never written to the trace,
    # the audit log, the tool-call log or the API's skill_calls.
    private: bool = False
    # The model may ask for a reasoned (advised) reply on this tool's result instead of its own sentence. Default off:
    # configuration declares what is permitted, the model only chooses within it.
    allow_advice: bool = False
