"""ArgumentPolicy ? repair declared Latin arguments after the control decode.

A tool parameter marked `format: latin` (place names, lookup queries) must reach the tool in Latin script: the
geocoder/API expects that, and a small model otherwise copies the user's script. The prompt asks for Latin;
this is the deterministic half that repairs a non-compliant decode before the call executes. Pure.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from domain.entities.control_decision import ControlDecision
from domain.entities.tool_manifest import ToolManifest
from domain.policies.place_names import to_latin


def normalize_latin_arguments(
    decision: ControlDecision, manifests: Mapping[str, ToolManifest]
) -> tuple[ControlDecision, list[str]]:
    """Return the decision with Latin-format arguments normalised, plus a note per change."""
    notes: list[str] = []
    calls: list[dict] = []
    for call in decision.calls:
        if not isinstance(call, dict):
            calls.append(call)
            continue
        args = dict(call.get("args") or {})
        manifest = manifests.get(call.get("tool"))
        if manifest is not None:
            for param in manifest.params:
                if param.format != "latin":
                    continue
                value = args.get(param.name)
                if not isinstance(value, str):
                    continue
                latin = to_latin(value)
                if latin and latin != value:
                    args[param.name] = latin
                    notes.append(f"{param.name}: {value!r} -> {latin!r}")
        calls.append({**call, "args": args})
    if not notes:
        return decision, notes
    return replace(decision, calls=tuple(calls)), notes
