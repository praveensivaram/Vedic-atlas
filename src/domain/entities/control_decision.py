"""ControlDecision — the single decision made for a turn (docs/10).

Produced by a decoder (the constrained control decode, or any later protocol) and consumed by
`dispatch_policy.resolve`. Pure data. `needs_live_data is None` means the protocol has no such flag.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlDecision:
    calls: tuple[dict, ...] = ()          # each {"tool": name, "args": {...}}
    needs_live_data: bool | None = None
    clarification: str | None = None
    language: str | None = None           # the language the user wrote in, as the model named it ("German"), or None
    advise: bool = False                  # the tool's sentence alone would not answer the question (needs a reasoned reply)
    also_chat: bool = False               # part of the message needs no tool at all
    valid: bool = True                    # False = the output could not be parsed or the model call failed
    note: str = ""                        # why it is invalid, for the trace
