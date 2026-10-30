"""Verbatim multi-line input for the REPL.

prompt_toolkit's line editor has no complex-script shaping and miscalculates the width of combining marks
(Devanagari, Telugu, ...), which can drop characters while it redraws the line. Non-Latin messages go through
this reader instead: it never edits the line, so what the user pastes is exactly what reaches the server.

Pure text I/O: the stream is injected, so the contract is testable without a terminal.
"""

from __future__ import annotations

import sys
from typing import TextIO

TERMINATOR = "."


def read_message(stream: TextIO | None = None, terminator: str = TERMINATOR) -> str:
    """Read lines until a lone `terminator` (or EOF) and return them joined by newlines.

    A KeyboardInterrupt returns "" (the paste is cancelled); EOF without a terminator still returns what was
    read, so Ctrl-Z + Enter works on Windows."""
    stream = stream if stream is not None else sys.stdin
    lines: list[str] = []
    while True:
        try:
            raw = stream.readline()
        except KeyboardInterrupt:
            return ""
        if raw == "":
            break
        text = raw.rstrip("\r\n")
        if text.strip() == terminator:
            break
        lines.append(text)
    return "\n".join(lines).strip()


def ensure_utf8_streams() -> None:
    """Best effort: make piped stdin/stdout UTF-8 so non-Latin input is not decoded as the locale code page.

    The Windows console is already UTF-8 (PEP 528); this matters when input arrives through a pipe or a
    redirected file. Streams that cannot be reconfigured are left alone."""
    for stream in (sys.stdin, sys.stdout):
        try:
            encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
            if encoding != "utf8":
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
