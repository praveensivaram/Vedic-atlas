"""Structured logging with a request-scoped identifier.

Donor: veda/log/__init__.py.

Fix applied (identified during migration analysis): the donor stored the
request identifier as a mutable attribute (`self.identifier`) on the single
process-wide logger instance. Under asyncio, concurrent requests race on
that attribute — request A can log with request B's identifier. Replaced
with a contextvars.ContextVar, which is the correct async-safe primitive:
each task gets its own value, no locking needed.

Also wires `log_level` from config instead of hardcoding DEBUG (previously
declared in VedaConfig.log_level but never read).
"""

import logging
import sys
from contextvars import ContextVar, Token

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_voice_turn_id: ContextVar[str] = ContextVar("voice_turn_id", default="-")


class _ReminderAccessLogFilter(logging.Filter):
    """Suppress successful reminder-poll access logs while keeping request errors visible."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        request_line = message.partition('"')[2].partition('"')[0].split()
        status = message.rpartition('" ')[2]
        is_reminder_poll = (
            len(request_line) >= 2
            and request_line[0] == "GET"
            and request_line[1].split("?", 1)[0] == "/api/reminders/recent"
        )
        return not (is_reminder_poll and status.startswith("200 "))


def set_request_id(identifier: str) -> None:
    """Bind a request identifier to the current async task / thread context."""
    _request_id.set(identifier)


def set_voice_turn_id(identifier: str) -> Token[str]:
    """Bind a voice turn identifier to the current async task / thread context."""
    return _voice_turn_id.set(identifier)


def get_voice_turn_id() -> str:
    """Return the voice turn identifier active in this execution context."""
    return _voice_turn_id.get()


def reset_voice_turn_id(token: Token[str]) -> None:
    """Restore the voice turn identifier that was active before this scope."""
    _voice_turn_id.reset(token)


class _RequestIdFilter(logging.Filter):
    """Injects the current context's request id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.identifier = _request_id.get()
        record.voice_turn_id = _voice_turn_id.get()
        return True



def log_text(value: object, limit: int | None = None) -> str:
    """Readable single-line text for logs.

    `repr()` renders Devanagari/Telugu/Japanese as \\uXXXX escapes, which makes voice and user-message logs
    useless. This keeps the characters as they are and escapes only the control characters that would break a
    log line (newline, carriage return, tab, backslash)."""
    text = "" if value is None else str(value)
    if limit is not None:
        text = text[:limit]
    return (
        text.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def _force_utf8_streams() -> None:
    """Write Devanagari/Telugu to redirected logs instead of the Windows locale code page."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

def configure_logging(level: str = "INFO") -> logging.Logger:
    """Build and return the app logger. Call once at process startup."""
    _force_utf8_streams()
    logger = logging.getLogger("veda")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not any(isinstance(item, _RequestIdFilter) for item in logger.filters):
        logger.addFilter(_RequestIdFilter())

    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, _ReminderAccessLogFilter) for item in access_logger.filters):
        access_logger.addFilter(_ReminderAccessLogFilter())

    if not any(getattr(handler, "_veda_configured", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)-7s][%(identifier)s][voice_turn=%(voice_turn_id)s][%(funcName)s]"
            "[%(module)s:%(filename)s].(%(lineno)d)] : %(message)s"
        )
        handler.setFormatter(formatter)
        setattr(handler, "_veda_configured", True)
        logger.addHandler(handler)
    return logger


# Default instance for modules that just need `from core.logging_config import logger`.
# bootstrap/lifespan.py should call configure_logging(cfg.log_level) once at startup to
# apply the configured level; this default keeps imports working before that.
logger = configure_logging()