"""Streaming request schema for POST /api/stream.

`image_paths` was dropped — vision is out of scope for this build.
"""
import re

from pydantic import BaseModel, Field, field_validator

unique_escape = re.compile(r"\\u([0-9a-fA-F]{4})")

class StreamRequest(BaseModel):
    message: str
    system_context: str = ""
    from_voice: bool = False
    # Optional caller-supplied session. When omitted the server derives one from the 30-minute activity gap.
    session_id: str | None = Field(default=None, min_length=1, max_length=32)

    @field_validator("message", mode="before")
    @classmethod
    def decode_message(cls, value: object):
        """Accept literal \\uXXXX escapes from scripted/JSON clients without rejecting real Unicode text."""
        if not isinstance(value, str) or "\\u" not in value:
            return value
        decoded = unique_escape.sub(lambda match: chr(int(match.group(1), 16)), value)
        return decoded.encode("utf-16", "surrogatepass").decode("utf-16")
