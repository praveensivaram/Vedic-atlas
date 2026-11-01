"""Voice API — status of the always-on listen/think/speak loop.

Donor: veda/routes/voice_config.py, now reading `service/voice/voice_session.py`.

Read-only. Muting is a privacy control and lives on /api/privacy/* behind the
capture gate; the voice session itself starts and stops with the server.
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class VoiceLanguageRequest(BaseModel):
    language: str


@router.get("/voice/status")
async def voice_status(request: Request) -> dict:
    s = getattr(request.app.state, "voice_session", None)
    if s is None:
        return {"available": False, "running": False, "reason": "voice session not built"}
    return {"available": True, **s.status()}


@router.post("/voice/language")
async def set_voice_language(body: VoiceLanguageRequest, request: Request) -> dict:
    """Pin the STT language ("en", "hi", ...) or "auto" for detection + sticky, without a restart."""
    session = getattr(request.app.state, "voice_session", None)
    if session is None:
        raise HTTPException(status_code=503, detail="voice session not built")
    try:
        return {"language": session.set_stt_language(body.language)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
