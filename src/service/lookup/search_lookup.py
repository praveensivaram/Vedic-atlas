"""SearchLookup — a real web search for current facts and news.

Returns the provider's own synthesised answer plus a few trimmed source
snippets (title, domain, date). No page fetching: the result stays small
enough for a 4B model to read, and the observation always carries the
source domains so the reply can say where it came from.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Callable

from core.enums import ErrorMessage, ExceptionCode
from core.logging_config import logger, log_text
from domain.entities.fact_query import FactQuery
from domain.entities.tool_observation import ToolObservation
from domain.policies.query_policy import resolve_relative_dates, results_look_relevant
from exceptions.exception import AppException
from service.lookup.lookup_service import LookupService

_MAX_QUERY = 200
_SPOKEN_CHARS = 320
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# A full stop after an initial ("N. R. Murthy") or a title ("Dr.", "St.") does not end a sentence.
_ABBREVIATION_END = re.compile(r"(?:^|\s)(?:[A-Z]|Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|No|Inc|Ltd|Co|Rs)\.$")


def _sentences(text: str) -> list[str]:
    merged: list[str] = []
    for part in _SENTENCE_RE.split(" ".join((text or "").split())):
        if merged and _ABBREVIATION_END.search(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _first_sentences(text: str, limit: int = _SPOKEN_CHARS) -> str:
    """Whole sentences up to `limit` characters (at least the first, clipped if it alone is too long)."""
    out = ""
    for sentence in _sentences(text):
        if out and len(out) + 1 + len(sentence) > limit:
            break
        out = f"{out} {sentence}".strip()
    if len(out) <= limit:
        return out
    cut = out[:limit]
    boundary = max(cut.rfind(", "), cut.rfind("; "))  # a list item boundary, so we never stop mid-title
    if boundary > limit // 2:
        return cut[:boundary].rstrip(",; ") + ", and more."
    return cut[: cut.rfind(" ")].rstrip(",; ") + "..."


_GENERIC_SECOND_LEVEL = frozenset({"co", "com", "org", "gov", "ac", "net", "edu"})


def _outlet(domain: str) -> str:
    """The site's own name, readable aloud: business-standard.com -> "business standard",
    en.wikipedia.org -> "wikipedia", isro.gov.in -> "isro", bbc.co.uk -> "bbc"."""
    parts = [p for p in (domain or "").split(".") if p]
    if not parts:
        return "the web"
    if len(parts) >= 3 and parts[-2] in _GENERIC_SECOND_LEVEL:
        name = parts[-3]
    elif len(parts) >= 2:
        name = parts[-2]
    else:
        name = parts[0]
    return name.replace("-", " ")


def _when(iso: str) -> str:
    try:
        return date.fromisoformat(iso).strftime("%d %b").lstrip("0")
    except ValueError:
        return ""


def _headlines(hits: list[dict]) -> str:
    """The top two hits (the provider ranks by relevance within the last week for news), spoken with outlet and date."""
    parts = []
    for h in hits[:2]:
        when = _when(h.get("published", ""))
        where = _outlet(h.get("domain", ""))
        parts.append(f"{h['title']} ({where}{', ' + when if when else ''})")
    return "Latest from the web: " + "; ".join(parts) + "."


class SearchLookup:
    def __init__(self, lookup: LookupService, today: Callable[[], date] = date.today):
        self._lookup = lookup
        self._today = today

    async def search(self, query: str, topic: str = "general") -> ToolObservation:
        q = " ".join(str(query or "").split())[:_MAX_QUERY]
        if not q:
            raise AppException(
                class_name="SearchLookup", code=ExceptionCode.VALIDATION_ERROR, error_message=ErrorMessage.GENERIC,
                detail="What should I search for?",
            )
        # "this month" means nothing to a search engine (and the model often leaves it as-is),
        # so relative dates become concrete ones before anything leaves the device.
        resolved = resolve_relative_dates(q, self._today())
        if resolved != q:
            logger.info(f"[search] query dates resolved: {log_text(q)} -> {log_text(resolved)}")
            q = resolved

        answer = await self._lookup.fetch(FactQuery(category="search", params={"query": q, "topic": topic}))
        if not self._on_topic(q, answer.data):
            # The provider's answer is machine-written and was sometimes about something else entirely
            # (Hollywood re-releases for a Tollywood question). Retry once with the deeper search.
            logger.warning(f"[search] results for {log_text(q)} look off-topic; retrying with advanced depth")
            answer = await self._lookup.fetch(
                FactQuery(category="search", params={"query": q, "topic": topic, "depth": "advanced"})
            )
            if not self._on_topic(q, answer.data):
                logger.warning(f"[search] results for {log_text(q)} still look off-topic; refusing to report them")
                raise AppException(
                    class_name="SearchLookup", code=ExceptionCode.NOT_FOUND, error_message=ErrorMessage.GENERIC,
                    detail=f"I couldn't find reliable results for '{q}'",
                )
        d = answer.data
        hits = d.get("hits") or []
        if not d.get("answer") and not hits:
            raise AppException(
                class_name="SearchLookup", code=ExceptionCode.NOT_FOUND, error_message=ErrorMessage.GENERIC,
                detail=f"the search for '{q}' returned nothing",
            )
        # The provider's synthesised answer can lag the news, so for news questions speak
        # the newest dated headlines instead. Both forms are finished text built from the
        # provider's data: no narrate-stage model call, nothing the model can embellish.
        final = True
        if d.get("topic") == "news" and hits:
            spoken = _headlines(hits)
        elif d.get("answer"):
            spoken = _first_sentences(d["answer"])
            if hits and hits[0].get("domain"):
                spoken += f" Source: {_outlet(hits[0]['domain'])}."
        else:
            spoken, final = f"{hits[0]['title']}. {hits[0]['snippet']}", False  # let the narrate stage phrase raw snippets
        text = f"WEB SEARCH for '{q}' (source tavily{', cached' if answer.cached else ''}):\n{answer.text}"
        d = {**d, "hits": [{k: v for k, v in h.items() if k != "match_text"} for h in hits]}
        return ToolObservation(
            text=text, spoken=spoken, source="tavily", cached=answer.cached, final=final,
            as_of=answer.fetched_at.isoformat(timespec="minutes") if answer.fetched_at else "", data=d,
        )


    @staticmethod
    def _on_topic(query: str, data: dict) -> bool:
        """Do the result titles/snippets mention what the query is about? (No hits at all: nothing to judge.)"""
        hits = data.get("hits") or []
        if not hits:
            return True
        return results_look_relevant(query, [h.get("match_text") or h.get("snippet", "") for h in hits])
