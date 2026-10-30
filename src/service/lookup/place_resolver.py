"""PlaceResolver — "Mumbai" / "" / "here" -> a Place with coordinates.

1. Blank or "here"-like input -> the configured default place (geocoded once, cached).
2. A name -> Open-Meteo geocoding (exact, free, deterministic). The best-ranked
   match is used and the others are logged, so an ambiguous name is visible.
3. Geocoding found nothing -> ask web search for the place's coordinates and
   accept them ONLY if they parse cleanly and are in range (the search text is
   untrusted). If that also fails, raise NOT_FOUND so the user is asked for a
   nearby city instead of getting weather for a guess.
"""

from __future__ import annotations

from core.enums import ErrorMessage, ExceptionCode
from core.logging_config import logger
from domain.entities.fact_query import FactQuery
from domain.entities.place import Place
from domain.policies.coordinate_policy import parse_lat_lon
from domain.policies.place_names import normalize_place_name, to_latin
from exceptions.exception import AppException
from service.lookup.lookup_service import LookupService

_HERE_WORDS = frozenset({
    "", "here", "my city", "my location", "current location", "my place", "home", "local", "today", "now",
    "outside", "near me", "this city",
})


class PlaceResolver:
    def __init__(self, lookup: LookupService, default_place: str = "Hyderabad"):
        self._lookup = lookup
        self._default_place = default_place

    async def resolve(self, name: str | None) -> Place:
        wanted = " ".join((name or "").split())
        normalized = normalize_place_name(wanted)
        if normalized != wanted:
            logger.info(f"[place] normalized {wanted!r} -> {normalized!r}")
        wanted = normalized
        if wanted.lower() in _HERE_WORDS:
            place = await self._geocode(self._default_place, source="default")
            if place is None:
                raise self._not_found(self._default_place)
            return place
        place = await self._geocode(wanted, source="geocoder")
        if place is not None:
            return place
        latin = to_latin(wanted)
        if latin != wanted:
            logger.info(f"[place] retrying with transliterated name {wanted!r} -> {latin!r}")
            place = await self._geocode(latin, source="geocoder")
            if place is not None:
                return place
            wanted = latin
        place = await self._from_search(wanted)
        if place is not None:
            return place
        raise self._not_found(wanted)

    async def _geocode(self, name: str, *, source: str) -> Place | None:
        answer = await self._lookup.fetch(FactQuery(category="geocode", params={"name": name}))
        places = answer.data.get("places") or []
        if not places:
            return None
        best = places[0]
        if len(places) > 1:
            logger.info(
                f"[place] '{name}' is ambiguous; using {best['name']}, {best['country']} "
                f"over {[f'{p['name']}, {p['country']}' for p in places[1:3]]}"
            )
        place = Place(
            name=best["name"], lat=float(best["lat"]), lon=float(best["lon"]),
            region=best.get("region", ""), country=best.get("country", ""), source=source,
        )
        logger.info(f"[place] '{name}' -> {place.label()} ({place.lat:.3f},{place.lon:.3f}) via {source}")
        return place

    async def _from_search(self, name: str) -> Place | None:
        if self._lookup.registry.get("search") is None:
            return None
        try:
            answer = await self._lookup.fetch(
                FactQuery(category="search", params={"query": f"latitude and longitude of {name}"})
            )
        except Exception as e:  # search down / not configured: just can't use this fallback
            logger.info(f"[place] search fallback for '{name}' unavailable: {e}")
            return None
        for text in [answer.data.get("answer", "")] + [h.get("snippet", "") for h in answer.data.get("hits", [])]:
            coords = parse_lat_lon(text)
            if coords is not None:
                place = Place(name=name.title(), lat=coords[0], lon=coords[1], source="search")
                logger.warning(f"[place] '{name}' not geocoded; coordinates taken from web search: {coords}")
                return place
        return None

    @staticmethod
    def _not_found(name: str) -> AppException:
        return AppException(
            class_name="PlaceResolver", code=ExceptionCode.NOT_FOUND, error_message=ErrorMessage.GENERIC,
            detail=f"I couldn't find a place called {name}",
        )
