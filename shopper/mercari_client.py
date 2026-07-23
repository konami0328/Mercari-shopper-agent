"""Async client for Mercari Japan search and listing details.

The client wraps `mercapi` (which handles Mercari's request signing) and
adds four things the agent needs:

  1. Normalisation into the local `Item` / `ItemDetails` types.
  2. A record/replay cassette layer so evaluation runs are reproducible.
  3. Client-side rate limiting, to stay a well-behaved consumer.
  4. A trace hook, so latency and failure counts are available from day one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
from mercapi import Mercapi
from mercapi.requests.search import SearchRequestData

from shopper import models
from shopper.models import (
    FixtureMissingError,
    InvalidParameterError,
    Item,
    ItemDetails,
    ItemNotFoundError,
    NetworkError,
    RateLimitError,
    SellerRating,
)

Mode = Literal["live", "record", "replay"]
TraceHook = Callable[[dict[str, Any]], None]

# Mercari returns a fixed page of ~120 results; there is no page-size
# parameter, so anything smaller is a client-side slice.
_MAX_PAGE_SIZE = 120

# Maps our stable sort names onto mercapi's two separate enums.
_SORT_OPTIONS: dict[str, tuple[SearchRequestData.SortBy,
                               SearchRequestData.SortOrder]] = {
    "score": (SearchRequestData.SortBy.SORT_SCORE,
              SearchRequestData.SortOrder.ORDER_DESC),
    "price_asc": (SearchRequestData.SortBy.SORT_PRICE,
                  SearchRequestData.SortOrder.ORDER_ASC),
    "price_desc": (SearchRequestData.SortBy.SORT_PRICE,
                   SearchRequestData.SortOrder.ORDER_DESC),
    "newest": (SearchRequestData.SortBy.SORT_CREATED_TIME,
               SearchRequestData.SortOrder.ORDER_DESC),
}


def _slugify(text: str, max_length: int = 24) -> str:
    """Builds a short filesystem-safe hint for cassette filenames."""
    cleaned = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")
    return cleaned[:max_length] or "empty"


def _cassette_key(operation: str, params: dict[str, Any]) -> str:
    """Hashes a call into a stable cassette key.

    Parameters are canonicalised (None dropped, keys sorted) so that
    equivalent calls written slightly differently still collide onto the
    same key. This absorbs some noise, but not all: a model that emits
    "vintage denim" one run and "vintage  denim" the next will miss. Run
    evaluations at temperature 0 to keep that drift down.
    """
    canonical = {k: v for k, v in sorted(params.items()) if v is not None}
    payload = json.dumps(
        {"op": operation, "params": canonical},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalise_status(raw_status: str | None) -> str:
    """Harmonises the two status spellings Mercari uses.

    Search results say `ITEM_STATUS_ON_SALE`; the detail endpoint says
    `on_sale`. Downstream code should only ever see the latter.
    """
    if not raw_status:
        return "unknown"
    return raw_status.replace("ITEM_STATUS_", "").lower()


class MercariClient:
    """Fetches listings from Mercari Japan.

    Modes:
        live: Always call the network, never touch cassettes.
        record: Call the network and persist each response as a cassette.
        replay: Serve from cassettes only.

    Attributes:
        mode: One of "live", "record", "replay".
        strict_replay: When True, a replay miss raises instead of falling
            back to the network. Keep this on for evaluation runs.
    """

    def __init__(
        self,
        mode: Mode = "live",
        fixture_dir: Path | str = "fixtures/cassettes",
        rate_limit_seconds: float = 1.0,
        strict_replay: bool = True,
        trace_hook: TraceHook | None = None,
    ) -> None:
        """Initialises the client.

        Args:
            mode: Whether to hit the network, record it, or replay it.
            fixture_dir: Directory holding recorded cassettes.
            rate_limit_seconds: Minimum spacing between network calls.
            strict_replay: Fail loudly on a replay miss.
            trace_hook: Called once per operation with a metrics dict.
        """
        self.mode = mode
        self.fixture_dir = Path(fixture_dir)
        self.rate_limit_seconds = rate_limit_seconds
        self.strict_replay = strict_replay
        self._trace_hook = trace_hook
        self._api = Mercapi()
        self._last_request_at = 0.0
        self._throttle_lock = asyncio.Lock()

        if mode in ("record", "replay"):
            self.fixture_dir.mkdir(parents=True, exist_ok=True)

    async def search(
        self,
        keyword: str,
        price_min: int | None = None,
        price_max: int | None = None,
        condition_at_least: str | None = None,
        sort: str = "score",
        limit: int = 60,
    ) -> list[Item]:
        """Searches on-sale listings.

        Args:
            keyword: Search terms. Japanese works best, but the Mercari
                index tolerates English reasonably well.
            price_min: Inclusive lower bound in JPY.
            price_max: Inclusive upper bound in JPY.
            condition_at_least: Worst acceptable condition slug, e.g.
                "no_noticeable_damage". See `models.CONDITION_SLUGS`.
            sort: One of "score", "price_asc", "price_desc", "newest".
            limit: Maximum listings to return (capped at one page).

        Returns:
            Matching listings, possibly empty. An empty result is a normal
            outcome, not an error: the agent should be free to see it and
            decide to broaden the query itself.

        Raises:
            InvalidParameterError: On an unknown sort or condition slug.
            NetworkError, RateLimitError: On transport failures.
            FixtureMissingError: In strict replay with no matching cassette.
        """
        if sort not in _SORT_OPTIONS:
            raise InvalidParameterError(
                f"Unknown sort {sort!r}. "
                f"Expected one of: {', '.join(_SORT_OPTIONS)}"
            )

        condition_ids = (
            models.condition_ids_at_least(condition_at_least)
            if condition_at_least
            else []
        )
        limit = max(1, min(limit, _MAX_PAGE_SIZE))

        # `limit` is a client-side slice, not part of the request Mercari
        # sees, so it is excluded from the key. Recording once therefore
        # serves every limit, which matters when an evaluation sweeps the
        # context-size knob.
        request_params = {
            "keyword": keyword,
            "price_min": price_min,
            "price_max": price_max,
            "condition_at_least": condition_at_least,
            "sort": sort,
        }
        key = _cassette_key("search", request_params)
        path = self.fixture_dir / f"search_{_slugify(keyword)}_{key}.json"
        traced_params = {**request_params, "limit": limit}

        started = time.perf_counter()
        cache_state = "live"
        try:
            cached = self._read_cassette(path)
            if cached is not None:
                cache_state = "hit"
                items = [Item.from_dict(row) for row in cached]
            else:
                items = await self._search_live(
                    keyword, price_min, price_max, condition_ids, sort
                )
                if self.mode == "record":
                    cache_state = "recorded"
                    self._write_cassette(path, [i.to_dict() for i in items])
        except Exception as error:
            self._trace(
                "search", traced_params, started,
                cache=self._failed_cache_state(path), error=error,
            )
            raise
        items = items[:limit]
        self._trace(
            "search", traced_params, started,
            count=len(items), cache=cache_state,
        )
        return items

    async def get_item_details(self, item_id: str) -> ItemDetails:
        """Fetches the full listing, including description and seller stats.

        Args:
            item_id: A Mercari listing id, e.g. "m42750316499".

        Returns:
            The enriched listing.

        Raises:
            ItemNotFoundError: If the listing does not exist.
            NetworkError, RateLimitError: On transport failures.
            FixtureMissingError: In strict replay with no matching cassette.
        """
        params = {"item_id": item_id}
        key = _cassette_key("details", params)
        path = self.fixture_dir / f"details_{_slugify(item_id)}_{key}.json"

        started = time.perf_counter()
        cache_state = "live"
        try:
            cached = self._read_cassette(path)
            if cached is not None:
                cache_state = "hit"
                details = ItemDetails.from_dict(cached)
            else:
                details = await self._details_live(item_id)
                if self.mode == "record":
                    cache_state = "recorded"
                    self._write_cassette(path, details.to_dict())
        except Exception as error:
            self._trace(
                "details", params, started,
                cache=self._failed_cache_state(path), error=error,
            )
            raise
        self._trace("details", params, started, count=1, cache=cache_state)
        return details

    async def _search_live(
        self,
        keyword: str,
        price_min: int | None,
        price_max: int | None,
        condition_ids: list[int],
        sort: str,
    ) -> list[Item]:
        """Performs the network search and normalises the full page."""
        sort_by, sort_order = _SORT_OPTIONS[sort]
        await self._throttle()
        try:
            results = await self._api.search(
                keyword,
                price_min=price_min,
                price_max=price_max,
                item_conditions=condition_ids,
                sort_by=sort_by,
                sort_order=sort_order,
                # Sold listings are never useful to a shopper, so this is
                # fixed rather than exposed as a tool parameter.
                status=[SearchRequestData.Status.STATUS_ON_SALE],
            )
        except Exception as error:
            raise _translate_transport_error(error) from error

        return [_to_item(raw) for raw in results.items[:_MAX_PAGE_SIZE]]

    async def _details_live(self, item_id: str) -> ItemDetails:
        """Performs the network detail lookup and normalises the response."""
        await self._throttle()
        try:
            raw = await self._api.item(item_id)
        except Exception as error:
            raise _translate_transport_error(error) from error

        if raw is None:
            raise ItemNotFoundError(f"No listing found for id {item_id!r}")
        return _to_item_details(raw)

    async def _throttle(self) -> None:
        """Spaces out network calls by `rate_limit_seconds`."""
        if self.rate_limit_seconds <= 0:
            return
        async with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_at
            wait = self.rate_limit_seconds - elapsed
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    def _read_cassette(self, path: Path) -> Any | None:
        """Returns cassette contents, or None if the network should be used.

        Cassettes store already-normalised objects rather than raw Mercari
        payloads. That keeps replay trivial, at the cost of having to
        re-record if the set of extracted fields ever changes.
        """
        if self.mode == "live":
            return None
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.mode == "replay" and self.strict_replay:
            raise FixtureMissingError(
                f"No cassette at {path}. Re-record, or disable strict replay."
            )
        return None

    def _failed_cache_state(self, path: Path) -> str:
        """Labels a failed call by where it was trying to read from.

        Without this the failure path would report every error as "live",
        which would quietly corrupt the cache-hit rate in the metrics.
        """
        if self.mode == "live":
            return "live"
        return "hit" if path.exists() else "miss"

    def _write_cassette(self, path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _trace(
        self,
        operation: str,
        params: dict[str, Any],
        started: float,
        count: int = 0,
        cache: str = "live",
        error: Exception | None = None,
    ) -> None:
        """Emits one metrics record; never raises into the caller's path."""
        if self._trace_hook is None:
            return
        record = {
            "op": operation,
            "params": params,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "count": count,
            "cache": cache,
            "error": type(error).__name__ if error else None,
        }
        try:
            self._trace_hook(record)
        except Exception:  # pylint: disable=broad-except
            pass


def _translate_transport_error(error: Exception) -> Exception:
    """Maps httpx failures onto the local error taxonomy.

    Keeping these distinct is what lets the evaluation harness say "three
    rate-limit failures, one timeout" instead of "four errors".
    """
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code == 429:
            return RateLimitError("Mercari rate-limited this client")
        return NetworkError(
            f"Mercari returned HTTP {error.response.status_code}"
        )
    if isinstance(error, httpx.RequestError):
        return NetworkError(f"Request to Mercari failed: {error}")
    return error


def _to_item(raw: Any) -> Item:
    """Normalises one search hit."""
    return Item(
        id=raw.id_,
        name=raw.name,
        price=raw.price,
        condition=models.condition_label(
            getattr(raw, "item_condition_id", None)
        ),
        status=_normalise_status(getattr(raw, "status", None)),
        url=models.ITEM_URL_TEMPLATE.format(item_id=raw.id_),
    )


def _to_item_details(raw: Any) -> ItemDetails:
    """Normalises one detail response."""
    seller = getattr(raw, "seller", None)
    ratings = getattr(seller, "ratings", None) if seller else None
    condition = getattr(raw, "item_condition", None)

    return ItemDetails(
        id=raw.id_,
        name=raw.name,
        price=raw.price,
        condition=getattr(condition, "name", "unknown"),
        status=_normalise_status(getattr(raw, "status", None)),
        url=models.ITEM_URL_TEMPLATE.format(item_id=raw.id_),
        description=getattr(raw, "description", "") or "",
        seller=SellerRating(
            score=getattr(seller, "score", 0) or 0,
            num_ratings=getattr(seller, "num_ratings", 0) or 0,
            num_good=getattr(ratings, "good", 0) or 0,
            num_bad=getattr(ratings, "bad", 0) or 0,
            num_sell_items=getattr(seller, "num_sell_items", 0) or 0,
        ),
        shipping_payer=getattr(
            getattr(raw, "shipping_payer", None), "name", "unknown"
        ),
        shipping_duration=getattr(
            getattr(raw, "shipping_duration", None), "name", "unknown"
        ),
        num_likes=getattr(raw, "num_likes", 0) or 0,
    )