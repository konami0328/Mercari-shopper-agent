"""Domain models and errors for the Mercari data layer.

These types are deliberately independent of `mercapi`, the third-party
library used to talk to Mercari. Every client implementation (live API,
replayed fixtures) converts into these types, so the rest of the agent
never sees a vendor-specific object.
"""

from __future__ import annotations

import dataclasses
from typing import Any


ITEM_URL_TEMPLATE = "https://jp.mercari.com/item/{item_id}"

# Mercari condition ids, ordered from best to worst. The agent works with
# the English slugs; the Japanese labels are what we show back to it.
CONDITION_SLUGS: tuple[str, ...] = (
    "new",
    "like_new",
    "no_noticeable_damage",
    "slight_damage",
    "damaged",
    "poor",
)

CONDITION_SLUG_TO_ID: dict[str, int] = {
    slug: index + 1 for index, slug in enumerate(CONDITION_SLUGS)
}

CONDITION_ID_TO_LABEL: dict[int, str] = {
    1: "新品、未使用",
    2: "未使用に近い",
    3: "目立った傷や汚れなし",
    4: "やや傷や汚れあり",
    5: "傷や汚れあり",
    6: "全体的に状態が悪い",
}


def condition_ids_at_least(slug: str) -> list[int]:
    """Expands a minimum condition into every condition id at or above it.

    Mercari's API takes a set of accepted condition ids, but users think in
    terms of a threshold ("at least in decent shape"). Letting the agent
    pass a single boundary instead of enumerating a set removes a whole
    class of tool-call mistakes.

    Args:
        slug: One of `CONDITION_SLUGS`, the worst condition still accepted.

    Returns:
        Condition ids from "new" down to `slug`, inclusive.

    Raises:
        InvalidParameterError: If `slug` is not a known condition.
    """
    if slug not in CONDITION_SLUG_TO_ID:
        raise InvalidParameterError(
            f"Unknown condition {slug!r}. "
            f"Expected one of: {', '.join(CONDITION_SLUGS)}"
        )
    return list(range(1, CONDITION_SLUG_TO_ID[slug] + 1))


def condition_label(condition_id: int | None) -> str:
    """Returns the Japanese label for a condition id, or 'unknown'."""
    if condition_id is None:
        return "unknown"
    return CONDITION_ID_TO_LABEL.get(condition_id, "unknown")


@dataclasses.dataclass(frozen=True)
class Item:
    """A single search hit, trimmed to what stage-one ranking needs.

    Search results carry roughly twenty raw fields. Everything that cannot
    change the ranking decision (checksums, pager ids, internal flags) is
    dropped here rather than being sent to the model as noise.
    """

    id: str
    name: str
    price: int
    condition: str
    status: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Item":
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class SellerRating:
    """Seller reputation, used as a tie-breaker between similar listings."""

    score: int
    num_ratings: int
    num_good: int
    num_bad: int
    num_sell_items: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SellerRating":
        return cls(**data)


@dataclasses.dataclass(frozen=True)
class ItemDetails:
    """A listing enriched with the fields only the detail endpoint returns.

    The description is the reason this second call exists: it is where
    sellers state measurements, defects and provenance, none of which the
    search endpoint exposes.
    """

    id: str
    name: str
    price: int
    condition: str
    status: str
    url: str
    description: str
    seller: SellerRating
    shipping_payer: str
    shipping_duration: str
    num_likes: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ItemDetails":
        payload = dict(data)
        payload["seller"] = SellerRating.from_dict(payload["seller"])
        return cls(**payload)


class MercariError(Exception):
    """Base class for every data-layer failure.

    Subclasses exist so the evaluation harness can group failures by cause
    instead of counting one undifferentiated pile of exceptions.
    """


class NetworkError(MercariError):
    """The request could not be completed (timeout, DNS, connection reset)."""


class RateLimitError(MercariError):
    """Mercari rejected the request for sending too many too quickly."""


class InvalidParameterError(MercariError):
    """A caller passed an argument the client cannot map onto the API."""


class ItemNotFoundError(MercariError):
    """The requested listing id does not exist or is no longer visible."""


class FixtureMissingError(MercariError):
    """Replay mode was asked for a call that was never recorded.

    In strict replay this is fatal on purpose: silently falling back to the
    network would mean an evaluation run is partly online and partly
    offline, which invalidates any before/after comparison drawn from it.
    """
