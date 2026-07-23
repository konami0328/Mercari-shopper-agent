"""Mercari Japan AI shopper."""

# Loaded first, before any submodule import. shopper.config reads
# ANTHROPIC_MODEL / ANTHROPIC_BASE_URL as soon as it is imported, and
# Python always runs a package's __init__.py before any of its submodules
# — including when the entry point is `python -m shopper`, which imports
# shopper.cli, which imports shopper.agent, which imports shopper.config.
# Calling load_dotenv() anywhere downstream of that chain is too late:
# the values are already resolved. This is the one place guaranteed to
# run before all of them, regardless of entry point.
from dotenv import load_dotenv
load_dotenv()

from shopper.agent import Agent, Incomplete, Question, Recommendation
from shopper.config import RunBudget
from shopper.mercari_client import MercariClient
from shopper.models import (
    CONDITION_SLUGS,
    FixtureMissingError,
    InvalidParameterError,
    Item,
    ItemDetails,
    ItemNotFoundError,
    MercariError,
    NetworkError,
    RateLimitError,
    SellerRating,
)
from shopper.trace import Tracer

__all__ = [
    "Agent",
    "CONDITION_SLUGS",
    "FixtureMissingError",
    "Incomplete",
    "InvalidParameterError",
    "Item",
    "ItemDetails",
    "ItemNotFoundError",
    "MercariClient",
    "MercariError",
    "NetworkError",
    "Question",
    "RateLimitError",
    "Recommendation",
    "RunBudget",
    "SellerRating",
    "Tracer",
]