"""Tool definitions and the dispatcher that executes them.

Four tools are exposed to the model, in two categories:

  Retrieval tools reach the outside world.
    - `search_mercari`     : structured query -> candidate listings
    - `get_item_details`   : shortlisted ids -> descriptions, seller stats

  Structured-output tools do nothing at all when executed. They exist so
  that the model's final act is a schema-shaped object rather than prose.
    - `present_recommendations`
    - `ask_clarification`

The second category is the reason the evaluation harness can check
"recommended a listing that was never retrieved" with one line of Python
instead of parsing free text or asking a judge.

Deliberately not exposed: `limit` and `status` (our context budget and a
fixed product decision, not user needs); translation (native to the
model); filtering and ranking (that is the reasoning under test).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from shopper.config import RunBudget
from shopper.mercari_client import MercariClient
from shopper.models import (
    CONDITION_SLUGS,
    Item,
    ItemDetails,
    MercariError,
)


SEARCH_TOOL = "search_mercari"
DETAILS_TOOL = "get_item_details"
PRESENT_TOOL = "present_recommendations"
CLARIFY_TOOL = "ask_clarification"

# Tools that end the run. They still receive a tool_result before the loop
# returns: the API rejects any message history containing a tool_use block
# without a matching tool_result, and this history is replayed on the next
# turn of a multi-turn conversation.
TERMINAL_TOOLS = frozenset({PRESENT_TOOL, CLARIFY_TOOL})


def build_tool_schemas(budget: RunBudget) -> list[dict[str, Any]]:
    """Builds the tool list sent with every LLM call.

    Args:
        budget: Supplies the limits that are mirrored into the schemas.

    Returns:
        Tool definitions in Anthropic tool-use format.
    """
    return [
        {
            "name": SEARCH_TOOL,
            "description": (
                "Search on-sale listings on Mercari Japan. Returns listing "
                "id, title, price in JPY, and condition.\n\n"
                "Keyword guidance: Mercari's index is Japanese. Translate "
                "the user's intent into Japanese search terms even when "
                "they wrote in English or Chinese. Sellers stuff titles "
                "with keywords, so two or three well-chosen terms retrieve "
                "better than one long phrase.\n\n"
                "Important: this endpoint almost never returns zero "
                "results. For a nonsense query it still returns "
                "plausible-looking but unrelated listings. Never assume "
                "the results are relevant just because they arrived."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": (
                            "Japanese search terms, space separated. "
                            "Example: 'ヴィンテージ デニム リーバイス'."
                        ),
                    },
                    "price_min": {
                        "type": "integer",
                        "description": "Inclusive lower bound in JPY.",
                        "minimum": 0,
                    },
                    "price_max": {
                        "type": "integer",
                        "description": "Inclusive upper bound in JPY.",
                        "minimum": 0,
                    },
                    "condition_at_least": {
                        "type": "string",
                        "enum": list(CONDITION_SLUGS),
                        "description": (
                            "Worst acceptable condition; everything at or "
                            "above it is included. Set this only when the "
                            "user expressed a condition preference."
                        ),
                    },
                    "sort": {
                        "type": "string",
                        "enum": ["score", "price_asc", "price_desc", "newest"],
                        "description": (
                            "Default to 'score'. Use 'price_asc' only when "
                            "the user explicitly asks to sort by price: on "
                            "Mercari it surfaces cheap accessories and "
                            "junk rather than cheap examples of the "
                            "requested item. To find affordable options, "
                            "keep 'score' and set price_max instead."
                        ),
                    },
                },
                "required": ["keyword"],
            },
        },
        {
            "name": DETAILS_TOOL,
            "description": (
                "Fetch full details for a shortlist of listings: the "
                "seller's description, seller rating, shipping terms and "
                "like count. Descriptions are where sellers state "
                "measurements, defects and authenticity, none of which "
                "appear in search results.\n\n"
                "Call this once, on your shortlist only, after you have "
                "narrowed the candidates down. Some ids may fail because "
                "listings sell out quickly; the response reports which."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "item_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": budget.max_detail_items,
                        "description": (
                            "Listing ids from a previous search, at most "
                            f"{budget.max_detail_items}."
                        ),
                    }
                },
                "required": ["item_ids"],
            },
        },
        {
            "name": PRESENT_TOOL,
            "description": (
                "Deliver the final recommendations. This ends the turn. "
                "Every item_id must come from a search you actually ran. "
                "Never invent an id, and never pad the list with listings "
                "you judged irrelevant just to reach three."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_id": {
                                    "type": "string",
                                    "description": "Id from a search result.",
                                },
                                "reason": {
                                    "type": "string",
                                    "description": (
                                        "Two or three sentences on why this "
                                        "listing fits this user. Cite "
                                        "concrete evidence: price against "
                                        "their budget, condition, a detail "
                                        "from the description, seller "
                                        "rating. Avoid generic praise."
                                    ),
                                },
                            },
                            "required": ["item_id", "reason"],
                        },
                        "description": (
                            "Up to three listings, best first. Return fewer "
                            "than three when fewer than three genuinely fit."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "high: several strong matches, requirements "
                            "clearly met. medium: acceptable matches with "
                            "reservations. low: thin or partly mismatched "
                            "candidate pool. Report low honestly rather "
                            "than overstating a weak result."
                        ),
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "Caveats the user should know: assumptions you "
                            "made about unstated requirements, how thin the "
                            "supply was, why fewer than three are listed."
                        ),
                    },
                },
                "required": ["items", "confidence", "notes"],
            },
        },
        {
            "name": CLARIFY_TOOL,
            "description": (
                "Ask the user for information without which any search "
                "would be guesswork. This ends the turn; you cannot both "
                "ask and recommend.\n\n"
                "Use it only in these situations:\n"
                "  1. No product category is identifiable at all "
                "('a gift for my mother', 'something nice').\n"
                "  2. No budget given AND the category spans orders of "
                "magnitude (watches run from 2,000 to 2,000,000 JPY).\n"
                "  3. The term has mutually exclusive readings and picking "
                "wrong invalidates the whole search.\n\n"
                "Otherwise make a reasonable assumption, search, and state "
                "the assumption in your notes. Available once per "
                "conversation, so ask everything you need at once."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 2,
                        "description": (
                            "At most two questions, in the user's language."
                        ),
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "One sentence on why searching now would waste "
                            "the user's time. Shown above the questions."
                        ),
                    },
                },
                "required": ["questions", "reason"],
            },
        },
    ]


@dataclasses.dataclass
class ToolOutcome:
    """Result of executing one tool call.

    Attributes:
        payload: JSON-serialisable body for the tool_result block.
        is_error: Marks the tool_result as an error so the model treats it
            as something to recover from rather than data.
        terminal: True when this call should end the run. A rejected
            terminal call (failed validation, budget exhausted) is not
            terminal: the model gets another chance.
    """

    payload: Any
    is_error: bool = False
    terminal: bool = False


class ToolExecutor:
    """Executes tool calls for a single conversation.

    Holds the run-scoped state that the loop itself should not own:

      * `seen_items`: every listing ever returned to the model. This is
        both the render source for the CLI (the model only returns ids)
        and the ground truth for catching hallucinated ids.
      * Call counters, which are the binding budget layer. Schema limits
        guide the model but are not enforced by the API.

    Attributes:
        seen_items: Listing id -> search-level record.
        seen_details: Listing id -> enriched record, when fetched.
        searches_used: Count of completed `search_mercari` calls.
    """

    def __init__(
        self,
        client: MercariClient,
        budget: RunBudget,
        on_event: Any = None,
    ) -> None:
        """Initialises the executor.

        Args:
            client: Data-layer client. Injected rather than constructed
                internally so one instance, and therefore one rate
                limiter, is shared across an evaluation sweep.
            budget: Per-run limits.
            on_event: Optional callable receiving trace records.
        """
        self.client = client
        self.budget = budget
        self._on_event = on_event

        self.seen_items: dict[str, Item] = {}
        self.seen_details: dict[str, ItemDetails] = {}
        self.searches_used = 0
        self.details_used = 0
        self.clarifications_used = 0
        self.validation_retries_used = 0
        # Tightest bounds the model has asked for, used to check that the
        # final answer respects the constraints the model itself set.
        self.declared_price_max: int | None = None
        self.declared_price_min: int | None = None

    def begin_run(self) -> None:
        """Resets the state that is scoped to a single user turn.

        Budgets bound the work done for one request, not for a whole
        conversation: a user who asks a second question is entitled to a
        fresh allowance, and cost is bounded by `max_turns` within each
        request rather than by starving later ones.

        Two things deliberately survive the reset:

          * `seen_items` / `seen_details`, because a follow-up turn refers
            back to listings from earlier ("the second one, but cheaper"),
            and because they are the basis for catching invented ids.
          * `clarifications_used`, because the tool promises the model one
            clarification per conversation. Resetting it per turn would
            let the agent interrogate the user indefinitely.

        The declared price bounds must reset: carrying a bound from an
        earlier, unrelated request produces warnings about a constraint
        the user never stated for this one.
        """
        self.searches_used = 0
        self.details_used = 0
        self.validation_retries_used = 0
        self.declared_price_max = None
        self.declared_price_min = None

    async def execute(self, name: str, tool_input: dict[str, Any]) -> ToolOutcome:
        """Runs one tool call and returns its outcome.

        Never raises: a failure becomes an error `ToolOutcome` so the model
        can recover, which is the whole point of tool-calling error
        handling. Crashing would throw away a run that is often still
        salvageable.
        """
        try:
            if name == SEARCH_TOOL:
                return await self._search(tool_input)
            if name == DETAILS_TOOL:
                return await self._details(tool_input)
            if name == PRESENT_TOOL:
                return self._present(tool_input)
            if name == CLARIFY_TOOL:
                return self._clarify(tool_input)
        except MercariError as error:
            return ToolOutcome(
                payload=_recoverable(
                    f"{type(error).__name__}: {error}",
                    "Try a different keyword, or proceed to "
                    f"{PRESENT_TOOL} with what you already have.",
                ),
                is_error=True,
            )
        return ToolOutcome(
            payload=_recoverable(
                f"Unknown tool {name!r}.",
                "Use only the tools provided.",
            ),
            is_error=True,
        )

    async def _search(self, tool_input: dict[str, Any]) -> ToolOutcome:
        """Runs a search, registers the results, and trims them for the model."""
        if self.searches_used >= self.budget.max_searches:
            return ToolOutcome(
                payload=_recoverable(
                    "Search budget exhausted "
                    f"({self.searches_used}/{self.budget.max_searches}).",
                    f"Do not search again. Proceed to {PRESENT_TOOL} using "
                    "the candidates you already have, and lower your "
                    "confidence if they are a poor fit.",
                ),
                is_error=True,
            )

        items = await self.client.search(
            keyword=tool_input["keyword"],
            price_min=tool_input.get("price_min"),
            price_max=tool_input.get("price_max"),
            condition_at_least=tool_input.get("condition_at_least"),
            sort=tool_input.get("sort", "score"),
            limit=self.budget.search_limit,
        )
        self.searches_used += 1
        for item in items:
            self.seen_items[item.id] = item

        if tool_input.get("price_max") is not None:
            self.declared_price_max = _tighter(
                self.declared_price_max, tool_input["price_max"], min
            )
        if tool_input.get("price_min") is not None:
            self.declared_price_min = _tighter(
                self.declared_price_min, tool_input["price_min"], max
            )

        return ToolOutcome(
            payload={
                "count": len(items),
                "searches_remaining": (
                    self.budget.max_searches - self.searches_used
                ),
                "items": [item.to_dict() for item in items],
            }
        )

    async def _details(self, tool_input: dict[str, Any]) -> ToolOutcome:
        """Fetches details for a shortlist, reporting per-id failures.

        Listings sell out constantly, so a partial failure is the normal
        case rather than an exception. Failing the whole call would throw
        away good data; silently dropping ids would leave the model
        believing it saw details it never received.
        """
        if self.details_used >= self.budget.max_detail_calls:
            return ToolOutcome(
                payload=_recoverable(
                    "Detail budget exhausted "
                    f"({self.details_used}/{self.budget.max_detail_calls}).",
                    f"Do not call {DETAILS_TOOL} again. Proceed to "
                    f"{PRESENT_TOOL} with the information you have.",
                ),
                is_error=True,
            )

        item_ids = list(tool_input.get("item_ids") or [])
        if not item_ids:
            return ToolOutcome(
                payload=_recoverable(
                    "No item_ids supplied.",
                    "Pass ids returned by a previous search.",
                ),
                is_error=True,
            )

        # The schema caps this, but the API does not enforce schemas, so
        # the cap is applied again here where it is binding.
        overflow = item_ids[self.budget.max_detail_items:]
        item_ids = item_ids[: self.budget.max_detail_items]
        self.details_used += 1

        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for item_id in item_ids:
            try:
                details = await self.client.get_item_details(item_id)
            except MercariError as error:
                errors.append({
                    "item_id": item_id,
                    "reason": type(error).__name__,
                })
                continue
            self.seen_details[item_id] = details
            results.append(details.to_dict())

        payload: dict[str, Any] = {"results": results, "errors": errors}
        if overflow:
            payload["ignored_ids"] = overflow
            payload["note"] = (
                f"At most {self.budget.max_detail_items} ids per call; "
                "the rest were ignored."
            )
        if errors:
            payload["guidance"] = (
                "Listings that failed are likely sold or withdrawn. "
                "Substitute another candidate or continue without them; "
                "do not retry the same ids."
            )
        return ToolOutcome(payload=payload)

    def _present(self, tool_input: dict[str, Any]) -> ToolOutcome:
        """Validates and accepts the final recommendation payload.

        Hard failures (unknown ids, empty list) are handed back once for
        repair, because an id the model invented means the recommendation
        is about a listing that does not exist. Soft failures (price
        outside the bounds the model itself declared) are recorded as
        warnings rather than blocking: they are an evaluation signal about
        internal consistency, not a reason to burn another turn.
        """
        items = tool_input.get("items") or []
        unknown = [
            entry.get("item_id")
            for entry in items
            if entry.get("item_id") not in self.seen_items
        ]
        problems: list[str] = []
        if not items:
            problems.append("The items list is empty.")
        if unknown:
            problems.append(
                "These ids were never returned by a search: "
                f"{', '.join(str(i) for i in unknown)}."
            )

        if problems and self.validation_retries_used < self.budget.max_validation_retries:
            self.validation_retries_used += 1
            return ToolOutcome(
                payload=_recoverable(
                    " ".join(problems),
                    "Recommend only listings whose ids appeared in a "
                    "search result. If you have fewer than three suitable "
                    "candidates, return fewer and say so in notes.",
                ),
                is_error=True,
            )

        warnings: list[str] = list(problems)
        for entry in items:
            item = self.seen_items.get(entry.get("item_id", ""))
            if item is None:
                continue
            if (self.declared_price_max is not None
                    and item.price > self.declared_price_max):
                warnings.append(
                    f"{item.id} costs {item.price} JPY, above the "
                    f"price_max of {self.declared_price_max} it searched with"
                )
            if (self.declared_price_min is not None
                    and item.price < self.declared_price_min):
                warnings.append(
                    f"{item.id} costs {item.price} JPY, below the "
                    f"price_min of {self.declared_price_min} it searched with"
                )

        self._emit("validation", unknown_ids=unknown, warnings=warnings)
        return ToolOutcome(
            payload={"status": "delivered", "shown_to_user": len(items)},
            terminal=True,
        )

    def _clarify(self, tool_input: dict[str, Any]) -> ToolOutcome:
        """Accepts a clarification request, once per conversation."""
        if self.clarifications_used >= self.budget.max_clarifications:
            return ToolOutcome(
                payload=_recoverable(
                    "Clarification already used once in this conversation.",
                    "Make your best assumption, search, and state the "
                    "assumption in your notes.",
                ),
                is_error=True,
            )
        self.clarifications_used += 1
        return ToolOutcome(
            payload={"status": "asked"},
            terminal=True,
        )

    def price_warnings(self, items: list[dict[str, Any]]) -> list[str]:
        """Re-derives the constraint warnings for a delivered payload."""
        warnings: list[str] = []
        for entry in items:
            item = self.seen_items.get(entry.get("item_id", ""))
            if item is None:
                warnings.append(f"{entry.get('item_id')} was never retrieved")
                continue
            if (self.declared_price_max is not None
                    and item.price > self.declared_price_max):
                warnings.append(
                    f"{item.id}: ¥{item.price} exceeds the searched "
                    f"maximum of ¥{self.declared_price_max}"
                )
        return warnings

    def _emit(self, event: str, **fields: Any) -> None:
        if self._on_event is not None:
            self._on_event(event, fields)


def _recoverable(problem: str, next_step: str) -> dict[str, str]:
    """Formats an error the model is expected to recover from.

    An error body that only states the problem invites the model to retry
    the same call until the turn budget is gone. Every error therefore
    carries an explicit instruction for what to do instead.
    """
    return {"error": problem, "what_to_do_next": next_step}


def _tighter(current: int | None, candidate: int, chooser: Any) -> int:
    """Combines two bounds, keeping whichever is more restrictive."""
    if current is None:
        return candidate
    return chooser(current, candidate)