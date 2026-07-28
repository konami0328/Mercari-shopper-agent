"""The agent loop.

No agent framework is used. The loop is the whole mechanism:

    messages = [user turn]
    repeat:
        response = LLM(system, tools, messages)
        append the assistant turn verbatim
        for each tool_use block: execute it, collect a tool_result
        append all tool_results as one user turn
        stop if a terminal tool succeeded, or the model stopped calling tools

Three details are easy to get wrong and are load-bearing here:

  * The assistant turn is appended as a whole, including any text blocks.
    Filtering it down to the tool calls corrupts the history.
  * A single assistant turn may contain several tool_use blocks. Every one
    needs a tool_result, and all of them go into one user message.
  * That applies to terminal tools too. A tool_use without a matching
    tool_result makes the API reject the history on the next turn, which
    would surface as a mysterious 400 the first time multi-turn is used.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any, Literal

from anthropic import AsyncAnthropic

from shopper.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MODEL,
    RunBudget,
)
from shopper.mercari_client import MercariClient
from shopper.models import Item, ItemDetails
from shopper.prompts import SYSTEM_PROMPT
from shopper.tools import ToolExecutor, build_tool_schemas
from shopper.trace import Tracer

import shopper.tools as tools_module


@dataclasses.dataclass
class Recommendation:
    """A delivered set of recommendations, joined back to listing data.

    The model returns only ids and reasons. The listing fields needed for
    display are looked up locally rather than parsed back out of the
    conversation, which is also what makes it possible to notice an id the
    model invented.
    """

    kind: Literal["recommendation"] = "recommendation"
    items: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    confidence: str = "low"
    notes: str = ""
    warnings: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Question:
    """The agent needs information before it can search usefully."""

    kind: Literal["question"] = "question"
    reason: str = ""
    questions: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Incomplete:
    """The run ended without a structured answer.

    Distinct from a recommendation with low confidence: this means the
    loop hit a wall (turn budget, repeated tool failures, or the model
    replying in prose instead of calling the closing tool).
    """

    kind: Literal["incomplete"] = "incomplete"
    reason: str = ""
    text: str = ""


Outcome = Recommendation | Question | Incomplete


class Agent:
    """A single conversation with the shopping agent.

    Message history lives on the instance, not inside `run()`, so that
    successive user turns share context. That is what makes clarification
    work: the agent asks, the user answers, and the follow-up turn still
    knows what was being searched for.

    Attributes:
        messages: The full conversation in Anthropic message format.
        executor: Holds retrieved listings and the run-scoped budgets.
    """

    def __init__(
        self,
        mercari: MercariClient,
        budget: RunBudget | None = None,
        tracer: Tracer | None = None,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = DEFAULT_BASE_URL,
    ) -> None:
        """Initialises the agent.

        Args:
            mercari: Data-layer client, injected so that a whole
                evaluation sweep shares one rate limiter.
            budget: Per-run limits; defaults are in `config.RunBudget`.
            tracer: Receives per-call metrics. One is created if omitted.
            model: Model id to call.
            api_key: API key. Falls back to ANTHROPIC_API_KEY.
            base_url: Messages API endpoint. Left as None for Anthropic;
                set it to point the same loop at another provider that
                serves the Anthropic wire format. This is why there is no
                provider abstraction layer in this codebase.
        """
        self.model = model
        self.base_url = base_url
        self.budget = budget or RunBudget()
        self.tracer = tracer or Tracer()
        self.llm = AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )
        self.tool_schemas = build_tool_schemas(self.budget)
        self.messages: list[dict[str, Any]] = []
        self.executor = ToolExecutor(
            mercari, self.budget, on_event=self._on_executor_event
        )

    async def run(self, user_message: str) -> Outcome:
        """Processes one user turn to completion.

        Args:
            user_message: What the user just said.

        Returns:
            A `Recommendation`, a `Question`, or an `Incomplete`.
        """
        self.tracer.start_run(user_message)
        self.executor.begin_run()
        self.messages.append({"role": "user", "content": user_message})

        consecutive_errors = 0

        for turn in range(self.budget.max_turns):
            response = await self._call_llm(turn)
            self.messages.append(
                {"role": "assistant", "content": response.content}
            )

            tool_uses = [
                block for block in response.content
                if getattr(block, "type", None) == "tool_use"
            ]
            if not tool_uses:
                # The model answered in prose. Under normal operation it
                # should have closed with a tool; treat the text as a
                # best-effort reply rather than discarding the turn.
                return Incomplete(
                    reason="model_stopped_without_tool",
                    text=_join_text(response.content),
                )

            tool_results: list[dict[str, Any]] = []
            outcome: Outcome | None = None

            for block in tool_uses:
                started = time.perf_counter()
                result = await self.executor.execute(block.name, block.input)
                self.tracer.emit(
                    "tool_call",
                    turn=turn,
                    tool=block.name,
                    input=block.input,
                    is_error=result.is_error,
                    duration_ms=round(
                        (time.perf_counter() - started) * 1000, 1
                    ),
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(
                        result.payload, ensure_ascii=False, default=str
                    ),
                    "is_error": result.is_error,
                })
                consecutive_errors = (
                    consecutive_errors + 1 if result.is_error else 0
                )
                # Every block is executed and answered before the loop
                # decides to stop, so a terminal call sitting alongside a
                # retrieval call in the same turn cannot leave a tool_use
                # unanswered in the history.
                if result.terminal and outcome is None:
                    outcome = self._build_outcome(block.name, block.input)

            self.messages.append({"role": "user", "content": tool_results})

            if outcome is not None:
                self._finish(outcome.kind)
                return outcome

            if consecutive_errors >= 3:
                self._finish("tool_failures")
                return Incomplete(
                    reason="three_consecutive_tool_failures",
                    text="The data layer kept failing; the run was stopped.",
                )

        self._finish("max_turns")
        return Incomplete(
            reason="max_turns_exceeded",
            text=(
                f"No answer after {self.budget.max_turns} turns. "
                "The run was stopped to avoid looping."
            ),
        )

    async def _call_llm(self, turn: int) -> Any:
        """Makes one LLM call and records its cost and latency."""
        started = time.perf_counter()
        response = await self.llm.messages.create(
            model=self.model,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            system=SYSTEM_PROMPT,
            tools=self.tool_schemas,
            messages=self.messages,
        )
        self.tracer.emit(
            "llm_call",
            turn=turn,
            model=self.model,
            endpoint=self.base_url or "anthropic",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )
        return response

    def _build_outcome(self, name: str, tool_input: dict[str, Any]) -> Outcome:
        """Turns an accepted terminal tool call into a domain object."""
        if name == tools_module.CLARIFY_TOOL:
            return Question(
                reason=tool_input.get("reason", ""),
                questions=list(tool_input.get("questions") or []),
            )

        entries = list(tool_input.get("items") or [])
        enriched: list[dict[str, Any]] = []
        for entry in entries:
            item_id = entry.get("item_id", "")
            enriched.append({
                "reason": entry.get("reason", ""),
                "listing": self.executor.seen_items.get(item_id),
                "details": self.executor.seen_details.get(item_id),
            })
        return Recommendation(
            items=enriched,
            confidence=tool_input.get("confidence", "low"),
            notes=tool_input.get("notes", ""),
            warnings=self.executor.price_warnings(entries),
        )

    def _finish(self, outcome_kind: str) -> None:
        self.tracer.emit(
            "run_end", outcome=outcome_kind, **self.tracer.run_totals()
        )

    def _on_executor_event(self, event: str, fields: dict[str, Any]) -> None:
        self.tracer.emit(event, **fields)


def _join_text(content: list[Any]) -> str:
    """Concatenates the text blocks of a response."""
    return "\n".join(
        block.text for block in content
        if getattr(block, "type", None) == "text"
    ).strip()


def listing_of(entry: dict[str, Any]) -> Item | ItemDetails | None:
    """Prefers the enriched record when one was fetched."""
    return entry.get("details") or entry.get("listing")