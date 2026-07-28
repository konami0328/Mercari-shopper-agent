"""Offline smoke test for the agent loop.

Both the model and the data layer are replaced with scripted fakes, so
this runs with no API key, no network, and in well under a second. What
it checks is loop mechanics, not answer quality:

  * the terminal tools produce the right domain object
  * budgets bind even when the model ignores them
  * a hallucinated listing id is handed back once and then accepted
  * a partial detail failure does not sink the call
  * the invariant that breaks multi-turn if violated: every tool_use block
    in the history has exactly one matching tool_result

Usage:
    python smoke_test_client.py
"""

from __future__ import annotations

import asyncio
import os
import types
from typing import Any

# This test never makes a real LLM call (FakeLLM below), but importing
# `shopper` pulls in `shopper.config`, which now requires ANTHROPIC_MODEL
# at import time so a missing model id fails at startup rather than as a
# buried 404 three modules away. Set a harmless placeholder before the
# import so this test stays runnable with no .env at all.
os.environ.setdefault("ANTHROPIC_MODEL", "offline-smoke-test")

from shopper.agent import Agent
from shopper.config import RunBudget
from shopper.models import Item, ItemDetails, ItemNotFoundError, SellerRating
from shopper.trace import Tracer


def _item(index: int) -> Item:
    return Item(
        id=f"m{index}",
        name=f"テスト商品{index}",
        price=5000 + index * 100,
        condition="目立った傷や汚れなし",
        status="on_sale",
        url=f"https://jp.mercari.com/item/m{index}",
    )


class FakeMercari:
    """Returns five listings for any search; one id always 404s."""

    async def search(self, **_: Any) -> list[Item]:
        return [_item(i) for i in range(1, 6)]

    async def get_item_details(self, item_id: str) -> ItemDetails:
        if item_id == "m3":
            raise ItemNotFoundError("listing sold")
        return ItemDetails(
            id=item_id, name="テスト商品", price=5000, condition="良好",
            status="on_sale", url=f"https://jp.mercari.com/item/{item_id}",
            description="出品者による説明文",
            seller=SellerRating(800, 810, 800, 2, 50),
            shipping_payer="seller", shipping_duration="1-2日", num_likes=3,
        )


class _Block:
    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


def text(body: str) -> _Block:
    return _Block(type="text", text=body)


def call(block_id: str, name: str, payload: dict[str, Any]) -> _Block:
    return _Block(type="tool_use", id=block_id, name=name, input=payload)


class _Response:
    def __init__(self, content: list[_Block], stop: str = "tool_use") -> None:
        self.content = content
        self.stop_reason = stop
        self.usage = types.SimpleNamespace(input_tokens=100, output_tokens=50)


class FakeLLM:
    """Replays a fixed list of responses in order."""

    def __init__(self, script: list[_Response]) -> None:
        self._script = script
        self._index = 0
        self.messages = self

    async def create(self, **_: Any) -> _Response:
        response = self._script[self._index]
        self._index += 1
        return response


def tool_blocks_balanced(messages: list[dict[str, Any]]) -> bool:
    """Every tool_use in the history must have a matching tool_result.

    The API rejects a history that violates this. It only bites once the
    history is replayed on a later turn, which is why it is asserted here
    rather than discovered during multi-turn work.
    """
    uses, results = set(), set()
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                uses.add(block.id)
            if isinstance(block, dict) and block.get("type") == "tool_result":
                results.add(block["tool_use_id"])
    return uses == results


async def scenario(
    label: str, script: list[_Response], budget: RunBudget | None = None
) -> tuple[Any, Agent]:
    agent = Agent(
        mercari=FakeMercari(),
        budget=budget or RunBudget(),
        tracer=Tracer(enabled=False),
        api_key="offline",
    )
    agent.llm = FakeLLM(script)
    outcome = await agent.run("1万円以下のヴィンテージデニム")
    balanced = tool_blocks_balanced(agent.messages)
    status = "ok " if balanced else "BAD"
    print(f"  [{status}] {label:<24} -> {outcome.kind}")
    assert balanced, f"{label}: unbalanced tool blocks"
    return outcome, agent


async def main() -> None:
    print("agent loop smoke test (offline)\n")

    outcome, agent = await scenario("happy path", [
        _Response([text("検索します"),
                   call("t1", "search_mercari",
                        {"keyword": "デニム", "price_max": 10000})]),
        _Response([call("t2", "get_item_details",
                        {"item_ids": ["m1", "m2", "m3"]})]),
        _Response([call("t3", "present_recommendations", {
            "items": [{"item_id": "m1", "reason": "理由1"},
                      {"item_id": "m2", "reason": "理由2"}],
            "confidence": "medium", "notes": "備考"})]),
    ])
    assert len(outcome.items) == 2
    assert sorted(agent.executor.seen_details) == ["m1", "m2"], "m3 should 404"
    print("        partial detail failure survived, 2 of 3 enriched")

    outcome, agent = await scenario("hallucinated id", [
        _Response([call("t1", "search_mercari", {"keyword": "デニム"})]),
        _Response([call("t2", "present_recommendations", {
            "items": [{"item_id": "m999", "reason": "存在しない"}],
            "confidence": "high", "notes": ""})]),
        _Response([call("t3", "present_recommendations", {
            "items": [{"item_id": "m1", "reason": "実在する"}],
            "confidence": "medium", "notes": ""})]),
    ])
    assert agent.executor.validation_retries_used == 1
    print("        rejected once, repaired answer accepted")

    _, agent = await scenario(
        "search budget binds",
        [_Response([call(f"s{i}", "search_mercari", {"keyword": f"k{i}"})])
         for i in range(4)]
        + [_Response([call("p", "present_recommendations", {
            "items": [{"item_id": "m1", "reason": "理由"}],
            "confidence": "low", "notes": ""})])],
        budget=RunBudget(max_searches=2, max_turns=6),
    )
    assert agent.executor.searches_used == 2
    print("        model asked 4 times, 2 reached the network")

    outcome, _ = await scenario("clarification", [
        _Response([call("c1", "ask_clarification", {
            "questions": ["ご予算はいくらですか？", "サイズは？"],
            "reason": "情報が足りません"})]),
    ])
    assert outcome.kind == "question" and len(outcome.questions) == 2

    outcome, _ = await scenario("price violation", [
        _Response([call("t1", "search_mercari",
                        {"keyword": "デニム", "price_max": 5200})]),
        _Response([call("t2", "present_recommendations", {
            "items": [{"item_id": "m5", "reason": "予算超過"}],
            "confidence": "high", "notes": ""})]),
    ])
    assert outcome.warnings, "over-budget listing should be flagged"
    print(f"        {outcome.warnings[0]}")

    outcome, _ = await scenario(
        "prose instead of tool",
        [_Response([text("見つかりませんでした")], stop="end_turn")],
    )
    assert outcome.kind == "incomplete"

    await scenario("terminal beside retrieval", [
        _Response([call("t1", "search_mercari", {"keyword": "デニム"})]),
        _Response([call("t2", "search_mercari", {"keyword": "ジーンズ"}),
                   call("t3", "present_recommendations", {
                       "items": [{"item_id": "m1", "reason": "理由"}],
                       "confidence": "low", "notes": ""})]),
    ])
    print("        both blocks answered before the loop returned")

    await multi_turn_budget_reset()

    print("\nall scenarios passed.")


async def multi_turn_budget_reset() -> None:
    """Regression: budgets are per user turn, not per conversation.

    Observed live before this was fixed: the counters lived on the
    executor for the life of the agent, so by the fourth question of a
    session every search and detail lookup was refused and the agent
    could only apologise. Retrieved listings and the clarification
    allowance must nonetheless survive across turns.
    """
    agent = Agent(
        mercari=FakeMercari(),
        budget=RunBudget(max_searches=2, max_detail_calls=1),
        tracer=Tracer(enabled=False),
        api_key="offline",
    )
    turn = [
        _Response([call("s1", "search_mercari",
                        {"keyword": "デニム", "price_max": 10000})]),
        _Response([call("d1", "get_item_details", {"item_ids": ["m1"]})]),
        _Response([call("p1", "present_recommendations", {
            "items": [{"item_id": "m1", "reason": "理由"}],
            "confidence": "high", "notes": ""})]),
    ]

    agent.llm = FakeLLM(list(turn))
    await agent.run("1万円以下のデニム")
    assert agent.executor.searches_used == 1
    assert agent.executor.declared_price_max == 10000

    # Second turn: no price bound stated, and the full allowance is back.
    agent.llm = FakeLLM([
        _Response([call("s2", "search_mercari", {"keyword": "キーボード"})]),
        _Response([call("d2", "get_item_details", {"item_ids": ["m2"]})]),
        _Response([call("p2", "present_recommendations", {
            "items": [{"item_id": "m2", "reason": "理由"}],
            "confidence": "high", "notes": ""})]),
    ])
    outcome = await agent.run("メカニカルキーボード")
    balanced = tool_blocks_balanced(agent.messages)
    print(f"  [{'ok ' if balanced else 'BAD'}] "
          f"{'budget resets per turn':<24} -> {outcome.kind}")
    assert balanced
    assert outcome.kind == "recommendation", "second turn must not starve"
    assert agent.executor.searches_used == 1, "counter did not reset"
    assert agent.executor.details_used == 1, "counter did not reset"
    assert agent.executor.declared_price_max is None, "price bound leaked"
    assert not outcome.warnings, "stale price bound produced a false warning"
    assert "m1" in agent.executor.seen_items, "earlier listings must persist"
    print("        counters cleared, listings kept, no stale price warning")


if __name__ == "__main__":
    asyncio.run(main())