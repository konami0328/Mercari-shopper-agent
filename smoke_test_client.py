"""Smoke test for the Mercari data layer.

Walks the client through the paths that matter before the agent is built:
recording from the network, replaying offline, failing loudly on a replay
miss, and fetching listing details.

Usage:
    python smoke_test_client.py
"""

import asyncio

from shopper import MercariClient
from shopper.models import FixtureMissingError


TRACES: list[dict] = []


def collect_trace(record: dict) -> None:
    """Trace hook standing in for the real telemetry added on day two."""
    TRACES.append(record)
    print(
        f"    [trace] {record['op']:<8} {record['duration_ms']:>7.1f}ms "
        f"cache={record['cache']:<9} n={record['count']} "
        f"err={record['error']}"
    )


async def main() -> None:
    print("1. record mode: hits the network, writes cassettes")
    client = MercariClient(mode="record", trace_hook=collect_trace)
    items = await client.search(
        keyword="ヴィンテージ デニム",
        price_max=10000,
        condition_at_least="no_noticeable_damage",
        limit=20,
    )
    print(f"    got {len(items)} items")
    for item in items[:3]:
        print(f"    - {item.name[:40]} | ¥{item.price} | {item.condition}")

    print("\n2. replay mode: same call, served from disk")
    replay = MercariClient(mode="replay", trace_hook=collect_trace)
    replayed = await replay.search(
        keyword="ヴィンテージ デニム",
        price_max=10000,
        condition_at_least="no_noticeable_damage",
        limit=20,
    )
    print(f"    got {len(replayed)} items, identical={replayed == items}")

    print("\n3. replay miss under strict mode: must raise")
    try:
        await replay.search(keyword="never recorded", limit=5)
        print("    UNEXPECTED: no error raised")
    except FixtureMissingError:
        print("    raised FixtureMissingError as expected")

    print("\n4. detail lookup on the first result")
    if items:
        details = await client.get_item_details(items[0].id)
        print(f"    seller score={details.seller.score} "
              f"ratings={details.seller.num_ratings}")
        print(f"    description length={len(details.description)} chars")
        print(f"    preview: {details.description[:60]}...")

    print("\n5. empty result is a normal return, not an error")
    empty = await client.search(keyword="ｚｚｚ存在しない商品ｚｚｚ", limit=5)
    print(f"    got {len(empty)} items, type={type(empty).__name__}")

    print(f"\ndone. {len(TRACES)} traced operations.")


if __name__ == "__main__":
    asyncio.run(main())
