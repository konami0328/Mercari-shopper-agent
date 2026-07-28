"""Probe: what does Mercari's search actually return when nothing matches?

Design Choices section 1 rests on a claim about Mercari's fallback
behaviour. This measures it instead of asserting it.

Four query classes, each expected to behave differently:

  A  garbage ascii      - tokenises to nothing
  B  real Japanese words in an impossible combination
  C  multi-constraint queries where no single listing satisfies all of them
  D  ordinary queries, as a control

Usage:
    python probe_fallback.py
"""

from __future__ import annotations

import asyncio
import os

os.environ.setdefault("ANTHROPIC_MODEL", "offline-probe")

from shopper.mercari_client import MercariClient

QUERIES: list[tuple[str, str]] = [
    ("A", "seit74ah.56awe"),
    ("A", "xkqvvzrt9931"),
    ("A", "zzzqqq.wwweee123"),

    ("B", "宇宙船 デニム"),
    ("B", "透明な 鉄 の 靴下"),
    ("B", "液体 キーボード 恐竜"),

    ("C", "リーバイス 501 1947年 デッドストック 未使用 W28"),
    ("C", "ローランド SH-101 ブルー 完動品 元箱付き"),
    ("C", "ライカ M3 ダブルストローク 1954年 前期型 未使用"),
    ("C", "エルメス バーキン 25 ヒマラヤ 新品未使用"),

    ("D", "デニム"),
    ("D", "メカニカルキーボード"),
    ("D", "アコースティックギター"),
]


async def main() -> None:
    client = MercariClient(mode="live", rate_limit_seconds=1.0)
    print(f"{'cls':<4}{'n':<5}{'query'}")
    print("-" * 64)

    counts: dict[str, list[int]] = {}
    for cls, query in QUERIES:
        try:
            items = await client.search(keyword=query, limit=30)
            n = len(items)
        except Exception as error:  # noqa: BLE001
            print(f"{cls:<4}{'ERR':<5}{query}  ({type(error).__name__})")
            continue
        counts.setdefault(cls, []).append(n)
        print(f"{cls:<4}{n:<5}{query}")
        if n:
            for item in items[:3]:
                print(f"          - {item.name[:52]}  ¥{item.price:,}")

    print("\nsummary")
    print("-" * 64)
    for cls in sorted(counts):
        values = counts[cls]
        nonempty = sum(1 for v in values if v > 0)
        print(f"  class {cls}: {nonempty}/{len(values)} non-empty, "
              f"counts={values}")


if __name__ == "__main__":
    asyncio.run(main())