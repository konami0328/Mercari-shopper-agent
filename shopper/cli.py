"""Interactive command line for the shopping agent.

The interface is a REPL rather than a one-shot command because
clarification is inherently a two-turn exchange: the agent asks, the user
answers, and the answer has to land in the same conversation. Multi-turn
plumbing therefore exists from the first version; what a later milestone
adds on top is preference memory, not the transport.

Usage:
    python -m shopper                  # interactive
    python -m shopper "<query>"        # single query, then exit
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from rich.console import Console, Group
from rich.padding import Padding
from typing import Any

# Redundant when the entry point is `python -m shopper` — shopper's
# __init__.py already calls this before any submodule loads. Kept here
# too as a defensive second call: python-dotenv is idempotent by default
# (it does not override variables already set), and this covers the case
# of running this file directly rather than through the package.
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from shopper.agent import Agent, Incomplete, Question, Recommendation, listing_of
from shopper.config import RunBudget
from shopper.mercari_client import MercariClient
from shopper.trace import Tracer


console = Console()

CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}


def render(outcome: Any) -> None:
    """Prints whichever of the three outcomes came back."""
    if isinstance(outcome, Question):
        _render_question(outcome)
    elif isinstance(outcome, Recommendation):
        _render_recommendation(outcome)
    elif isinstance(outcome, Incomplete):
        _render_incomplete(outcome)


def _render_question(outcome: Question) -> None:
    body = Text(outcome.reason + "\n\n" if outcome.reason else "")
    for index, question in enumerate(outcome.questions, start=1):
        body.append(f"  {index}. {question}\n")
    console.print(Panel(body, title="Need a little more", border_style="cyan"))


def _render_recommendation(outcome: Recommendation) -> None:
    if not outcome.items:
        console.print(Panel(
            Text(outcome.notes or "Nothing suitable was found."),
            title="No recommendations",
            border_style="red",
        ))
        return

    for rank, entry in enumerate(outcome.items, start=1):
        listing = listing_of(entry)
        if listing is None:
            continue

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", width=10, no_wrap=True)
        # overflow="fold" is the fix. Rich's default is "ellipsis", which
        # silently truncates anything wider than the measured column —
        # including the reason, which is the point of the whole run.
        table.add_column(overflow="fold")
        table.add_row("Price", f"[bold]¥{listing.price:,}[/bold]")
        table.add_row("Condition", listing.condition)
        seller = getattr(listing, "seller", None)
        if seller is not None:
            table.add_row(
                "Seller",
                f"score {seller.score} "
                f"({seller.num_good} good / {seller.num_bad} bad)",
            )
            # The value already reads "送料込み(出品者負担)"; prefixing it
            # with "paid by" says the same thing twice.
            table.add_row("Shipping", listing.shipping_payer)
        table.add_row("Link", f"[link={listing.url}]{listing.url}[/link]")

        # The reason gets the full panel width, below the facts.
        body = Group(table, Padding(Markdown(entry["reason"]), (1, 1, 0, 1)))

        console.print(Panel(
            body,
            title=f"[bold]{rank}. {listing.name}[/bold]",
            title_align="left",
            border_style="blue",
        ))

    style = CONFIDENCE_STYLE.get(outcome.confidence, "white")
    console.print(f"Confidence: [{style}]{outcome.confidence}[/{style}]")
    if outcome.notes:
        console.print(f"[dim]{outcome.notes}[/dim]")
    for warning in outcome.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")


def _render_incomplete(outcome: Incomplete) -> None:
    console.print(Panel(
        Text(outcome.text or outcome.reason),
        title=f"Incomplete ({outcome.reason})",
        border_style="red",
    ))


async def converse(agent: Agent, first_query: str | None, verbose: bool) -> None:
    """Drives the REPL until the user leaves."""
    query = first_query
    while True:
        if query is None:
            try:
                query = console.input("\n[bold cyan]you >[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nBye.")
                return
        if not query:
            query = None
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            console.print("Bye.")
            return

        with console.status("Searching Mercari...", spinner="dots"):
            outcome = await agent.run(query)
        render(outcome)

        if verbose:
            totals = agent.tracer.run_totals()
            console.print(
                f"[dim]{totals['llm_calls']} LLM calls, "
                f"{totals['tool_calls']} tool calls, "
                f"{totals['input_tokens']}+{totals['output_tokens']} tokens, "
                f"{totals['latency_ms']:.0f}ms in the model[/dim]"
            )

        if first_query is not None and not isinstance(outcome, Question):
            return
        first_query = None
        query = None


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Mercari Japan AI shopper")
    parser.add_argument("query", nargs="?", help="Run one query and exit.")
    parser.add_argument(
        "--mode",
        default="live",
        choices=["live", "record", "replay"],
        help="Data-layer mode. Use record/replay for reproducible runs.",
    )
    parser.add_argument(
        "--searches", type=int, default=RunBudget.max_searches,
        help="Maximum searches per turn.",
    )
    parser.add_argument(
        "--limit", type=int, default=RunBudget.search_limit,
        help="Listings shown to the model per search.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print token and latency totals."
    )
    args = parser.parse_args(argv)

    tracer = Tracer()
    budget = RunBudget(max_searches=args.searches, search_limit=args.limit)
    client = MercariClient(mode=args.mode, trace_hook=tracer.mercari_hook)
    agent = Agent(mercari=client, budget=budget, tracer=tracer)

    console.print(Panel(
        Text("Ask for anything on Mercari Japan. "
             "Japanese or English. Type 'exit' to leave."),
        title="Mercari shopper",
        border_style="green",
    ))
    try:
        asyncio.run(converse(agent, args.query, args.verbose))
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())