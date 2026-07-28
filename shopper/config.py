"""Run-scoped configuration for the agent.

Every knob that bounds a single conversation lives here rather than being
scattered as literals through the loop. That is deliberate: the evaluation
harness sweeps these values to produce ablation numbers (e.g. "how much
does quality drop when the agent is allowed only two searches instead of
four?"), which is only cheap if they are all reachable from one object.
"""

from __future__ import annotations

import dataclasses
import os


# No hardcoded fallback on purpose: a wrong guess here (a model id that
# happens not to exist on the caller's key) fails silently as a runtime
# 404 on the first request, three modules away from this file. Requiring
# ANTHROPIC_MODEL to be set surfaces the mistake at startup instead, and
# makes the model actually used explicit in every .env — which matters
# once DeepSeek or another Anthropic-format provider is swapped in via
# ANTHROPIC_BASE_URL, since a model id that is valid on one endpoint is
# usually not valid on the other.
def _require_model() -> str:
    model = os.getenv("ANTHROPIC_MODEL")
    if not model:
        raise RuntimeError(
            "ANTHROPIC_MODEL is not set. Put it in .env, e.g.\n"
            "  ANTHROPIC_MODEL=claude-sonnet-4-5\n"
            "or, when pointing at DeepSeek's Anthropic-format endpoint:\n"
            "  ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic\n"
            "  ANTHROPIC_MODEL=deepseek-v4-pro"
        )
    return model


DEFAULT_MODEL = _require_model()

# Optional override for the Messages API endpoint. Several providers now
# serve the Anthropic wire format, so a cheaper model can be swapped in
# for development iteration by setting this rather than by writing a
# provider abstraction layer. Any numbers reported for the project are
# produced against the default Anthropic endpoint; see README.
DEFAULT_BASE_URL = os.getenv("ANTHROPIC_BASE_URL") or None

# Upper bound on a single completion. The final answer is a tool call with
# three short reasons, so this is generous.
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0


@dataclasses.dataclass(frozen=True)
class RunBudget:
    """Hard limits enforced for one `Agent.run()` call.

    These are the second of three enforcement layers. The first is the tool
    schema itself (`maxItems`, parameter descriptions), which guides the
    model but is not binding: the API does not reject a tool call that
    violates its own schema. The third is `max_turns` in the loop, which
    catches anything the per-tool counters miss.

    Attributes:
        max_turns: Maximum LLM round-trips before the run is abandoned.
        max_searches: Maximum `search_mercari` calls per run.
        max_detail_calls: Maximum `get_item_details` calls per run.
        max_detail_items: Maximum listing ids per `get_item_details` call.
        max_clarifications: Maximum `ask_clarification` calls per run.
        max_validation_retries: How many times a malformed
            `present_recommendations` payload is handed back for repair
            before it is accepted with a warning.
        search_limit: Listings passed to the model per search. Not exposed
            as a tool parameter: it is our context budget, not a user need.
    """

    max_turns: int = 8
    max_searches: int = 4
    max_detail_calls: int = 2
    max_detail_items: int = 5
    max_clarifications: int = 1
    max_validation_retries: int = 1
    search_limit: int = 30