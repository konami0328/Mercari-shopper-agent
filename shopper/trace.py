"""Structured tracing for agent runs.

Every LLM call, every tool call and every terminal outcome is appended to
a JSONL file keyed by a run id. The evaluation harness reads these files
rather than re-instrumenting the agent, which is why tracing exists from
the first version instead of being retrofitted once numbers are needed.

One trace file per day; one line per event; one `run_id` per user turn.
"""

from __future__ import annotations

import datetime
import json
import threading
import uuid
from pathlib import Path
from typing import Any


class Tracer:
    """Appends newline-delimited JSON events to disk.

    Attributes:
        run_id: Identifier shared by every event in the current run.
        events: In-memory copy of this session's events, so a caller can
            summarise a run without re-reading the file.
    """

    def __init__(self, trace_dir: Path | str = "traces", enabled: bool = True):
        """Initialises the tracer.

        Args:
            trace_dir: Directory that will hold the JSONL files.
            enabled: When False, events are still collected in memory but
                nothing is written. Useful for unit tests.
        """
        self.trace_dir = Path(trace_dir)
        self.enabled = enabled
        self.run_id = ""
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def start_run(self, user_message: str) -> str:
        """Begins a new run and returns its id."""
        self.run_id = uuid.uuid4().hex[:12]
        self.emit("run_start", user_message=user_message)
        return self.run_id

    def emit(self, event: str, **fields: Any) -> None:
        """Writes one event. Never raises into the caller's path."""
        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        self.events.append(record)
        if not self.enabled:
            return
        try:
            day = datetime.date.today().isoformat()
            path = self.trace_dir / f"{day}.jsonl"
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception:  # pylint: disable=broad-except
            # Losing a trace line must never take down a user-facing run.
            pass

    def mercari_hook(self, record: dict[str, Any]) -> None:
        """Adapter for `MercariClient(trace_hook=...)`.

        Lets data-layer metrics (cache hit rate, per-request latency) land
        in the same stream as agent-level metrics, so one run can be
        reconstructed end to end from a single file.
        """
        self.emit("mercari_call", **record)

    def run_totals(self) -> dict[str, Any]:
        """Aggregates the current run into the numbers the CLI reports."""
        llm = [e for e in self.events
               if e["event"] == "llm_call" and e["run_id"] == self.run_id]
        tools = [e for e in self.events
                 if e["event"] == "tool_call" and e["run_id"] == self.run_id]
        return {
            "llm_calls": len(llm),
            "input_tokens": sum(e.get("input_tokens", 0) for e in llm),
            "output_tokens": sum(e.get("output_tokens", 0) for e in llm),
            "latency_ms": round(sum(e.get("duration_ms", 0.0) for e in llm), 1),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for e in tools if e.get("is_error")),
        }
