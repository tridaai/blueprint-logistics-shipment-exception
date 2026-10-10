"""Agent tools for the diagnosis step (provider mode).

In provider mode the diagnosis is *agentic*: instead of one composed
call over pre-assembled facts, the model runs a bounded tool loop —
it may pull more policy context, the lane/consignee history, the
computed shipment facts, or the carrier's exception history before it
composes the root cause. The loop itself lives in the backends
(``model_backends.py``, per-provider tool formats); this module owns
what the tools ARE: their schemas (one neutral definition both
provider formats render from) and their dispatch.

Every tool here is a read over data the pipeline already holds — the
policy corpus behind the retriever interface, the store's history
entries, and the computed facts. No tool sends, files, or mutates
anything; the graph's no-external-action rule is untouched.

The bound: ``DIAGNOSIS_MAX_TOOL_CALLS`` (default 4, hard cap 6) caps
how many tool calls one diagnosis may make. When the budget runs out
the model is told to compose with what it has.
"""

from __future__ import annotations

import time

from .config import env_int, load_dotenv
from .schemas import ShipmentInput
from .store import carrier_summary, format_type_counts
from .tracing import tool_span

DEFAULT_MAX_TOOL_CALLS = 4
HARD_MAX_TOOL_CALLS = 6

# Neutral tool definitions: name, description, JSON-schema parameters.
# The Anthropic backend renders these as {name, description,
# input_schema}; the OpenAI backend as {type: "function", function:
# {name, description, parameters}}. One definition, two wire formats.
TOOL_SPECS: list[dict] = [
    {
        "name": "search_policies",
        "description": (
            "Search the policy corpus with your own query wording and get "
            "the top matching SOP snippets with their policy IDs. Use when "
            "the governing policies already provided look incomplete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query in operational wording."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "lane_history",
        "description": (
            "Prior exception history for a consignee and a lane "
            "(origin -> destination) from the stored history of analysed "
            "shipments: counts and most recent exception types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "consignee": {"type": "string"},
                "origin": {"type": "string"},
                "destination": {"type": "string"},
            },
            "required": ["consignee", "origin", "destination"],
        },
    },
    {
        "name": "shipment_facts",
        "description": (
            "The computed facts for THIS shipment: exception type, "
            "severity, computed delay hours, document mismatches, and "
            "extraction cross-check discrepancies. Always prefer these "
            "over assumptions."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "carrier_history",
        "description": (
            "Prior shipments with a carrier from the stored history: "
            "total count and exception counts by type (damage, delay, …)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"carrier": {"type": "string"}},
            "required": ["carrier"],
        },
    },
]


def diagnosis_max_tool_calls() -> int:
    """The tool-call budget for one diagnosis (env, read at run time)."""
    load_dotenv()
    return min(max(env_int("DIAGNOSIS_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS), 0), HARD_MAX_TOOL_CALLS)


class DiagnosisToolBox:
    """Dispatches the diagnosis tools over the data one run already holds.

    ``dispatch(name, args)`` returns ``(result_text, summary)``: the
    text fed back to the model, and a one-line summary for the trace.
    Unknown tools and bad arguments raise — the backend loop converts a
    raise into an error tool-result and the model composes without it,
    so a broken tool degrades the diagnosis, never the run.
    """

    def __init__(
        self,
        *,
        shipment: ShipmentInput,
        classification: dict,
        delay_hours: float | None,
        mismatches: list[dict],
        discrepancies: list[str] | None = None,
        document_check_warning: str | None = None,
        retriever=None,
        priors: list[dict] | None = None,
    ) -> None:
        self._shipment = shipment
        self._classification = classification
        self._delay_hours = delay_hours
        self._mismatches = mismatches
        self._discrepancies = discrepancies or []
        self._document_check_warning = document_check_warning
        self._retriever = retriever
        self._priors = priors or []

    def dispatch(self, name: str, args: dict) -> tuple[str, str]:
        handler = {
            "search_policies": self._search_policies,
            "lane_history": self._lane_history,
            "shipment_facts": self._shipment_facts,
            "carrier_history": self._carrier_history,
        }.get(name)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        # One tool span per invocation (tracing.tool_span): the
        # trace tree gains the diagnosis' lookups between its
        # provider calls, each carrying its measured duration —
        # so a slow diagnosis answers "waiting on the model, or
        # looking things up?" at a glance. The span carries the
        # tool's name, duration, and outcome only; arguments and
        # results stay out of the trace, per the tracing module's
        # no-content contract. With tracing off this is a no-op
        # hop, exactly like every other span in the run.
        started = time.perf_counter()
        with tool_span(name) as handle:
            try:
                result = handler(args or {})
            except Exception:
                handle.set_attribute("status", "error")
                handle.set_attribute(
                    "duration_ms", round((time.perf_counter() - started) * 1000, 3)
                )
                raise
            handle.set_attribute("status", "ok")
            handle.set_attribute(
                "duration_ms", round((time.perf_counter() - started) * 1000, 3)
            )
            return result

    # -- the tools ------------------------------------------------------

    def _search_policies(self, args: dict) -> tuple[str, str]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("search_policies needs a non-empty 'query'")
        if self._retriever is None:
            raise ValueError("no retriever is configured for this run")
        results = self._retriever.retrieve(query, top_k=3)
        if not results:
            return "No policies matched that query.", f"search_policies({query!r}) -> 0 policies"
        text = "\n".join(
            f"[{r.policy_id}] {r.title}: {r.snippet}" for r in results
        )
        ids = ", ".join(r.policy_id for r in results)
        return text, f"search_policies({query!r}) -> {len(results)} policies: {ids}"

    def _lane_history(self, args: dict) -> tuple[str, str]:
        shipment = self._shipment
        consignee = str(args.get("consignee") or shipment.customer_name)
        origin = str(args.get("origin") or shipment.origin)
        destination = str(args.get("destination") or shipment.destination)
        lane = f"{origin} -> {destination}"

        def exceptions(entries: list[dict]) -> list[dict]:
            return [e for e in entries if e.get("exception_type") != "none"]

        consignee_hits = exceptions(
            [e for e in self._priors if consignee and e.get("consignee") == consignee]
        )
        lane_hits = exceptions([e for e in self._priors if e.get("lane") == lane])
        if not consignee_hits and not lane_hits:
            return (
                f"No prior exceptions for consignee {consignee!r} or lane {lane} in the stored history.",
                f"lane_history -> no prior exceptions (consignee {consignee!r}, lane {lane})",
            )
        lines = []
        if consignee_hits:
            recent = ", ".join(e["exception_type"] for e in consignee_hits[:3])
            lines.append(
                f"{len(consignee_hits)} prior exception(s) for consignee {consignee!r} "
                f"(most recent: {recent})"
            )
        if lane_hits:
            recent = ", ".join(e["exception_type"] for e in lane_hits[:3])
            lines.append(
                f"{len(lane_hits)} prior exception(s) on lane {lane} (most recent: {recent})"
            )
        summary = (
            f"lane_history -> {len(consignee_hits)} consignee exception(s), "
            f"{len(lane_hits)} lane exception(s)"
        )
        return "; ".join(lines), summary

    def _shipment_facts(self, args: dict) -> tuple[str, str]:
        c = self._classification
        lines = [
            f"exception_type: {c.get('exception_type')} (severity {c.get('severity')}, "
            f"confidence {c.get('confidence')})",
            f"computed delay_hours: {self._delay_hours}",
            f"route: {self._shipment.origin} -> {self._shipment.destination} "
            f"(carrier: {self._shipment.carrier})",
        ]
        if self._mismatches:
            lines.append(
                "document mismatches: "
                + "; ".join(
                    f"{m['field']}: BOL={m.get('bol_value')} vs invoice={m.get('invoice_value')}"
                    for m in self._mismatches
                )
            )
        else:
            lines.append("document mismatches: none computed")
        if self._document_check_warning:
            lines.append(f"document check warning: {self._document_check_warning}")
        lines += [f"extraction discrepancy: {d}" for d in self._discrepancies]
        summary = (
            f"shipment_facts -> {c.get('exception_type')}, delay_hours={self._delay_hours}, "
            f"{len(self._mismatches)} mismatch(es)"
        )
        return "\n".join(lines), summary

    def _carrier_history(self, args: dict) -> tuple[str, str]:
        carrier = str(args.get("carrier") or self._shipment.carrier)
        summary_data = carrier_summary(self._priors, carrier)
        total = summary_data["carrier_count"]
        counts = summary_data["carrier_type_counts"]
        if total == 0:
            return (
                f"No prior shipments with carrier {carrier!r} in the stored history.",
                f"carrier_history({carrier!r}) -> no prior shipments",
            )
        breakdown = format_type_counts(counts) if counts else "no exceptions"
        return (
            f"{total} prior shipment(s) with carrier {carrier!r} in the stored "
            f"history; exceptions by type: {breakdown}.",
            f"carrier_history({carrier!r}) -> {total} prior shipment(s), exceptions: {breakdown}",
        )
