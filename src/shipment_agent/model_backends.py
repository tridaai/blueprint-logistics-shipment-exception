"""Model backends.

Default = ``MockModelBackend``: deterministic, offline, no API key. This is
what the tests, the evals, and the default demo run use, so results are
reproducible.

Optional = OpenAI / Anthropic / Ollama backends, selected with
``MODEL_BACKEND``. Ollama is the fully local option: a preset over the
OpenAI-compatible client pointed at a local Ollama server, no API key.
Any other OpenAI-compatible hosted provider (LiteLLM, Together, Groq,
…) works through the plain OpenAI backend + ``OPENAI_BASE_URL``.
All surfaces (API, CLI, traced demo) honour the same variables — see the
configuration table in the README. The LLM backends genuinely call the
provider APIs through the official SDKs (the optional ``llm`` extra) and
feed the same pipeline as the mock: the graph assembles the claim packet
and citations, and the guardrails in ``guardrails.py`` run on the model's
draft afterwards, exactly as they do on a template draft.

Extra LLM-only capabilities: when a real backend is active, the graph
also uses it for document extraction (``extractor.py``), an independent
classification that is cross-checked against the deterministic rules
(``crosscheck.py``), the root-cause diagnosis, and recovery-option
proposals. The mock backend deliberately exposes none of those methods,
so none of them can fire in the default mode — the deterministic
fallbacks run instead.

Missing key or missing SDK raises a loud, actionable ``RuntimeError`` —
the backends never run implicitly and never fail silently.
"""

from __future__ import annotations

import json
import re

from .config import env_float, env_str, load_dotenv
from .errors import translate_construction_error, translate_provider_error
from .tracing import provider_span
from .prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_TEMPLATE,
    DIAGNOSE_SYSTEM_PROMPT,
    DIAGNOSE_TOOLS_SYSTEM_PROMPT,
    DIAGNOSE_USER_TEMPLATE,
    INFO_REQUEST_SYSTEM_PROMPT,
    INFO_REQUEST_USER_TEMPLATE,
    DRAFT_SYSTEM_PROMPT,
    DRAFT_USER_TEMPLATE,
    EXTRACT_SYSTEM_PROMPT,
    EXTRACT_USER_TEMPLATE,
    OPTIONS_SYSTEM_PROMPT,
    OPTIONS_USER_TEMPLATE,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
    VERIFY_SYSTEM_PROMPT,
    VERIFY_USER_TEMPLATE,
)

# Shared request timeout for provider API calls (seconds), overridable
# with the LLM_TIMEOUT_SECONDS environment variable.
DEFAULT_TIMEOUT_SECONDS = 60.0

# Indicative USD list prices per 1M tokens (input, output) for the
# telemetry cost ESTIMATE. Verify against your provider — prices move.
# A model missing from the table reports cost as None (unknown), never
# a guessed number; local models price at 0.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "llama3.1": (0.0, 0.0),
    "nomic-embed-text": (0.0, 0.0),
}


def estimate_cost_usd(
    model: str | None, input_tokens: int | None, output_tokens: int | None
) -> float | None:
    """Estimated USD cost of a run's token usage, or None when the
    model is not in the price table (or usage is unknown)."""
    if not model or model not in PRICE_TABLE or input_tokens is None or output_tokens is None:
        return None
    input_rate, output_rate = PRICE_TABLE[model]
    return round(
        (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000, 6
    )

_EXCEPTION_TYPES = {"delay", "damage", "document_mismatch", "missed_appointment", "none"}
_SEVERITIES = {"low", "medium", "high", "critical"}


class DraftContext(dict):
    """Loose dict of values used to render a draft (see graph.draft node)."""


# The ModelBackend protocol is declared in ports.py (the seam
# registry); it is re-exported here so existing imports keep working.
from .ports import ModelBackend  # noqa: E402,F401


_NEXT_STEP = {
    "delay": "We will send the next update as soon as the carrier confirms the revised arrival, and no later than the next business day.",
    "damage": "Our team has opened a claim packet for this shipment. The next step is the condition review; we will send the next update within one business day.",
    "document_mismatch": "We have placed the shipment on a brief document hold and contacted the shipper to correct the conflicting fields. The next step is receiving the corrected documents; we will send the next update within one business day.",
    "missed_appointment": "We are securing the next available delivery appointment with the receiving facility today. The next update will confirm the new appointment window.",
    "none": "No action is needed. The next update will be sent at the next tracking milestone.",
}

_WHAT_HAPPENED = {
    "delay": "Your shipment is delayed. The revised estimated delivery is later than originally scheduled.",
    "damage": "Damage to the shipment was reported during handling/delivery and has been recorded on the delivery documentation.",
    "document_mismatch": "The shipping documents for this shipment do not agree, so we are verifying them before final delivery.",
    "missed_appointment": "The scheduled delivery appointment for this shipment was missed and needs to be rescheduled.",
    "none": "Your shipment is in transit and progressing as planned.",
}


class MockModelBackend:
    """Deterministic template renderer. No network, no randomness.

    The template quotes the source record (latest event, condition notes)
    verbatim in the draft, the way an ops drafter pastes what the carrier
    reported. That is deliberate — and it is why the guardrail layer
    matters: unvetted source text can carry language (a refund promise a
    carrier agent typed into the notes) that must never reach a customer.
    Sample SYN-1013 demonstrates exactly that path.
    """

    name = "mock"

    def __init__(self) -> None:
        self._calls = 0

    def usage_totals(self) -> dict:
        """Call accounting for telemetry. Token counts are honestly
        zero-shaped here — no model ran, and ``run_shipment`` reports
        mock tokens as None rather than dressing up a template render
        as model usage. Only the call count is real."""
        return {"input_tokens": 0, "output_tokens": 0, "calls": getattr(self, "_calls", 0)}

    def reset_usage(self) -> None:
        self._calls = 0

    def draft_customer_update(self, context: DraftContext) -> tuple[str, str]:
        self._calls = getattr(self, "_calls", 0) + 1
        exception = str(context["exception_type"])
        subject = (
            f"Update on shipment {context['shipment_id']}: {exception.replace('_', ' ')}"
            if exception != "none"
            else f"Status update on shipment {context['shipment_id']}"
        )
        citations = context.get("citations") or []
        citation_text = " ".join(f"[{c}]" for c in citations)
        delay_line = ""
        if context.get("delay_hours") is not None and exception == "delay":
            delay_line = f" Current delay against schedule: {context['delay_hours']} hours."
        mismatch_line = ""
        if context.get("mismatches"):
            fields = ", ".join(m["field"] for m in context["mismatches"])
            mismatch_line = f" Fields in conflict: {fields}."

        paragraphs = [
            f"Dear {context['customer_name']},",
            (
                f"{_WHAT_HAPPENED[exception]} "
                f"Shipment {context['shipment_id']} is travelling from {context['origin']} "
                f"to {context['destination']} with {context['carrier']}."
                f"{delay_line}{mismatch_line}"
            ),
        ]
        quoted = []
        if context.get("latest_event"):
            quoted.append(f"Carrier's latest report: \"{context['latest_event']}\"")
        if context.get("condition_notes"):
            quoted.append(f"Condition notes on file: \"{context['condition_notes']}\"")
        if quoted:
            paragraphs.append(" ".join(quoted))
        if context.get("recommended_option_text"):
            paragraphs.append(f"Planned recovery: {context['recommended_option_text']}")
        paragraphs += [
            _NEXT_STEP[exception],
            f"Reference policies: {citation_text}",
            "Thank you for your patience.\nLogistics Operations Team",
        ]
        return subject, "\n\n".join(paragraphs)


def _missing_key_error(backend: str, variable: str) -> RuntimeError:
    return RuntimeError(
        f"MODEL_BACKEND={backend} but {variable} is not set. "
        f"Add {variable}=<your key> to the .env file in the repo root "
        "(copy .env.example to .env) or export it in your shell, then re-run."
    )


def _missing_sdk_error(backend: str, package: str) -> RuntimeError:
    return RuntimeError(
        f"MODEL_BACKEND={backend} needs the {package} SDK, which is not installed. "
        "Install the optional llm extra: uv sync --extra llm "
        "(pip fallback: pip install -e \".[llm]\"), then re-run."
    )


def parse_json_object(text: str) -> dict | None:
    """Parse a model reply that should be a single JSON object.

    Tolerates markdown fences and surrounding prose by falling back to
    the outermost ``{...}`` span. Returns ``None`` when nothing usable
    parses — callers treat an unparseable LLM reply as "no result" and
    fall back to the deterministic path, never as a run failure.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    return data if isinstance(data, dict) else None


def parse_json_list(text: str) -> list | None:
    """Parse a model reply that should be a JSON array (or an object
    wrapping one under an ``options`` key). ``None`` when unusable."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(cleaned[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(data, dict) and isinstance(data.get("options"), list):
        return data["options"]
    return data if isinstance(data, list) else None


def _parse_classification(text: str, backend_name: str) -> dict | None:
    """Parse an LLM classification reply; ``None`` when unusable.

    The cross-check degrades gracefully, so a malformed reply records
    "no LLM classification" instead of failing the run — the rule result
    stands on its own.
    """
    data = parse_json_object(text)
    if data is None:
        return None
    exception_type = str(data.get("exception_type", "")).strip().lower()
    severity = str(data.get("severity", "")).strip().lower()
    if exception_type not in _EXCEPTION_TYPES or severity not in _SEVERITIES:
        return None
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    return {
        "exception_type": exception_type,
        "severity": severity,
        "confidence": min(max(confidence, 0.0), 1.0),
        "rationale": str(data.get("rationale", ""))[:500],
        "backend": backend_name,
    }


class _BaseLLMBackend:
    name = "llm"

    def _complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        """One provider completion. Implemented by each provider backend."""
        raise NotImplementedError

    # -- usage accounting (the LLM eval pack reports tokens + cost) ------
    def _record_usage(self, usage: dict) -> None:
        totals = getattr(self, "_usage_totals", None)
        if totals is None:
            totals = self._usage_totals = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        totals["calls"] += 1

    def usage_totals(self) -> dict:
        """Cumulative provider usage since construction / last reset."""
        return dict(
            getattr(self, "_usage_totals", {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        )

    def reset_usage(self) -> None:
        self._usage_totals = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def complete_with_usage(
        self, system: str, user: str, max_tokens: int = 600, model: str | None = None
    ) -> tuple[str, dict | None]:
        """One completion plus the usage it consumed (``None`` when the
        provider/fake reported none). Used by the LLM-judge eval pack,
        where the judge may run on a different model than the pipeline."""
        original_model = self._model
        if model:
            self._model = model
        before = self.usage_totals()
        try:
            text = self._complete(system, user, max_tokens=max_tokens)
        finally:
            self._model = original_model
        after = self.usage_totals()
        delta = {k: after[k] - before[k] for k in after}
        return text, (delta if delta["calls"] else None)

    def _render_prompt(self, context: DraftContext) -> str:
        policies = "\n".join(
            f"- [{p['policy_id']}] {p['title']}: {p['snippet']}"
            for p in context.get("policy_details", [])
        )
        prompt = DRAFT_USER_TEMPLATE.format(
            shipment_id=context["shipment_id"],
            origin=context["origin"],
            destination=context["destination"],
            carrier=context["carrier"],
            exception_type=context["exception_type"],
            severity=context["severity"],
            rationale=context["rationale"],
            signals="; ".join(context.get("signals", [])) or "none",
            delay_hours=context.get("delay_hours"),
            mismatches=context.get("mismatches") or "none",
            latest_event=context.get("latest_event") or "none recorded",
            condition_notes=context.get("condition_notes") or "none recorded",
            diagnosis=context.get("diagnosis_summary") or "none recorded",
            recommended_option=context.get("recommended_option_text") or "none scored",
            policies=policies or "none retrieved",
        )
        feedback = context.get("repair_feedback")
        if feedback:
            # Bounded-repair redraft: the previous attempt failed
            # validation — hand the model exactly what to fix.
            prompt += (
                "\n\nCORRECTION REQUIRED — a previous draft of this update "
                "failed validation. Rewrite it, fixing every problem "
                "below, and introduce no claim the facts above do not "
                f"support:\n{feedback}"
            )
        return prompt

    @staticmethod
    def _split_subject(text: str, fallback_subject: str) -> tuple[str, str]:
        lines = text.strip().splitlines()
        if lines and lines[0].lower().startswith("subject:"):
            return lines[0].split(":", 1)[1].strip(), "\n".join(lines[1:]).strip()
        return fallback_subject, text.strip()

    def draft_customer_update(self, context: DraftContext) -> tuple[str, str]:
        text = self._complete(DRAFT_SYSTEM_PROMPT, self._render_prompt(context))
        return self._split_subject(text, f"Update on shipment {context['shipment_id']}")

    def classify_with_llm(self, context: DraftContext) -> dict | None:
        """Independent LLM classification for the cross-check (LLM only).

        The model is deliberately NOT shown the rule classifier's result
        — the two paths classify independently and the graph cross-checks
        them (see ``crosscheck.py`` for the resolution policy). Returns a
        dict with exception_type / severity / confidence / rationale /
        backend, or ``None`` when the reply is unusable. Never raises
        for a malformed reply.
        """
        user = CLASSIFY_USER_TEMPLATE.format(
            shipment_id=context["shipment_id"],
            origin=context["origin"],
            destination=context["destination"],
            carrier=context["carrier"],
            status=context.get("status") or "unknown",
            latest_event=context.get("latest_event") or "none recorded",
            condition_notes=context.get("condition_notes") or "none recorded",
            delay_hours=context.get("delay_hours"),
            mismatches=context.get("mismatches") or "none",
        )
        text = self._complete(CLASSIFY_SYSTEM_PROMPT, user, max_tokens=300)
        return _parse_classification(text, self.name)

    def extract_document_fields(
        self, doc_type: str, document_id: str, raw_text: str, fields: list[str]
    ) -> dict[str, dict] | None:
        """Extract typed fields from one document's text (LLM backends only).

        Returns ``{field: {"value": str | None, "confidence": float}}``
        covering exactly the requested fields, or ``None`` when the
        reply is unusable — the extraction node then falls back to the
        source-provided fields for that document.
        """
        user = EXTRACT_USER_TEMPLATE.format(
            doc_type=doc_type,
            document_id=document_id,
            fields=", ".join(fields),
            raw_text=raw_text,
        )
        text = self._complete(EXTRACT_SYSTEM_PROMPT, user, max_tokens=400)
        data = parse_json_object(text)
        if data is None:
            return None
        extracted: dict[str, dict] = {}
        for field in fields:
            entry = data.get(field)
            if not isinstance(entry, dict):
                extracted[field] = {"value": None, "confidence": 0.0}
                continue
            value = entry.get("value")
            try:
                confidence = float(entry.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            extracted[field] = {
                "value": None if value is None else str(value),
                "confidence": min(max(confidence, 0.0), 1.0),
            }
        return extracted

    def _policy_block(self, context: DraftContext) -> str:
        return "\n".join(
            f"- [{p['policy_id']}] {p['title']}: {p['snippet']}"
            for p in context.get("policy_details", [])
        ) or "none retrieved"

    def _diagnose_user(self, context: DraftContext) -> str:
        return DIAGNOSE_USER_TEMPLATE.format(
            shipment_id=context["shipment_id"],
            origin=context["origin"],
            destination=context["destination"],
            carrier=context["carrier"],
            exception_type=context["exception_type"],
            severity=context["severity"],
            rationale=context["rationale"],
            delay_hours=context.get("delay_hours"),
            mismatches=context.get("mismatches") or "none",
            discrepancies="; ".join(context.get("discrepancies", [])) or "none",
            latest_event=context.get("latest_event") or "none recorded",
            condition_notes=context.get("condition_notes") or "none recorded",
            policies=self._policy_block(context),
        )

    @staticmethod
    def _parse_diagnosis(text: str) -> dict | None:
        data = parse_json_object(text)
        if data is None:
            return None
        root_cause = str(data.get("root_cause", "")).strip()
        summary = str(data.get("summary", "")).strip()
        if not root_cause or not summary:
            return None
        return {"root_cause": root_cause[:600], "summary": summary[:300]}

    def diagnose(self, context: DraftContext) -> dict | None:
        """LLM root-cause diagnosis over the computed facts (LLM only).

        Returns ``{"root_cause": ..., "summary": ...}`` or ``None`` when
        the reply is unusable — the pipeline then uses the deterministic
        template diagnosis built from the same evidence.
        """
        return self._parse_diagnosis(
            self._complete(DIAGNOSE_SYSTEM_PROMPT, self._diagnose_user(context), max_tokens=400)
        )

    def _tool_loop(
        self,
        system: str,
        user: str,
        tool_specs: list[dict],
        dispatch,
        max_tool_calls: int,
    ) -> tuple[str, list[dict]]:
        """Run a bounded tool loop; return (final_text, tool_call_log).

        Implemented per provider (the tool wire formats differ). The
        log carries {name, summary} per executed call for the trace.
        """
        raise NotImplementedError

    def diagnose_with_tools(
        self,
        context: DraftContext,
        tool_specs: list[dict],
        dispatch,
        max_tool_calls: int,
    ) -> dict | None:
        """Agentic diagnosis (LLM only): a bounded tool loop, then compose.

        The model may call the tools in ``tools_agent.TOOL_SPECS``
        (policy search, lane history, shipment facts, carrier history)
        through ``dispatch`` before composing. Same return contract as
        :meth:`diagnose`, plus ``tool_calls`` — the executed calls with
        one-line summaries, which the trace shows. ``None`` when the
        final reply is unusable; the pipeline falls back as it does for
        the single-call path. A tool that raises never fails the loop:
        the loop feeds the error back and the model composes without it.
        """
        text, tool_log = self._tool_loop(
            DIAGNOSE_TOOLS_SYSTEM_PROMPT,
            self._diagnose_user(context),
            tool_specs,
            dispatch,
            max_tool_calls,
        )
        data = self._parse_diagnosis(text)
        if data is None:
            return None
        data["tool_calls"] = tool_log
        return data

    def propose_options(self, context: DraftContext) -> list[dict] | None:
        """LLM recovery-option proposals (LLM only) — names, not numbers.

        Returns a list of ``{"kind", "title", "description"}`` dicts in
        reply order, or ``None`` when the reply has no usable entry.
        Kind validation and ALL scoring happen in ``options.py``.
        """
        user = OPTIONS_USER_TEMPLATE.format(
            shipment_id=context["shipment_id"],
            origin=context["origin"],
            destination=context["destination"],
            carrier=context["carrier"],
            exception_type=context["exception_type"],
            severity=context["severity"],
            delay_hours=context.get("delay_hours"),
            mismatches=context.get("mismatches") or "none",
            diagnosis=context.get("diagnosis_summary") or "none recorded",
            policies=self._policy_block(context),
        )
        data = parse_json_list(self._complete(OPTIONS_SYSTEM_PROMPT, user, max_tokens=500))
        if not data:
            return None
        proposals: list[dict] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("kind", "")).strip().lower()
            title = str(entry.get("title", "")).strip()
            description = str(entry.get("description", "")).strip()
            if kind and title and description:
                proposals.append(
                    {"kind": kind, "title": title[:120], "description": description[:300]}
                )
        return proposals or None

    def compose_information_request(self, context: DraftContext) -> str | None:
        """Compose a clarification request (LLM backends only).

        The missing-items list is computed by code (``clarify.py``);
        the model only words the message. Returns the message text, or
        ``None`` when the reply is empty — the caller then uses the
        template wording with the same items.
        """
        user = INFO_REQUEST_USER_TEMPLATE.format(
            shipment_id=context["shipment_id"],
            origin=context["origin"],
            destination=context["destination"],
            carrier=context["carrier"],
            customer_name=context.get("customer_name") or "the consignee",
            missing_items="\n".join(
                f"- {item}" for item in context.get("missing_items", [])
            ),
        )
        text = self._complete(INFO_REQUEST_SYSTEM_PROMPT, user, max_tokens=400)
        return text.strip() or None

    def verify_draft(self, context: dict) -> dict | None:
        """LLM self-verification critique (LLM backends only).

        ``context`` is the plain dict built by ``verify.py``: verified
        facts + the draft. Returns ``{"grounded", "issues", "summary"}``
        or ``None`` when the reply is unusable — the caller then runs
        the deterministic checklist and records the degradation.
        """
        user = VERIFY_USER_TEMPLATE.format(
            shipment_id=context["shipment_id"],
            origin=context.get("origin") or "unknown",
            destination=context.get("destination") or "unknown",
            exception_type=context.get("exception_type") or "unknown",
            severity=context.get("severity") or "unknown",
            delay_hours=context.get("delay_hours"),
            mismatches=context.get("mismatches") or "none",
            policies=self._policy_block(context),
            subject=context.get("subject", ""),
            body=context.get("body", ""),
        )
        data = parse_json_object(self._complete(VERIFY_SYSTEM_PROMPT, user, max_tokens=500))
        if data is None or "grounded" not in data:
            return None
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        return {
            "grounded": bool(data["grounded"]),
            "issues": [str(issue)[:300] for issue in issues],
            "summary": str(data.get("summary", ""))[:300],
        }

    def review_draft(self, context: dict) -> dict | None:
        """Independent LLM review (LLM backends only) — the critic half
        of the generator/critic split, with its own persona prompt.

        ``context`` is the plain dict built by ``reviewer.py``: verified
        facts, the diagnosis, claim-packet contents, and the draft.
        Returns ``{"verdict", "findings", "model"}`` or ``None`` when
        the reply is unusable — the caller then runs the deterministic
        checklist review and records the degradation. The reviewer may
        run on a different model than the pipeline: ``REVIEWER_MODEL``
        when set, else the run's own model.
        """
        packet = context.get("claim_packet") or {}
        packet_diagnosis = packet.get("diagnosis")
        user = REVIEWER_USER_TEMPLATE.format(
            shipment_id=context["shipment_id"],
            origin=context.get("origin") or "unknown",
            destination=context.get("destination") or "unknown",
            exception_type=context.get("exception_type") or "unknown",
            severity=context.get("severity") or "unknown",
            delay_hours=context.get("delay_hours"),
            mismatches=context.get("mismatches") or "none",
            policies=self._policy_block(context),
            diagnosis_root_cause=context.get("diagnosis_root_cause") or "none recorded",
            diagnosis_citations=", ".join(context.get("diagnosis_citations", [])) or "none",
            packet_diagnosis=(
                "present"
                if isinstance(packet_diagnosis, dict) and packet_diagnosis.get("root_cause")
                else "MISSING"
            ),
            packet_citations=", ".join(packet.get("policy_citations") or []) or "none",
            packet_option_count=len(packet.get("recovery_options") or []),
            packet_recommended=packet.get("recommended_option_id") or "none",
            subject=context.get("subject", ""),
            body=context.get("body", ""),
        )
        model = env_str("REVIEWER_MODEL") or None
        text, _usage = self.complete_with_usage(
            REVIEWER_SYSTEM_PROMPT, user, max_tokens=500, model=model
        )
        data = parse_json_object(text)
        if data is None:
            return None
        verdict = str(data.get("verdict", "")).strip().lower()
        if verdict not in {"pass", "concerns", "block"}:
            return None
        findings = data.get("findings") or []
        if not isinstance(findings, list):
            findings = [str(findings)]
        return {
            "verdict": verdict,
            "findings": [str(finding)[:300] for finding in findings],
            "model": model or self._model,
        }


class OpenAIBackend(_BaseLLMBackend):
    name = "openai"

    def __init__(self) -> None:
        api_key = env_str("OPENAI_API_KEY")
        if not api_key:
            raise _missing_key_error("openai", "OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise _missing_sdk_error("openai", "OpenAI") from exc
        client_kwargs: dict = {
            "api_key": api_key,
            "timeout": env_float("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            # No silent SDK retries: a dead endpoint must fail in seconds,
            # and degradation is the pipeline's job (recorded fallbacks),
            # not the transport's. Retrying is a deployment choice made
            # in front of this service, not inside it.
            "max_retries": 0,
        }
        # Optional: point at Azure OpenAI, a LiteLLM gateway, or any hosted
        # OpenAI-compatible endpoint without code changes.
        base_url = env_str("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self.base_url = base_url or "https://api.openai.com/v1"
        try:
            self._client = OpenAI(**client_kwargs)
        except Exception as exc:  # construction failures are translated too
            raise translate_construction_error(
                exc, backend=self.name, base_url=self.base_url
            ) from exc
        self._model = env_str("OPENAI_MODEL", "gpt-4o-mini")

    def _complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        with provider_span(self.name, self._model, "chat_completion") as handle:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            except Exception as exc:
                raise translate_provider_error(
                    exc, backend=self.name, base_url=getattr(self, "base_url", None)
                ) from exc
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._record_usage(
                    {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                    }
                )
                handle.set_counts(
                    input_tokens=getattr(usage, "prompt_tokens", 0),
                    output_tokens=getattr(usage, "completion_tokens", 0),
                )
            return response.choices[0].message.content or ""

    def _create_chat(self, kwargs: dict):
        """chat.completions.create, tolerating clients without tool support.

        Some OpenAI-compatible clients reject the ``tools`` parameter
        outright (TypeError); those get one retry without it and the
        loop then composes from the up-front facts alone.
        """
        with provider_span(self.name, self._model, "chat_completion"):
            try:
                return self._client.chat.completions.create(**kwargs)
            except TypeError:
                if "tools" not in kwargs:
                    raise
                self._tools_unsupported = True
                kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
                return self._client.chat.completions.create(**kwargs)

    def _tool_loop(self, system, user, tool_specs, dispatch, max_tool_calls):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            for spec in tool_specs
        ]
        messages: list = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        log: list[dict] = []
        calls = 0
        while True:
            kwargs: dict = {"model": self._model, "max_tokens": 800, "messages": messages}
            if calls < max_tool_calls and not getattr(self, "_tools_unsupported", False):
                kwargs["tools"] = tools
            try:
                response = self._create_chat(kwargs)
            except Exception as exc:
                raise translate_provider_error(
                    exc, backend=self.name, base_url=getattr(self, "base_url", None)
                ) from exc
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._record_usage(
                    {
                        "input_tokens": getattr(usage, "prompt_tokens", 0),
                        "output_tokens": getattr(usage, "completion_tokens", 0),
                    }
                )
            message = response.choices[0].message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            if not tool_calls or calls >= max_tool_calls:
                return message.content or "", log
            messages.append(message)
            for tool_call in tool_calls:
                if calls >= max_tool_calls:
                    # Still answer the call (the API requires it), but
                    # do not execute it — the budget is spent.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "tool budget exhausted — compose the final answer from the facts you have",
                        }
                    )
                    continue
                calls += 1
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except (json.JSONDecodeError, ValueError):
                    args = {}
                try:
                    result_text, summary = dispatch(name, args)
                except Exception as exc:  # a broken tool degrades, never fails
                    result_text = f"tool error: {exc}"
                    summary = f"{name} -> error: {exc}"
                log.append({"name": name, "summary": summary})
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result_text}
                )


class AnthropicBackend(_BaseLLMBackend):
    name = "anthropic"

    def __init__(self) -> None:
        api_key = env_str("ANTHROPIC_API_KEY")
        if not api_key:
            raise _missing_key_error("anthropic", "ANTHROPIC_API_KEY")
        try:
            import anthropic
        except ImportError as exc:
            raise _missing_sdk_error("anthropic", "Anthropic") from exc
        client_kwargs: dict = {
            "api_key": api_key,
            "timeout": env_float("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            "max_retries": 0,  # see OpenAIBackend — degradation is the pipeline's job
        }
        base_url = env_str("ANTHROPIC_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self.base_url = base_url or "https://api.anthropic.com"
        try:
            self._client = anthropic.Anthropic(**client_kwargs)
        except Exception as exc:  # construction failures are translated too
            raise translate_construction_error(
                exc, backend=self.name, base_url=self.base_url
            ) from exc
        self._model = env_str("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def _complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        with provider_span(self.name, self._model, "messages") as handle:
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except Exception as exc:
                raise translate_provider_error(
                    exc, backend=self.name, base_url=getattr(self, "base_url", None)
                ) from exc
            usage = getattr(message, "usage", None)
            if usage is not None:
                self._record_usage(
                    {
                        "input_tokens": getattr(usage, "input_tokens", 0),
                        "output_tokens": getattr(usage, "output_tokens", 0),
                    }
                )
                handle.set_counts(
                    input_tokens=getattr(usage, "input_tokens", 0),
                    output_tokens=getattr(usage, "output_tokens", 0),
                )
            return "".join(
                block.text for block in message.content if block.type == "text"
            )

    def _create_message(self, kwargs: dict):
        """messages.create, tolerating clients without tool support
        (see the OpenAI backend's ``_create_chat``)."""
        with provider_span(self.name, self._model, "messages"):
            try:
                return self._client.messages.create(**kwargs)
            except TypeError:
                if "tools" not in kwargs:
                    raise
                self._tools_unsupported = True
                kwargs = {k: v for k, v in kwargs.items() if k != "tools"}
                return self._client.messages.create(**kwargs)

    def _tool_loop(self, system, user, tool_specs, dispatch, max_tool_calls):
        tools = [
            {
                "name": spec["name"],
                "description": spec["description"],
                "input_schema": spec["parameters"],
            }
            for spec in tool_specs
        ]
        messages: list = [{"role": "user", "content": user}]
        log: list[dict] = []
        calls = 0
        while True:
            kwargs: dict = {
                "model": self._model,
                "max_tokens": 800,
                "system": system,
                "messages": messages,
            }
            if calls < max_tool_calls and not getattr(self, "_tools_unsupported", False):
                kwargs["tools"] = tools
            try:
                message = self._create_message(kwargs)
            except Exception as exc:
                raise translate_provider_error(
                    exc, backend=self.name, base_url=getattr(self, "base_url", None)
                ) from exc
            usage = getattr(message, "usage", None)
            if usage is not None:
                self._record_usage(
                    {
                        "input_tokens": getattr(usage, "input_tokens", 0),
                        "output_tokens": getattr(usage, "output_tokens", 0),
                    }
                )
            tool_uses = [
                block for block in message.content if getattr(block, "type", None) == "tool_use"
            ]
            if not tool_uses or calls >= max_tool_calls:
                return (
                    "".join(
                        block.text
                        for block in message.content
                        if getattr(block, "type", None) == "text"
                    ),
                    log,
                )
            messages.append({"role": "assistant", "content": message.content})
            results = []
            for block in tool_uses:
                if calls >= max_tool_calls:
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "tool budget exhausted — compose the final answer from the facts you have",
                        }
                    )
                    continue
                calls += 1
                name = block.name
                try:
                    result_text, summary = dispatch(name, dict(block.input or {}))
                except Exception as exc:  # a broken tool degrades, never fails
                    result_text = f"tool error: {exc}"
                    summary = f"{name} -> error: {exc}"
                log.append({"name": name, "summary": summary})
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
                )
            messages.append({"role": "user", "content": results})


class OllamaBackend(OpenAIBackend):
    """Local models via Ollama — a preset over the OpenAI-compatible path.

    Ollama serves an OpenAI-compatible API on localhost, so this backend
    is the OpenAI client pointed at ``OLLAMA_BASE_URL`` (default
    ``http://localhost:11434/v1``) with a placeholder key — Ollama needs
    no real API key. The result is a fully local real-LLM mode: pull a
    model (``ollama pull llama3.1``), set ``MODEL_BACKEND=ollama``, done.
    Embeddings in semantic/hybrid retrieval also come from Ollama in
    this mode (see ``retriever.py``).
    """

    name = "ollama"

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise _missing_sdk_error("ollama", "OpenAI") from exc
        self.base_url = env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        try:
            self._client = OpenAI(
                api_key="ollama",  # placeholder — Ollama ignores it
                base_url=self.base_url,
                timeout=env_float("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
                max_retries=0,  # see OpenAIBackend — degradation is the pipeline's job
            )
        except Exception as exc:  # construction failures are translated too
            raise translate_construction_error(
                exc, backend=self.name, base_url=self.base_url
            ) from exc
        self._model = env_str("OLLAMA_MODEL", "llama3.1")


def get_backend(name: str | None = None) -> ModelBackend:
    """Select a backend by name or the MODEL_BACKEND env var (default: mock).

    Loads the repo-root ``.env`` first (real environment variables win), so
    every surface — API, CLI, traced demo — selects the backend the same way.
    """
    load_dotenv()
    selected = (name or env_str("MODEL_BACKEND") or "mock").lower()
    if selected == "mock":
        return MockModelBackend()
    if selected == "openai":
        return OpenAIBackend()
    if selected == "anthropic":
        return AnthropicBackend()
    if selected == "ollama":
        return OllamaBackend()
    raise ValueError(
        f"Unknown MODEL_BACKEND: {selected!r} (expected mock | openai | anthropic | ollama)"
    )
