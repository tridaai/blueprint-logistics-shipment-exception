"""Model backends.

Default = ``MockModelBackend``: deterministic, offline, no API key. This is
what the tests, the evals, and the default demo run use, so results are
reproducible.

Optional = OpenAI / Anthropic backends, selected with ``MODEL_BACKEND``.
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
from typing import Protocol

from .config import env_float, env_str, load_dotenv
from .prompts import (
    CLASSIFY_SYSTEM_PROMPT,
    CLASSIFY_USER_TEMPLATE,
    DRAFT_SYSTEM_PROMPT,
    DRAFT_USER_TEMPLATE,
    EXTRACT_SYSTEM_PROMPT,
    EXTRACT_USER_TEMPLATE,
)

# Shared request timeout for provider API calls (seconds), overridable
# with the LLM_TIMEOUT_SECONDS environment variable.
DEFAULT_TIMEOUT_SECONDS = 60.0

_EXCEPTION_TYPES = {"delay", "damage", "document_mismatch", "missed_appointment", "none"}
_SEVERITIES = {"low", "medium", "high", "critical"}


class DraftContext(dict):
    """Loose dict of values used to render a draft (see graph.draft node)."""


class ModelBackend(Protocol):
    name: str

    def draft_customer_update(self, context: DraftContext) -> tuple[str, str]:
        """Return (subject, body) for a customer update draft."""
        ...


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

    def draft_customer_update(self, context: DraftContext) -> tuple[str, str]:
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

    def _render_prompt(self, context: DraftContext) -> str:
        policies = "\n".join(
            f"- [{p['policy_id']}] {p['title']}: {p['snippet']}"
            for p in context.get("policy_details", [])
        )
        return DRAFT_USER_TEMPLATE.format(
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
            policies=policies or "none retrieved",
        )

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
        }
        # Optional: point at Azure OpenAI, a LiteLLM gateway, or any hosted
        # OpenAI-compatible endpoint without code changes.
        base_url = env_str("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAI(**client_kwargs)
        self._model = env_str("OPENAI_MODEL", "gpt-4o-mini")

    def _complete(self, system: str, user: str, max_tokens: int = 600) -> str:  # pragma: no cover - network
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


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
        }
        base_url = env_str("ANTHROPIC_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**client_kwargs)
        self._model = env_str("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def _complete(self, system: str, user: str, max_tokens: int = 600) -> str:  # pragma: no cover - network
        message = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if block.type == "text")


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
    raise ValueError(f"Unknown MODEL_BACKEND: {selected!r} (expected mock | openai | anthropic)")
