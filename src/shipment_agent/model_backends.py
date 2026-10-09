"""Model backends.

Default = ``MockModelBackend``: deterministic, offline, no API key. This is
what the tests, the evals, and the demo use, so results are reproducible.

Optional = OpenAI / Anthropic backends, selected with ``MODEL_BACKEND`` in
the environment. They are only constructed when explicitly requested and
raise a clear error if their SDK or API key is missing — they never run
implicitly.
"""

from __future__ import annotations

import os
from typing import Protocol

from .prompts import DRAFT_SYSTEM_PROMPT, DRAFT_USER_TEMPLATE


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
    """Deterministic template renderer. No network, no randomness."""

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
        body = (
            f"Dear {context['customer_name']},\n\n"
            f"{_WHAT_HAPPENED[exception]} "
            f"Shipment {context['shipment_id']} is travelling from {context['origin']} "
            f"to {context['destination']} with {context['carrier']}."
            f"{delay_line}{mismatch_line}\n\n"
            f"{_NEXT_STEP[exception]}\n\n"
            f"Reference policies: {citation_text}\n\n"
            f"Thank you for your patience.\nLogistics Operations Team"
        )
        return subject, body


class _BaseLLMBackend:
    name = "llm"

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
            policies=policies or "none retrieved",
        )

    @staticmethod
    def _split_subject(text: str, fallback_subject: str) -> tuple[str, str]:
        lines = text.strip().splitlines()
        if lines and lines[0].lower().startswith("subject:"):
            return lines[0].split(":", 1)[1].strip(), "\n".join(lines[1:]).strip()
        return fallback_subject, text.strip()


class OpenAIBackend(_BaseLLMBackend):
    name = "openai"

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("MODEL_BACKEND=openai requires OPENAI_API_KEY to be set.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'llm' extra: pip install '.[llm]'") from exc
        self._client = OpenAI()
        self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def draft_customer_update(self, context: DraftContext) -> tuple[str, str]:  # pragma: no cover - network
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": self._render_prompt(context)},
            ],
        )
        text = response.choices[0].message.content or ""
        return self._split_subject(text, f"Update on shipment {context['shipment_id']}")


class AnthropicBackend(_BaseLLMBackend):
    name = "anthropic"

    def __init__(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("MODEL_BACKEND=anthropic requires ANTHROPIC_API_KEY to be set.")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the 'llm' extra: pip install '.[llm]'") from exc
        self._client = anthropic.Anthropic()
        self._model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def draft_customer_update(self, context: DraftContext) -> tuple[str, str]:  # pragma: no cover - network
        message = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            system=DRAFT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self._render_prompt(context)}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return self._split_subject(text, f"Update on shipment {context['shipment_id']}")


def get_backend(name: str | None = None) -> ModelBackend:
    """Select a backend by name or the MODEL_BACKEND env var (default: mock)."""
    selected = (name or os.environ.get("MODEL_BACKEND") or "mock").lower()
    if selected == "mock":
        return MockModelBackend()
    if selected == "openai":
        return OpenAIBackend()
    if selected == "anthropic":
        return AnthropicBackend()
    raise ValueError(f"Unknown MODEL_BACKEND: {selected!r} (expected mock | openai | anthropic)")
