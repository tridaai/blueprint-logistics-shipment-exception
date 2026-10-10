"""Draft structure repair: the live-proof drafting fix.

On 2026-10-10 the full demo ran LIVE on NVIDIA Nemotron-3-Super
(13 provider calls). The model's draft was fluent — and failed the
guardrails: it never referenced the shipment ID and stated no next
step, and the bounded repair could not fix it, because the repair
feedback carried only the failure *details*, never the requirement
itself. The fix has three halves, all pinned here:

- the draft prompt carries the required structural elements
  explicitly (shipment ID in the body, a closing next step, the
  cited policies) — ``prompts.py``;
- the repair feedback names each failed check and the element the
  redraft must contain (``guardrails.repair_instructions``), the
  advisory ``states_next_step`` included;
- a stub backend whose first draft repeats the live failure and
  whose repair reply includes the missing elements is repaired
  into a passing draft, end to end through the graph.

The offline template path is untouched by all of this: the mock
backend renders none of these prompts, and its drafts stay
byte-stable (the rest of the suite pins them).
"""

from __future__ import annotations

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.guardrails import guardrail_checks, repair_instructions
from shipment_agent.model_backends import MockModelBackend, OpenAIBackend
from shipment_agent.prompts import DRAFT_SYSTEM_PROMPT, DRAFT_USER_TEMPLATE
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.schemas import ExceptionType, ShipmentInput

SHIPMENT = {
    "shipment_id": "STR-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

# The live failure, distilled: fluent, grounded, properly cited in
# the packet — but the body never names the shipment and never says
# what happens next.
STRUCTURELESS_DRAFT = (
    "Dear Synthetic Customer,\n\n"
    "Your parcel is delayed at the regional hub while the carrier "
    "works through a weather hold. We are sorry for the inconvenience "
    "and are pressing the carrier for progress on your behalf."
)

REPAIRED_DRAFT = (
    "Dear Synthetic Customer,\n\n"
    "Your shipment STR-1 is delayed at the regional hub while the "
    "carrier works through a weather hold. [POL-DELAY-01] The next "
    "step is the carrier's revised arrival confirmation; we will "
    "send the next update within one business day."
)


class StructureStubBackend:
    """A backend that drafts like the live model did — then listens.

    The first ``draft_customer_update`` returns the structureless
    draft; a call carrying ``repair_feedback`` returns the repaired
    one. Every context is recorded so the test can read exactly
    what the repair loop told the drafter.
    """

    name = "stub"

    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def draft_customer_update(self, context) -> tuple[str, str]:
        self.contexts.append(dict(context))
        if context.get("repair_feedback"):
            return "Update on shipment STR-1: delay", REPAIRED_DRAFT
        return "Update on your shipment", STRUCTURELESS_DRAFT

    def usage_totals(self) -> dict:
        return {"input_tokens": 0, "output_tokens": 0, "calls": len(self.contexts)}

    def reset_usage(self) -> None:
        pass


def test_stub_draft_missing_structure_is_repaired_into_a_pass():
    backend = StructureStubBackend()
    result = run_shipment(
        ShipmentInput.model_validate(SHIPMENT),
        backend=backend,
        retriever=KeywordRetriever(),
    )
    # The first draft failed exactly the live failure's checks.
    assert result.repair_attempted is True
    assert result.original_validation is not None
    assert result.original_validation.passed is False
    assert any(
        "shipment ID" in error for error in result.original_validation.errors
    )
    failed = {
        check.name
        for check in result.original_validation.checks
        if not check.passed
    }
    assert "references_shipment_id" in failed
    assert "states_next_step" in failed
    # The repair feedback named the failed checks and the missing
    # elements — including the advisory next-step check, which is
    # not in the blocking errors the old feedback carried.
    assert len(backend.contexts) == 2
    feedback = backend.contexts[1]["repair_feedback"]
    assert "references_shipment_id" in feedback
    assert "states_next_step" in feedback
    assert "STR-1" in feedback
    # And the redraft — which included the elements — passed.
    assert result.repaired is True
    assert result.validation.passed is True
    assert "STR-1" in result.draft.body
    assert "next update" in result.draft.body.lower()


def test_repair_instructions_name_every_failed_check_and_its_fix():
    checks = guardrail_checks(
        STRUCTURELESS_DRAFT,
        shipment_id="STR-1",
        exception_type=ExceptionType.DELAY,
        citations=["POL-DELAY-01"],
    )
    lines = repair_instructions(
        checks, shipment_id="STR-1", citations=["POL-DELAY-01"]
    )
    by_check = {line.split("`")[1]: line for line in lines}
    assert set(by_check) == {"references_shipment_id", "states_next_step"}
    assert "STR-1" in by_check["references_shipment_id"]
    assert "subject line does not count" in by_check["references_shipment_id"]
    assert "next step" in by_check["states_next_step"]


def test_repair_instructions_cover_promises_pii_and_citations():
    body = (
        "Shipment STR-1: we guarantee a full refund. "
        "Contact 123-45-6789 for details. The next update follows."
    )
    checks = guardrail_checks(
        body,
        shipment_id="STR-1",
        exception_type=ExceptionType.DELAY,
        citations=[],
    )
    lines = repair_instructions(checks, shipment_id="STR-1", citations=[])
    text = "\n".join(lines)
    assert "no_prohibited_promises" in text
    assert "no_pii_in_draft" in text
    assert "policy_citations_present" in text
    # A passing check earns no line.
    assert "references_shipment_id" not in text
    assert "states_next_step" not in text


def test_draft_prompt_carries_the_required_elements():
    assert "shipment ID in the body" in DRAFT_SYSTEM_PROMPT
    assert "Required elements" in DRAFT_USER_TEMPLATE
    assert "subject line does not count" in DRAFT_USER_TEMPLATE
    # And the rendered provider prompt carries them too, with the
    # case's own shipment ID filled in.
    backend = OpenAIBackend.__new__(OpenAIBackend)
    prompt = backend._render_prompt(
        {
            "shipment_id": "STR-1",
            "origin": "Memphis, TN",
            "destination": "Charlotte, NC",
            "carrier": "Synthetic Carrier",
            "exception_type": "delay",
            "severity": "high",
            "rationale": "Weather hold.",
            "signals": [],
            "delay_hours": 36.0,
            "mismatches": [],
            "latest_event": "Delayed at regional hub",
            "condition_notes": "",
            "diagnosis_summary": "Carrier weather hold.",
            "recommended_option_text": "",
            "policy_details": [
                {"policy_id": "POL-DELAY-01", "title": "Delay", "snippet": "…"}
            ],
        }
    )
    assert "Required elements" in prompt
    assert "The shipment ID (STR-1) referenced in the body itself" in prompt


def test_mock_template_output_is_unchanged_by_the_prompt_fix():
    """Byte-stability of the offline path: the mock renderer never
    reads the provider prompts, and its draft for a delay case is
    the same text the templates have always produced."""
    backend = MockModelBackend()
    _, body = backend.draft_customer_update(
        {
            "shipment_id": "STR-1",
            "customer_name": "Synthetic Customer",
            "origin": "Memphis, TN",
            "destination": "Charlotte, NC",
            "carrier": "Synthetic Carrier",
            "exception_type": "delay",
            "delay_hours": 36.0,
            "mismatches": [],
            "citations": ["POL-DELAY-01"],
            "latest_event": "",
            "condition_notes": "",
            "recommended_option_text": "",
        }
    )
    assert "Shipment STR-1 is travelling from Memphis, TN to Charlotte, NC" in body
    assert "The next step is" not in body  # delay template: next-update phrasing
    assert "next update" in body
    assert "[POL-DELAY-01]" in body


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr(
        "shipment_agent.model_backends.load_dotenv", lambda *a, **k: None
    )
    monkeypatch.setattr("shipment_agent.graph.load_dotenv", lambda *a, **k: None)
