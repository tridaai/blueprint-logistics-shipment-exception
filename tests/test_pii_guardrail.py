"""PII guardrail tests: `no_pii_in_draft`.

The draft quotes the source record verbatim, so a personal identifier
typed into the condition notes would ride straight into a customer
message without this check. The guardrail scans for SSN-like patterns,
Luhn-valid card-like sequences, and passport-like patterns — and it
must stay silent on everything the bundled samples produce today.
"""

from __future__ import annotations

from shipment_agent.graph import run_shipment
from shipment_agent.guardrails import guardrail_checks, pii_hits, validate_draft
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import sample_shipment_models
from shipment_agent.schemas import ExceptionType, ShipmentInput

BASE_BODY = (
    "Dear Synthetic Customer,\n\n"
    "Your shipment PII-1 is delayed. Reference policies: [POL-DELAY-01]. "
    "The next update will arrive within one business day."
)


def _pii_check(body: str):
    checks = guardrail_checks(body, "PII-1", ExceptionType.DELAY, ["POL-DELAY-01"])
    return next(c for c in checks if c.name == "no_pii_in_draft")


def test_ssn_pattern_fails_and_is_masked_in_the_detail():
    check = _pii_check(BASE_BODY + " Contact on file: 123-45-6789.")
    assert check.passed is False
    assert "SSN-like pattern" in check.detail
    assert "123-45-6789" not in check.detail  # the report masks the value


def test_luhn_valid_card_sequence_fails():
    check = _pii_check(BASE_BODY + " Card noted: 4111 1111 1111 1111.")
    assert check.passed is False
    assert "credit-card-like" in check.detail


def test_luhn_invalid_long_number_passes():
    # Same shape, last digit changed -> Luhn fails -> not a card number.
    check = _pii_check(BASE_BODY + " Reference noted: 4111 1111 1111 1112.")
    assert check.passed is True


def test_passport_like_pattern_fails():
    check = _pii_check(BASE_BODY + " Traveller document C12345678 on file.")
    assert check.passed is False
    assert "passport-like" in check.detail


def test_clean_draft_passes():
    check = _pii_check(BASE_BODY)
    assert check.passed is True
    assert pii_hits(BASE_BODY) == []


def test_validation_error_carries_the_pii_failure():
    result = validate_draft(
        BASE_BODY + " SSN 123-45-6789.", "PII-1", ExceptionType.DELAY, ["POL-DELAY-01"]
    )
    assert result.passed is False
    assert any("PII found in draft" in e for e in result.errors)


def _damage_with_notes(shipment_id: str, notes: str) -> ShipmentInput:
    return ShipmentInput.model_validate(
        {
            "shipment_id": shipment_id,
            "origin": "Memphis, TN",
            "destination": "Charlotte, NC",
            "customer_name": "Acme Parts",
            "latest_event": "Arrived at destination terminal",
            "condition_notes": notes,
            "documents": [],
        }
    )


def test_planted_ssn_fails_the_run_and_engages_repair():
    # The mock template quotes the condition notes verbatim, so the SSN
    # lands in the draft; the guardrail must catch it and the bounded
    # repair loop must engage (and, template being deterministic, the
    # redraft fails identically and approval stays refused).
    shipment = _damage_with_notes(
        "PII-2", "Two cartons crushed. Contact SSN noted as 123-45-6789."
    )
    result = run_shipment(shipment, backend=MockModelBackend(), retriever=KeywordRetriever())
    assert result.validation.passed is False
    pii = next(c for c in result.validation.checks if c.name == "no_pii_in_draft")
    assert pii.passed is False
    assert result.repair_attempted is True
    assert result.repaired is False


def test_luhn_invalid_number_in_notes_leaves_the_run_valid():
    shipment = _damage_with_notes(
        "PII-3", "Two cartons crushed. Dock reference 4111 1111 1111 1112 recorded."
    )
    result = run_shipment(shipment, backend=MockModelBackend(), retriever=KeywordRetriever())
    pii = next(c for c in result.validation.checks if c.name == "no_pii_in_draft")
    assert pii.passed is True
    assert result.validation.passed is True


def test_no_sample_draft_trips_the_pii_guardrail():
    # Zero false positives across the bundled set: every sample's PII
    # check passes today (SYN-1013 fails on promises, not PII).
    for shipment in sample_shipment_models():
        result = run_shipment(shipment, backend=MockModelBackend(), retriever=KeywordRetriever())
        pii = next(c for c in result.validation.checks if c.name == "no_pii_in_draft")
        assert pii.passed, f"{shipment.shipment_id}: {pii.detail}"
