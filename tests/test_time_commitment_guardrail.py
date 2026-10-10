"""Time-commitment guardrail tests.

Found by the live LLM-judge pack: a real model draft for a
missed-appointment case passed every guardrail while guaranteeing
notification "by end of business tomorrow" — a prohibited promise
the phrase list did not cover. `no_prohibited_promises` now also
catches time-bound commitments: hard anchors (end of business,
tomorrow, a weekday) and digit-bound windows framed as commitments.
The house template's sanctioned wording ("within one business day",
a number word) must stay clean, as must every bundled draft.
"""

from __future__ import annotations

from shipment_agent.graph import run_shipment
from shipment_agent.guardrails import guardrail_checks, time_commitment_hits
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import sample_shipment_models
from shipment_agent.schemas import ExceptionType

BASE_BODY = (
    "Dear Synthetic Customer,\n\n"
    "Your shipment TIME-1 missed its delivery appointment. "
    "Reference policies: [POL-APPT-01]. We are securing the next "
    "available appointment. The next update will confirm the window."
)


def _promise_check(body: str):
    checks = guardrail_checks(body, "TIME-1", ExceptionType.MISSED_APPOINTMENT, ["POL-APPT-01"])
    return next(c for c in checks if c.name == "no_prohibited_promises")


def test_judge_flagged_sentence_now_fails():
    # The exact shape the live judge flagged.
    check = _promise_check(BASE_BODY + " We will notify you by end of business tomorrow.")
    assert check.passed is False
    assert "by end of business" in check.detail


def test_guaranteed_by_weekday_fails():
    check = _promise_check(BASE_BODY + " Delivery is guaranteed by Friday.")
    assert check.passed is False


def test_we_will_deliver_within_digit_hours_fails():
    check = _promise_check(BASE_BODY + " We will deliver within 24 hours.")
    assert check.passed is False


def test_by_tomorrow_and_by_next_weekday_fail():
    assert _promise_check(BASE_BODY + " Expect it by tomorrow.").passed is False
    assert _promise_check(BASE_BODY + " We will update you by next Monday.").passed is False


def test_noncommittal_update_language_passes():
    check = _promise_check(BASE_BODY + " We will update you when the carrier confirms.")
    assert check.passed is True
    assert time_commitment_hits(BASE_BODY) == []


def test_sanctioned_template_wording_stays_clean():
    # Number words, not digits: the house style the mock template uses.
    body = BASE_BODY + " We will send the next update within one business day."
    assert _promise_check(body).passed is True


def test_no_sample_draft_trips_the_time_patterns():
    # Zero false positives across the bundled set (mock mode), on top
    # of the eval set exercised by run_evals.
    for shipment in sample_shipment_models():
        result = run_shipment(shipment, backend=MockModelBackend(), retriever=KeywordRetriever())
        promise = next(
            c for c in result.validation.checks if c.name == "no_prohibited_promises"
        )
        if shipment.shipment_id == "SYN-1013":
            assert promise.passed is False  # its pre-existing refund-promise failure
            assert "full refund" in promise.detail
        else:
            assert promise.passed, f"{shipment.shipment_id}: {promise.detail}"
