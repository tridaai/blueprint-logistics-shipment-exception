from shipment_agent.guardrails import validate_draft
from shipment_agent.schemas import ExceptionType

GOOD_BODY = (
    "Dear Customer, shipment SHP-1 is delayed. "
    "We will send the next update tomorrow. Reference policies: [POL-DELAY-01]"
)


def test_good_draft_passes():
    result = validate_draft(GOOD_BODY, "SHP-1", ExceptionType.DELAY, ["POL-DELAY-01"])
    assert result.passed
    assert result.errors == []


def test_banned_promise_fails():
    body = GOOD_BODY + " We guarantee a full refund."
    result = validate_draft(body, "SHP-1", ExceptionType.DELAY, ["POL-DELAY-01"])
    assert not result.passed
    assert any("prohibited promise" in e for e in result.errors)


def test_missing_shipment_id_fails():
    result = validate_draft(GOOD_BODY, "OTHER-9", ExceptionType.DELAY, ["POL-DELAY-01"])
    assert not result.passed


def test_exception_without_citations_fails():
    result = validate_draft(GOOD_BODY, "SHP-1", ExceptionType.DAMAGE, [])
    assert not result.passed


def test_none_exception_needs_no_citations():
    body = "Dear Customer, shipment SHP-1 is in transit. The next update will follow at the next milestone."
    result = validate_draft(body, "SHP-1", ExceptionType.NONE, [])
    assert result.passed
