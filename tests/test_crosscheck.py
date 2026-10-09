"""Pure resolution-policy tests for the rules-vs-LLM cross-check.

The policy (crosscheck.py): agreement keeps the rule result; disagreement
keeps the rule result UNLESS the rules were weak (type ``none`` or
confidence below 0.85) AND the LLM confidence is at least 0.85 — the one
case where the LLM classification is adopted, always flagged.
"""

from shipment_agent.crosscheck import (
    LLM_ADOPT_CONFIDENCE,
    RULE_WEAK_CONFIDENCE,
    resolve_classification,
)
from shipment_agent.schemas import Classification, ExceptionType, Severity


def _rule(type_: str, severity: str = "high", confidence: float = 0.94) -> Classification:
    return Classification(
        exception_type=ExceptionType(type_),
        severity=Severity(severity),
        confidence=confidence,
        signals=["rule signal"],
        rationale="Rule rationale.",
    )


def _llm(type_: str, severity: str = "high", confidence: float = 0.9) -> dict:
    return {
        "exception_type": type_,
        "severity": severity,
        "confidence": confidence,
        "rationale": "LLM rationale.",
        "backend": "fake",
    }


def test_no_llm_resolves_rules_only():
    final, cc = resolve_classification(_rule("delay"), None, "openai")
    assert final.exception_type == ExceptionType.DELAY
    assert cc.resolution == "rules_only"
    assert cc.adopted_source == "rules"
    assert cc.agrees is None
    assert cc.llm_exception_type is None


def test_agreement_keeps_rule_result_and_notes_severity_difference():
    final, cc = resolve_classification(
        _rule("delay", severity="high"), _llm("delay", severity="medium")
    )
    assert cc.resolution == "agree"
    assert cc.agrees is True
    assert final.severity == Severity.HIGH  # rule severity kept
    assert "Severity differs" in cc.note


def test_disagreement_with_strong_rules_stays_rules_authoritative():
    final, cc = resolve_classification(_rule("damage", confidence=0.95), _llm("delay"))
    assert cc.resolution == "rules_authoritative"
    assert cc.agrees is False
    assert final.exception_type == ExceptionType.DAMAGE


def test_disagreement_none_rules_high_confidence_llm_is_adopted():
    final, cc = resolve_classification(
        _rule("none", severity="low", confidence=0.9), _llm("damage", confidence=0.92)
    )
    assert cc.resolution == "llm_adopted"
    assert cc.adopted_source == "llm"
    assert final.exception_type == ExceptionType.DAMAGE
    assert final.confidence == 0.92
    assert any("llm-adopted" in s for s in final.signals)
    assert "flagged" in final.rationale.lower()


def test_disagreement_low_confidence_rules_can_be_adopted():
    final, cc = resolve_classification(
        _rule("delay", confidence=RULE_WEAK_CONFIDENCE - 0.05),
        _llm("document_mismatch", confidence=LLM_ADOPT_CONFIDENCE),
    )
    assert cc.resolution == "llm_adopted"
    assert final.exception_type == ExceptionType.DOCUMENT_MISMATCH


def test_adoption_requires_llm_confidence_at_the_bar():
    final, cc = resolve_classification(
        _rule("none", severity="low", confidence=0.9),
        _llm("damage", confidence=LLM_ADOPT_CONFIDENCE - 0.01),
    )
    assert cc.resolution == "rules_authoritative"
    assert final.exception_type == ExceptionType.NONE


def test_adoption_never_happens_at_the_weak_boundary():
    # Rule confidence exactly at the bar is NOT weak.
    final, cc = resolve_classification(
        _rule("delay", confidence=RULE_WEAK_CONFIDENCE), _llm("damage", confidence=0.99)
    )
    assert cc.resolution == "rules_authoritative"
    assert final.exception_type == ExceptionType.DELAY
