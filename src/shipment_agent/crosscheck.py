"""Rules-vs-LLM classification cross-check — the resolution policy.

Both paths classify the same shipment independently:

- the deterministic rule classifier (``classifier.py``), and
- in provider mode, the LLM, which is never shown the rule result.

This module resolves the pair into the single classification the rest
of the pipeline uses. The policy is explicit and deliberately narrow:

1. **Agreement** (same exception type): the rule result stands —
   identical conclusion, and the rule path is the auditable one.
   A severity difference is noted but does not change the outcome.
2. **Disagreement, rules strong**: the rule result stands
   (``rules_authoritative``). The disagreement is recorded in state and
   surfaced in the trace and console for the human approver.
3. **Disagreement, rules weak, LLM confident**: when the rules landed
   on ``none`` or below ``RULE_WEAK_CONFIDENCE`` AND the LLM confidence
   is at least ``LLM_ADOPT_CONFIDENCE``, the LLM classification is
   adopted (``llm_adopted``) — this is the case rules are known to
   miss: an exception described in phrasing no rule covers. The adopted
   result is flagged in its signals and rationale so the approver can
   see exactly why the classification is not the rule output.

Everything else — no LLM configured (default mode), a provider error,
an unusable reply — resolves to the rule result alone (``rules_only``).
Classification adoption never touches computed facts: delay hours and
document mismatches come from ``tools.py`` regardless of which path
wins the type.
"""

from __future__ import annotations

from .schemas import Classification, ClassificationCrossCheck, ExceptionType, Severity

RULE_WEAK_CONFIDENCE = 0.85
LLM_ADOPT_CONFIDENCE = 0.85


def resolve_classification(
    rule: Classification, llm: dict | None, backend_name: str = ""
) -> tuple[Classification, ClassificationCrossCheck]:
    """Apply the resolution policy to a (rule, LLM) classification pair."""
    base = {
        "rule_exception_type": rule.exception_type.value,
        "rule_severity": rule.severity.value,
        "rule_confidence": rule.confidence,
        "llm_backend": backend_name,
    }
    if llm is None:
        return rule, ClassificationCrossCheck(
            **base,
            resolution="rules_only",
            adopted_source="rules",
            note="No LLM classification available; the rule result stands alone.",
        )

    llm_fields = {
        "llm_exception_type": llm["exception_type"],
        "llm_severity": llm["severity"],
        "llm_confidence": llm["confidence"],
    }
    if llm["exception_type"] == rule.exception_type.value:
        note = "Both paths classified this shipment the same way; the rule result stands."
        if llm["severity"] != rule.severity.value:
            note += (
                f" Severity differs (rules: {rule.severity.value}, LLM: "
                f"{llm['severity']}) — the rule severity is kept."
            )
        return rule, ClassificationCrossCheck(
            **base, **llm_fields, agrees=True, resolution="agree",
            adopted_source="rules", note=note,
        )

    rules_weak = (
        rule.exception_type == ExceptionType.NONE
        or rule.confidence < RULE_WEAK_CONFIDENCE
    )
    if rules_weak and llm["confidence"] >= LLM_ADOPT_CONFIDENCE:
        final = Classification(
            exception_type=ExceptionType(llm["exception_type"]),
            severity=Severity(llm["severity"]),
            confidence=llm["confidence"],
            signals=list(rule.signals) + [
                f"llm-adopted: rules gave '{rule.exception_type.value}' at confidence "
                f"{rule.confidence}; the LLM cross-check classified "
                f"'{llm['exception_type']}' at confidence {llm['confidence']} — "
                "flagged for the approver"
            ],
            rationale=(
                f"{llm['rationale']} (Adopted from the LLM cross-check: the "
                f"deterministic rules produced '{rule.exception_type.value}' at "
                f"confidence {rule.confidence}, so this adoption is flagged for "
                "the human approver.)"
            ),
        )
        return final, ClassificationCrossCheck(
            **base, **llm_fields, agrees=False, resolution="llm_adopted",
            adopted_source="llm",
            note=(
                "Disagreement: the rules were weak (none or low confidence) and "
                "the LLM was highly confident, so the LLM classification was "
                "adopted — flagged for the approver."
            ),
        )

    return rule, ClassificationCrossCheck(
        **base, **llm_fields, agrees=False, resolution="rules_authoritative",
        adopted_source="rules",
        note=(
            "Disagreement recorded for the approver; the deterministic rule "
            "result stays authoritative."
        ),
    )
