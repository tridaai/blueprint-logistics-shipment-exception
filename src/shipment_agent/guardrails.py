"""Guardrails: deterministic validation applied to every draft.

Guardrails here are code, not prompts. Each guardrail is a *named check*
with its own pass/fail, so the demo, the UI, and the human approver see
exactly which check failed and why — never just a red light. A draft that
fails any check must not be approved until it is fixed.
"""

from __future__ import annotations

from .schemas import ExceptionType, GuardrailCheck, ValidationResult

# Phrases a customer draft must never contain (POL-COMM-01).
BANNED_PHRASES = (
    "we guarantee",
    "guaranteed refund",
    "full refund",
    "we will compensate",
    "compensation is approved",
    "100% guaranteed",
)


def guardrail_checks(
    body: str,
    shipment_id: str,
    exception_type: ExceptionType,
    citations: list[str],
) -> list[GuardrailCheck]:
    """Run every guardrail and return the named checks, in order."""
    lowered = body.lower()
    checks: list[GuardrailCheck] = []

    id_ok = shipment_id in body
    checks.append(
        GuardrailCheck(
            name="references_shipment_id",
            passed=id_ok,
            detail=f"Draft references shipment {shipment_id}." if id_ok else "Draft does not reference the shipment ID.",
        )
    )

    banned_hits = [phrase for phrase in BANNED_PHRASES if phrase in lowered]
    checks.append(
        GuardrailCheck(
            name="no_prohibited_promises",
            passed=not banned_hits,
            detail=(
                "No prohibited promise phrases found."
                if not banned_hits
                else "Prohibited promise(s) found: " + ", ".join(f"'{p}'" for p in banned_hits)
            ),
        )
    )

    citations_needed = exception_type != ExceptionType.NONE
    citations_ok = bool(citations) or not citations_needed
    checks.append(
        GuardrailCheck(
            name="policy_citations_present",
            passed=citations_ok,
            detail=(
                f"Cites {len(citations)} policy citation(s): {', '.join(citations)}."
                if citations
                else (
                    "No citations required for a no-exception update."
                    if not citations_needed
                    else "Exception draft has no policy citations."
                )
            ),
        )
    )

    next_step_ok = "next update" in lowered or "next step" in lowered
    checks.append(
        GuardrailCheck(
            name="states_next_step",
            passed=next_step_ok,
            detail=(
                "Draft states the next update / next step."
                if next_step_ok
                else "Draft does not state a next update / next step."
            ),
            blocking=False,
        )
    )
    return checks


def validate_draft(
    body: str,
    shipment_id: str,
    exception_type: ExceptionType,
    citations: list[str],
) -> ValidationResult:
    checks = guardrail_checks(body, shipment_id, exception_type, citations)
    errors = [c.detail for c in checks if c.blocking and not c.passed]
    # Keep the historical error wording for the promise check.
    errors = [
        e.replace("Prohibited promise(s) found: ", "Draft contains a prohibited promise: ")
        if e.startswith("Prohibited promise")
        else e
        for e in errors
    ]
    warnings = [c.detail for c in checks if not c.blocking and not c.passed]
    if len(body) < 50:
        warnings.append("Draft is unusually short.")
    return ValidationResult(passed=not errors, errors=errors, warnings=warnings, checks=checks)
