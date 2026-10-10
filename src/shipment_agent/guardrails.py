"""Guardrails: deterministic validation applied to every draft.

Guardrails here are code, not prompts. Each guardrail is a *named check*
with its own pass/fail, so the demo, the UI, and the human approver see
exactly which check failed and why — never just a red light. A draft that
fails any check must not be approved until it is fixed.
"""

from __future__ import annotations

import re

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

# Time-bound commitments are prohibited promises too (POL-COMM-01) —
# a gap the live LLM-judge pack found: a draft that "guarantees
# notification by end of business tomorrow" passed the phrase list
# above while promising exactly what the policy forbids. Patterns:
# hard time anchors (end of business/day, tomorrow, a weekday) and
# digit-bound windows framed as commitments. Wording matters: the
# sanctioned template phrase "within one business day" uses a number
# WORD, and these patterns require digits, so the house style stays
# clean — verified against every bundled sample and golden draft.
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_TIME_COMMITMENT_RES = (
    re.compile(r"\bby end of business\b"),
    re.compile(r"\bby end of (the )?day\b"),
    re.compile(r"\bby (tomorrow|tonight)\b"),
    re.compile(rf"\bby (next )?({_WEEKDAYS})\b"),
    re.compile(
        r"\b(we will|we'll|will|guarantee\w*|promise\w*|commit\w*)\b[^.]{0,60}?"
        r"\bwithin\s+\d+\s*(business\s+)?(hours?|days?)\b"
    ),
    re.compile(
        r"\bwithin\s+\d+\s*(business\s+)?(hours?|days?)\b[^.]{0,40}?"
        r"\b(guarantee\w*|promise\w*)\b"
    ),
    re.compile(r"\bguarantee\w*\b[^.]{0,80}?\bby\b"),
)


def time_commitment_hits(body: str) -> list[str]:
    """Matched time-commitment fragments in ``body`` (lower-cased)."""
    lowered = body.lower()
    hits: list[str] = []
    for pattern in _TIME_COMMITMENT_RES:
        match = pattern.search(lowered)
        if match:
            hits.append(match.group().strip())
    return hits

# PII patterns a customer draft must never carry. The draft quotes the
# source record (condition notes, carrier text) verbatim — exactly the
# place a stray personal identifier typed by a human would ride along.
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PASSPORT_RE = re.compile(r"\b[A-Z]\d{8,9}\b")
# Candidate card sequences: 13–19 digits, optionally separated by single
# spaces or dashes. A candidate only counts when it is Luhn-valid — a
# long reference number that fails Luhn is not a card number.
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _mask(value: str) -> str:
    """Mask a found identifier before it lands in a trace or result —
    the guardrail report must not become a second copy of the PII."""
    return f"{value[:2]}…{value[-2:]}" if len(value) > 4 else "…"


def pii_hits(body: str) -> list[tuple[str, str]]:
    """(kind, masked value) for every PII-like pattern in ``body``."""
    hits: list[tuple[str, str]] = []
    for match in _SSN_RE.finditer(body):
        hits.append(("SSN-like pattern", _mask(match.group())))
    for match in _CARD_CANDIDATE_RE.finditer(body):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            hits.append(("credit-card-like sequence (Luhn-valid)", _mask(match.group())))
    for match in _PASSPORT_RE.finditer(body):
        hits.append(("passport-like pattern", _mask(match.group())))
    return hits


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
    promise_hits = banned_hits + time_commitment_hits(body)
    checks.append(
        GuardrailCheck(
            name="no_prohibited_promises",
            passed=not promise_hits,
            detail=(
                "No prohibited promise phrases found."
                if not promise_hits
                else "Prohibited promise(s) found: " + ", ".join(f"'{p}'" for p in promise_hits)
            ),
        )
    )

    found_pii = pii_hits(body)
    checks.append(
        GuardrailCheck(
            name="no_pii_in_draft",
            passed=not found_pii,
            detail=(
                "No PII patterns (SSN / card / passport-like) found in the draft."
                if not found_pii
                else "PII found in draft: "
                + "; ".join(f"{kind} '{value}'" for kind, value in found_pii)
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
