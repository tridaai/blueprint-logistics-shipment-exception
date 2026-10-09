"""Self-verification: the agent critiques its own draft before guardrails.

Drafting is the one place the pipeline generates language; generation
can drift from the computed facts (a wrong delay figure, a policy cited
that retrieval never returned, a promise nothing supports). This node
checks the draft against the verified facts and cited policies and
produces a structured verdict:

- provider mode — an LLM critique call (prompts in ``prompts.py``),
  parsed into the same verdict shape;
- default/mock mode — a deterministic evidence checklist producing the
  same shape, honestly labelled ``source="checklist"``.

The verdict never rewrites the draft and never blocks by itself — it is
evidence for the approver, input to the bounded repair loop, and a
disqualifier in the autonomy recommendation.
"""

from __future__ import annotations

import re

from .schemas import DraftOutput, VerificationResult

_CITATION_RE = re.compile(r"\[(POL-[A-Z0-9-]+)\]")
_HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hours?\b", re.IGNORECASE)
_SHIPMENT_ID_RE = re.compile(r"\b[A-Z]{2,4}-\d{3,}\b")
_MONEY_RE = re.compile(r"\$\s?\d")


def checklist_verification(
    *,
    draft: DraftOutput,
    delay_hours: float | None,
    policies: list[dict],
    shipment_id: str,
) -> VerificationResult:
    """Deterministic grounding checks over the draft text.

    Every check compares the draft against a fact the pipeline computed:
    cited policy ids must be among the retrieved policies, hour figures
    must equal the computed delay, shipment references must be this
    case, and no monetary figure may appear (the facts carry cost in
    abstract units, never dollars — a dollar figure is invented).
    """
    issues: list[str] = []
    text = f"{draft.subject}\n{draft.body}"
    cited_ids = {p["policy_id"] for p in policies}
    for token in sorted(set(_CITATION_RE.findall(draft.body))):
        if token not in cited_ids:
            issues.append(
                f"draft cites [{token}], which is not among the retrieved policies"
            )
    if delay_hours is not None:
        for figure in _HOURS_RE.findall(text):
            if abs(float(figure) - delay_hours) > 0.01:
                issues.append(
                    f"draft states {figure} hours but the computed delay is "
                    f"{delay_hours} hours"
                )
    for token in sorted(set(_SHIPMENT_ID_RE.findall(text))):
        if token != shipment_id:
            issues.append(
                f"draft references shipment {token} but this case is {shipment_id}"
            )
    if _MONEY_RE.search(text):
        issues.append(
            "draft states a monetary figure, which no verified fact supports "
            "(costs are tracked in abstract units)"
        )
    grounded = not issues
    summary = (
        "Checklist: every citation, figure, and reference in the draft matches "
        "the verified facts."
        if grounded
        else f"Checklist found {len(issues)} grounding issue(s) in the draft."
    )
    return VerificationResult(
        grounded=grounded, issues=issues, source="checklist", summary=summary
    )


def verify_draft(
    *,
    draft: DraftOutput,
    classification: dict,
    delay_hours: float | None,
    mismatches: list[dict],
    policies: list[dict],
    shipment_id: str,
    backend=None,
) -> VerificationResult:
    """Verify the draft — LLM critique when available, checklist otherwise.

    An LLM critique that fails or replies unusably degrades to the
    checklist with the reason recorded in ``note`` — the verdict is
    never silently weaker than it looks.
    """
    checklist = checklist_verification(
        draft=draft,
        delay_hours=delay_hours,
        policies=policies,
        shipment_id=shipment_id,
    )
    verify_fn = getattr(backend, "verify_draft", None) if backend is not None else None
    if verify_fn is None:
        return checklist
    mismatch_text = "; ".join(
        f"{m.get('field')}: BOL={m.get('bol_value')} vs invoice={m.get('invoice_value')}"
        for m in mismatches
    )
    context = {
        "shipment_id": shipment_id,
        "origin": draft.claim_packet.get("origin", ""),
        "destination": draft.claim_packet.get("destination", ""),
        "exception_type": classification.get("exception_type", ""),
        "severity": classification.get("severity", ""),
        "delay_hours": delay_hours,
        "mismatches": mismatch_text,
        "policies": policies,
        "policy_details": policies,  # key the backends' prompt builders read
        "subject": draft.subject,
        "body": draft.body,
    }
    try:
        verdict = verify_fn(context)
    except Exception as exc:  # verification never fails the run
        checklist.note = f"LLM verification failed ({exc}) — checklist verification used"
        return checklist
    if not verdict:
        checklist.note = "LLM verification reply was unusable — checklist verification used"
        return checklist
    return VerificationResult(
        grounded=bool(verdict.get("grounded")),
        issues=[str(i) for i in verdict.get("issues", [])],
        source="llm",
        summary=str(verdict.get("summary", "")),
    )
