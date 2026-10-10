"""Independent reviewer: a second, adversarial read of the draft.

Self-verification (``verify.py``) is the pipeline critiquing its own
work. This step is a generator/critic split: a distinct reviewer —
with its own persona prompt in provider mode, its own model when
``REVIEWER_MODEL`` is set — reads the finished draft the way a senior
operations reviewer would: grounding, policy compliance, tone, the
next-update commitment, and whether the claim packet gives the
approver everything needed to decide.

- provider mode — an LLM reviewer call (prompts in ``prompts.py``),
  returning a structured verdict: ``pass`` | ``concerns`` | ``block``
  with findings;
- default/mock mode — a deterministic *second* checklist whose checks
  deliberately differ from self-verification's: the next-update
  commitment, claim-packet completeness, and citation coverage of the
  diagnosis (verification checks the draft text against the facts;
  the reviewer checks the packet the approver actually decides on).

A ``block`` verdict never auto-rejects and never opens or closes the
gate by itself: it flags the result (``reviewer_blocked``), forces the
autonomy recommendation to ineligible with that reason, and is shown
prominently to the approver — the human still decides. Set
``REVIEWER=off`` to disable the step entirely (the trace records that
no review ran).
"""

from __future__ import annotations

from .config import env_str, load_dotenv
from .schemas import DraftOutput, ReviewerResult


def reviewer_enabled() -> bool:
    """``REVIEWER`` env var: on by default; off/0/false/no disables."""
    load_dotenv()
    raw = (env_str("REVIEWER") or "on").strip().lower()
    return raw not in {"off", "0", "false", "no"}


def checklist_review(
    *,
    draft: DraftOutput,
    classification: dict,
    diagnosis: dict | None,
) -> ReviewerResult:
    """Deterministic independent review — packet- and approver-focused.

    Blockers are packet defects that leave the approver unable to
    decide (no diagnosis, no policy citations on an exception case);
    findings are quality gaps the approver should see (no next-update
    commitment, no recommended option, a policy the diagnosis relies
    on that the draft never cites).
    """
    findings: list[str] = []
    blockers: list[str] = []
    packet = draft.claim_packet or {}
    exception = classification.get("exception_type", "")

    if "next update" not in draft.body.lower():
        findings.append(
            "draft does not commit to when the next update will arrive"
        )

    packet_diagnosis = packet.get("diagnosis")
    if not isinstance(packet_diagnosis, dict) or not packet_diagnosis.get("root_cause"):
        blockers.append(
            "claim packet carries no diagnosis — the approver cannot see "
            "why this happened"
        )
    if exception != "none" and not (packet.get("policy_citations") or []):
        blockers.append("claim packet cites no policies for an exception case")
    if not packet.get("recovery_options"):
        findings.append("claim packet proposes no recovery options")
    elif not packet.get("recommended_option_id"):
        findings.append("claim packet names no recommended recovery option")

    draft_citations = set(draft.citations)
    for policy_id in (diagnosis or {}).get("citations", []):
        if policy_id not in draft_citations:
            findings.append(
                f"diagnosis relies on {policy_id} but the draft does not cite it"
            )

    all_findings = blockers + findings
    verdict = "block" if blockers else ("concerns" if findings else "pass")
    return ReviewerResult(verdict=verdict, findings=all_findings, source="checklist")


def review_draft(
    *,
    draft: DraftOutput,
    classification: dict,
    diagnosis: dict | None,
    delay_hours: float | None,
    mismatches: list[dict],
    policies: list[dict],
    backend=None,
) -> ReviewerResult | None:
    """Review the draft — LLM reviewer when available, checklist otherwise.

    Returns ``None`` only when the step is disabled (``REVIEWER=off``).
    An LLM review that fails or replies unusably degrades to the
    checklist with the reason recorded in ``note`` — the same honesty
    rule as self-verification.
    """
    if not reviewer_enabled():
        return None
    checklist = checklist_review(
        draft=draft, classification=classification, diagnosis=diagnosis
    )
    review_fn = getattr(backend, "review_draft", None) if backend is not None else None
    if review_fn is None:
        return checklist
    mismatch_text = "; ".join(
        f"{m.get('field')}: BOL={m.get('bol_value')} vs invoice={m.get('invoice_value')}"
        for m in mismatches
    )
    packet = draft.claim_packet or {}
    context = {
        "shipment_id": packet.get("shipment_id", ""),
        "origin": packet.get("origin", ""),
        "destination": packet.get("destination", ""),
        "exception_type": classification.get("exception_type", ""),
        "severity": classification.get("severity", ""),
        "delay_hours": delay_hours,
        "mismatches": mismatch_text,
        "policies": policies,
        "policy_details": policies,  # key the backends' prompt builders read
        "diagnosis_root_cause": (diagnosis or {}).get("root_cause", ""),
        "diagnosis_citations": (diagnosis or {}).get("citations", []),
        "claim_packet": packet,
        "subject": draft.subject,
        "body": draft.body,
    }
    try:
        verdict = review_fn(context)
    except Exception as exc:  # the review never fails the run
        checklist.note = f"LLM review failed ({exc}) — checklist review used"
        return checklist
    if not verdict:
        checklist.note = "LLM review reply was unusable — checklist review used"
        return checklist
    return ReviewerResult(
        verdict=str(verdict.get("verdict", "concerns")),
        findings=[str(f) for f in verdict.get("findings", [])],
        source="llm",
        model=verdict.get("model"),
    )
