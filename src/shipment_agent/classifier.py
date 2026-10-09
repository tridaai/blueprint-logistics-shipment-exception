"""Deterministic exception classification.

Why rules first, model second? In a production FDE deployment, exception
classification drives customer communication and claims. A deterministic
classifier is auditable, testable against a golden dataset, and behaves
identically on every run and in production. An LLM backend can be layered on for
ambiguous free-text cases (see ``model_backends.py``), but it should never
silently override computed facts such as a BOL/invoice mismatch.

Robustness rules that matter when real humans type the input:

- **Word boundaries.** Signals are matched as whole words/phrases, so
  "undamaged" never fires the damage rule and "delays" in a quoted policy
  title don't count either.
- **Negation.** A signal inside a negated clause does not fire: "no damage
  reported", "inspected — not damaged", "no discrepancy found", and
  "damage: none" all classify on their *other* merits. Negation cues are
  only honoured within the same clause and a short window before the
  signal, so "no damage, but delayed two days" still classifies as delay.
- **Recovery.** "Back on schedule" / "delay resolved" cancels a
  keyword-only delay signal — but never a *computed* delay: if the ETA is
  still >= the threshold past schedule, the shipment is still delayed.

Priority order matters and is deliberate:
  damage > missed appointment > document mismatch > delay > none
Damage and missed appointments are the most time-sensitive and the most
specific signals; a generic "late" keyword must not swallow them.
"""

from __future__ import annotations

import re

from .schemas import Classification, DocumentMismatch, ExceptionType, Severity, ShipmentInput
from .tools import combined_event_text, compute_delay_hours

_DAMAGE_WORDS = (
    "damaged", "damage", "crushed", "leaking", "leak", "broken", "torn",
    "dented", "water damage", "shattered", "destroyed", "punctured",
    "soaked", "cracked", "bent", "spilled", "spillage", "contaminated",
    "contamination", "waterlogged",
)
_SEVERE_DAMAGE_WORDS = ("crushed", "leaking", "shattered", "destroyed", "contaminated")
_PARTIAL_DAMAGE_WORDS = ("minor", "partial", "partially", "slight", "slightly", "cosmetic", "superficial")
_APPOINTMENT_PHRASES = (
    "missed appointment",
    "appointment missed",
    "missed delivery appointment",
    "appointment rescheduled",
    "refused at dock",
    "refused delivery",
    "delivery refused",
    "turned away",
    "appointment no-show",
    "appointment no show",
    "no-show at dock",
    "receiver closed",
    "dock closed",
)
_MISMATCH_PHRASES = (
    "mismatch", "discrepancy", "short shipment", "short-shipped", "short shipped",
    "quantity discrepancy", "quantity mismatch", "qty mismatch",
    "weight discrepancy", "does not match", "wrong quantity", "wrong sku",
    "overage", "invoice discrepancy",
)
_DELAY_WORDS = (
    "delayed", "delay", "late", "held", "weather hold", "mechanical",
    "detention", "behind schedule", "running late", "running behind",
    "stuck at", "congestion", "customs hold", "held at customs",
    "eta pushed", "pushed back",
)
_DELAY_THRESHOLD_HOURS = 4.0

# Phrases that mean an earlier delay no longer applies.
_RECOVERY_PHRASES = (
    "back on schedule",
    "now on schedule",
    "delay resolved",
    "delay cleared",
    "recovered the delay",
    "caught up",
    "made up the time",
    "recovered and on schedule",
)

_NEGATION_CUE_RE = re.compile(
    r"\b(no|not|never|none|without|nil|free of|absence of)\b|n't"
)
_NEGATION_WINDOW = 40  # characters before a signal in which a cue suppresses it
_CLAUSE_SPLIT = re.compile(r"[,.;:!?()\n]| — | -- ")
_FIELD_NONE = re.compile(
    r"\b(damage|delay|discrepancy|mismatch|leak|leaking|damage reported)\s*[:\-]\s*(none|no|nil|n/a)\b"
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b")


def _clauses(text: str) -> list[tuple[int, str]]:
    """Split text into (start_offset, clause) pairs on clause punctuation."""
    clauses: list[tuple[int, str]] = []
    start = 0
    for match in _CLAUSE_SPLIT.finditer(text):
        clauses.append((start, text[start:match.start()]))
        start = match.end()
    clauses.append((start, text[start:]))
    return clauses


def _is_negated(text: str, match_start: int) -> bool:
    """True when a negation cue governs the signal at ``match_start``.

    Honoured only inside the same clause (punctuation-separated) and a
    short window before the signal, or via a "field: none" construction.
    """
    field_match = _FIELD_NONE.search(text)
    if field_match and abs(field_match.start() - match_start) < 30:
        return True
    clause_start = 0
    for sep in _CLAUSE_SPLIT.finditer(text, 0, match_start):
        clause_start = sep.end()
    window = text[max(clause_start, match_start - _NEGATION_WINDOW):match_start]
    # Word-boundary match, so "noted"/"north" never read as the cue "no".
    return bool(_NEGATION_CUE_RE.search(window))


def _active_hits(text: str, phrases: tuple[str, ...]) -> list[str]:
    """Phrases present in text as whole words and NOT negated."""
    hits: list[str] = []
    for phrase in phrases:
        for match in _phrase_pattern(phrase).finditer(text):
            if not _is_negated(text, match.start()):
                hits.append(phrase)
                break
    return hits


def _severity_for_delay(delay_hours: float | None) -> Severity:
    if delay_hours is None:
        return Severity.MEDIUM
    if delay_hours >= 48:
        return Severity.CRITICAL
    if delay_hours >= 24:
        return Severity.HIGH
    if delay_hours >= 8:
        return Severity.MEDIUM
    return Severity.LOW


def classify_shipment(
    shipment: ShipmentInput, mismatches: list[DocumentMismatch]
) -> Classification:
    """Classify one shipment into an exception type + severity."""
    text = combined_event_text(shipment.latest_event, shipment.condition_notes, shipment.status)
    delay_hours = compute_delay_hours(shipment.scheduled_delivery, shipment.estimated_delivery)
    recovered = any(phrase in text for phrase in _RECOVERY_PHRASES)

    # 1 — Damage (most time-sensitive: photos, claim windows).
    damage_hits = _active_hits(text, _DAMAGE_WORDS)
    if damage_hits:
        partial = any(w in text for w in _PARTIAL_DAMAGE_WORDS)
        severe = any(w in text for w in _SEVERE_DAMAGE_WORDS)
        severity = Severity.MEDIUM if partial else (Severity.HIGH if severe else Severity.MEDIUM)
        return Classification(
            exception_type=ExceptionType.DAMAGE,
            severity=severity,
            confidence=0.95,
            signals=[f"damage keyword: {w}" for w in damage_hits]
            + (["damage qualified as partial/minor"] if partial else []),
            rationale="Condition/event text reports physical damage to the shipment.",
        )

    # 2 — Missed appointment (specific phrase beats the generic 'late' signal).
    appt_hits = _active_hits(text, _APPOINTMENT_PHRASES)
    if appt_hits:
        return Classification(
            exception_type=ExceptionType.MISSED_APPOINTMENT,
            severity=Severity.MEDIUM,
            confidence=0.93,
            signals=[f"appointment signal: {p}" for p in appt_hits],
            rationale="Delivery appointment was missed, refused, or rescheduled.",
        )

    # 3 — Document mismatch (computed fact beats free-text suspicion).
    mismatch_phrases = _active_hits(text, _MISMATCH_PHRASES)
    if mismatches or mismatch_phrases:
        signals = [f"field mismatch: {m.field}" for m in mismatches] + [
            f"mismatch phrase: {p}" for p in mismatch_phrases
        ]
        return Classification(
            exception_type=ExceptionType.DOCUMENT_MISMATCH,
            severity=Severity.HIGH if len(mismatches) >= 2 else Severity.MEDIUM,
            confidence=0.97 if mismatches else 0.82,
            signals=signals,
            rationale="Shipping documents disagree, or the latest event reports a document discrepancy.",
        )

    # 4 — Delay (structured hours first, keywords as backup). A recovery
    # phrase cancels the keyword signal only — never the computed hours.
    delay_hits = [] if recovered else _active_hits(text, _DELAY_WORDS)
    computed_delay = delay_hours is not None and delay_hours >= _DELAY_THRESHOLD_HOURS
    if computed_delay or delay_hits:
        signals = delay_hits[:]
        if delay_hours is not None:
            signals.insert(0, f"delay_hours={delay_hours}")
        return Classification(
            exception_type=ExceptionType.DELAY,
            severity=_severity_for_delay(delay_hours),
            confidence=0.94 if delay_hours is not None else 0.8,
            signals=[f"delay signal: {s}" for s in signals],
            rationale="Estimated delivery is materially later than scheduled, or the carrier reports a delay.",
        )

    # 5 — No exception.
    return Classification(
        exception_type=ExceptionType.NONE,
        severity=Severity.LOW,
        confidence=0.9,
        signals=["recovered: back on schedule"] if recovered else [],
        rationale="No exception signals found; shipment appears to be progressing normally.",
    )
