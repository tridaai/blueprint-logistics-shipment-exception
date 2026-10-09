"""Recovery options: proposed by a template or an LLM, scored by code.

The split is the point. Proposing *what could be done* is judgement —
a deterministic template in the default mode, the LLM in provider
mode. Scoring *what each option is worth* is arithmetic, and a model
never does it: :func:`score_option` computes ETA improvement, added
cost, and SLA impact from the computed facts (delay hours, severity)
with fixed, inspectable formulas, and the recommended option is simply
the highest score. Every number in the console, the packet, and the
draft traces back to this function.

Score (0-100-ish, clamped at 0):
    0.60 * sla_score * 100
  + 0.40 * min(eta_improvement, 48) / 48 * 100
  - 0.25 * min(added_cost, 200) / 200 * 100
"""

from __future__ import annotations

from .model_backends import DraftContext
from .schemas import RecoveryOption

OPTION_KINDS = (
    "expedite",
    "reroute",
    "partial_reship",
    "reschedule_appointment",
    "correct_documents",
    "wait_and_monitor",
)

_SEVERITY_WEIGHT = {"low": 1, "medium": 2, "high": 3, "critical": 4}

_TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "delay": [
        ("expedite", "Expedite the remaining leg",
         "Upgrade the remaining transport to the fastest available service and re-book the final leg today."),
        ("reroute", "Reroute via the alternate hub",
         "Move the shipment through the alternate regional hub to bypass the current hold."),
        ("wait_and_monitor", "Hold and monitor",
         "Keep the current routing, watch the next carrier scan, and update the customer at the next milestone."),
    ],
    "damage": [
        ("partial_reship", "Ship replacement units for the damaged quantity",
         "Release a replacement shipment for the damaged units while the claim packet is reviewed."),
        ("expedite", "Expedite the replacement shipment",
         "Send the replacement units on the fastest available service once released."),
        ("wait_and_monitor", "Hold pending condition review",
         "Keep the shipment at the terminal until the condition review and photographs are complete."),
    ],
    "document_mismatch": [
        ("correct_documents", "Expedite corrected documents from the shipper",
         "Have the shipper issue corrected documents for the conflicting fields today and re-present them for clearance."),
        ("partial_reship", "Release the undisputed portion now",
         "Deliver the portion of the shipment whose documents agree and hold only the disputed units."),
        ("wait_and_monitor", "Hold pending document correction",
         "Keep the shipment on document hold until the corrected documents arrive."),
    ],
    "missed_appointment": [
        ("reschedule_appointment", "Book the next dock appointment today",
         "Secure the next available delivery appointment with the receiving facility and confirm the window."),
        ("expedite", "Expedite re-delivery",
         "Move the shipment to the front of the re-delivery queue with an upgraded service level."),
        ("wait_and_monitor", "Wait for the standard rebooking",
         "Let the carrier rebook the appointment on its standard cycle and update the customer when confirmed."),
    ],
    "none": [
        ("wait_and_monitor", "Continue as planned",
         "No recovery action is needed; keep the standard tracking updates."),
    ],
}


def score_option(
    kind: str, delay_hours: float | None, severity: str
) -> tuple[float, float, float, float]:
    """Deterministic impact scoring. Returns (eta_h, cost, sla, score)."""
    d = delay_hours or 0.0
    w = _SEVERITY_WEIGHT.get(severity, 2)
    if kind == "expedite":
        eta, cost, sla = min(0.8 * d + 4, 48.0), 60 + 15 * w + 1.2 * d, min(0.55 + 0.10 * w, 0.95)
    elif kind == "reroute":
        eta, cost, sla = min(0.5 * d + 2, 30.0), 25 + 8 * w + 0.6 * d, min(0.45 + 0.10 * w, 0.90)
    elif kind == "partial_reship":
        eta, cost, sla = min(0.4 * d + 6, 36.0), 45 + 12 * w + 0.4 * d, min(0.50 + 0.09 * w, 0.92)
    elif kind == "reschedule_appointment":
        eta, cost, sla = 4 + 2 * w, 10 + 3 * w, min(0.50 + 0.08 * w, 0.90)
    elif kind == "correct_documents":
        eta, cost, sla = 6 + 2 * w, 8 + 2 * w, min(0.60 + 0.07 * w, 0.95)
    else:  # wait_and_monitor
        eta, cost, sla = 0.0, 0.0, max(0.05, 0.55 - 0.008 * d - 0.05 * w)
    score = (
        0.60 * sla * 100
        + 0.40 * (min(eta, 48.0) / 48.0 * 100)
        - 0.25 * (min(cost, 200.0) / 200.0 * 100)
    )
    return round(eta, 2), round(cost, 2), round(sla, 4), round(max(score, 0.0), 2)


def _template_proposals(exception_type: str) -> list[dict]:
    return [
        {"kind": kind, "title": title, "description": description}
        for kind, title, description in _TEMPLATES.get(exception_type, _TEMPLATES["none"])
    ]


def build_recovery_options(
    *,
    exception_type: str,
    severity: str,
    delay_hours: float | None,
    backend=None,
    context: DraftContext | None = None,
    notes: list[str] | None = None,
) -> list[RecoveryOption]:
    """Propose (template or LLM), validate kinds, score in code, recommend.

    When ``notes`` is given, a provider failure that degraded this node
    to template proposals is appended to it — degradation is recorded,
    never silent.
    """
    proposals: list[dict] | None = None
    propose_fn = getattr(backend, "propose_options", None) if backend is not None else None
    if propose_fn is not None and context is not None:
        try:
            raw = propose_fn(context)
        except Exception as exc:  # proposals never fail the run
            raw = None
            if notes is not None:
                notes.append(
                    f"LLM option proposals failed ({exc}) — template proposals used"
                )
        if raw:
            seen: set[str] = set()
            valid: list[dict] = []
            for entry in raw:
                kind = entry.get("kind")
                if kind in OPTION_KINDS and kind not in seen:
                    seen.add(kind)
                    valid.append(entry)
            if len(valid) >= 2:
                proposals = valid[:3]
    if proposals is None:
        proposals = _template_proposals(exception_type)

    options: list[RecoveryOption] = []
    for i, proposal in enumerate(proposals):
        eta, cost, sla, score = score_option(proposal["kind"], delay_hours, severity)
        options.append(
            RecoveryOption(
                option_id=f"OPT-{i + 1}",
                kind=proposal["kind"],
                title=proposal["title"],
                description=proposal["description"],
                eta_improvement_hours=eta,
                added_cost_units=cost,
                sla_score=sla,
                score=score,
            )
        )
    if options:
        best = max(range(len(options)), key=lambda i: (options[i].score, -i))
        options[best].recommended = True
    return options
