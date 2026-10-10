"""Autonomy recommendation: deterministic routing policy, in code.

Every result carries a recommendation for how the case should be
routed: could this class of case be auto-approved, or must a human
decide? The policy is deliberately conservative and entirely
deterministic — no model judgement:

eligible only when ALL of these hold:

- the exception is ``none`` or the severity is ``low``;
- the guardrails passed;
- the rules/LLM cross-check did not disagree (when it ran);
- no repair redraft was needed;
- the independent reviewer did not block the draft (when it ran).

It is a RECOMMENDATION, printed on the result, the console, and the
claim packet. It never acts: the human-approval gate is unchanged, and
nothing in the pipeline reads this value to skip it. Its purpose is to
show a customer where the autonomy dial could sit for their operation,
case by case, with the reasons attached.
"""

from __future__ import annotations

from .schemas import AutonomyRecommendation

_DISAGREEING_RESOLUTIONS = {"rules_authoritative", "llm_adopted"}


def compute_autonomy(
    *,
    classification: dict,
    validation: dict,
    cross_check: dict | None,
    repair_attempted: bool,
    reviewer_blocked: bool = False,
) -> AutonomyRecommendation:
    reasons: list[str] = []
    eligible = True

    exception_type = classification.get("exception_type", "")
    severity = classification.get("severity", "")
    if exception_type == "none" or severity == "low":
        reasons.append(
            f"exception is '{exception_type}' with severity '{severity}' "
            "(auto-approval band: none or low)"
        )
    else:
        eligible = False
        reasons.append(
            f"exception '{exception_type}' at severity '{severity}' is above "
            "the auto-approval band (none or low) — a human should decide"
        )

    if validation.get("passed"):
        reasons.append("guardrails passed")
    else:
        eligible = False
        reasons.append("guardrails did not pass — a human must decide")

    if cross_check is None:
        reasons.append("no cross-check ran (default mode) — nothing disagreed")
    elif cross_check.get("resolution") in _DISAGREEING_RESOLUTIONS:
        eligible = False
        reasons.append(
            "the rules and the LLM disagreed on classification "
            f"(resolution: {cross_check.get('resolution')}) — a human should decide"
        )
    else:
        reasons.append(
            f"cross-check resolution: {cross_check.get('resolution')} — no disagreement"
        )

    if repair_attempted:
        eligible = False
        reasons.append("the draft needed a repair redraft — a human should review it")
    else:
        reasons.append("no repair was needed")

    if reviewer_blocked:
        eligible = False
        reasons.append(
            "the independent reviewer blocked this draft — a human must decide"
        )
    else:
        reasons.append("the independent reviewer did not block the draft")

    return AutonomyRecommendation(
        eligible_for_auto_approval=eligible, reasons=reasons
    )
