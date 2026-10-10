"""Negation-aware keyword retrieval: a denied signal is evidence of
absence, not evidence for the exception it names.

The labelled relevance set named the weakness (RQ-17/21/34 sat at
recall 0): a clean shipment's text says "no damage reported", and
plain token overlap ranked the *damage* policy first, because the
negated noun counted at full weight. The keyword ranker now reads
negation with the classifier's own rules (``signal_phrase_states``
— no second vocabulary): denied-signal tokens do not score for the
documents affirming them, a document affirming a denied signal
loses a point per denial (conflict), and a document affirming no
exception signal at all gains a point per denial (the absence
credit — denial is evidence for the routine-update policy). These
tests pin the mechanics on small synthetic corpora; the labelled
set itself is the acceptance gate (``test_retrieval_evals.py``).
"""

from __future__ import annotations

from shipment_agent.classifier import signal_phrase_states
from shipment_agent.policies_data import POLICIES
from shipment_agent.retriever import KeywordRetriever, _query_analysis

DAMAGE_DOC = {
    "policy_id": "DOC-DAMAGE",
    "title": "Damage documentation",
    "text": "For damaged freight, photograph the damage and open the claim packet.",
}
ROUTINE_DOC = {
    "policy_id": "DOC-ROUTINE",
    "title": "Routine in-transit update",
    "text": "For shipments progressing normally, send a brief status update "
    "confirming the unchanged estimated delivery time.",
}
DELAY_DOC = {
    "policy_id": "DOC-DELAY",
    "title": "Delay notification",
    "text": "When delivery slips, notify the customer with the revised estimate.",
}
CORPUS = [DAMAGE_DOC, ROUTINE_DOC, DELAY_DOC]


def _ids(results):
    return [r.policy_id for r in results]


def test_signal_states_split_active_from_denied():
    active, denied = signal_phrase_states("Cartons crushed, no leak found")
    assert "crushed" in active
    assert "leak" in denied
    # A recovery phrase inactivates the delay family, as in the classifier.
    active, denied = signal_phrase_states(
        "Earlier weather delay cleared at hub — back on schedule"
    )
    assert "delay" in denied
    assert "delay" not in active


def test_denied_damage_ranks_the_routine_document_first():
    retriever = KeywordRetriever(CORPUS)
    results = retriever.retrieve(
        "none Delivered on time, signed by receiver. No damage reported; "
        "packaging intact on delivery"
    )
    assert _ids(results)[0] == "DOC-ROUTINE"
    # The damage document affirms the denied signal: its remaining
    # overlap (delivery) is cancelled by the conflict, so it drops
    # out entirely rather than ranking on the denial's coattails.
    assert "DOC-DAMAGE" not in _ids(results)


def test_affirmed_damage_still_ranks_the_damage_document_first():
    retriever = KeywordRetriever(CORPUS)
    results = retriever.retrieve(
        "damage Two cartons crushed, contents leaking at inspection"
    )
    assert _ids(results)[0] == "DOC-DAMAGE"


def test_recovered_delay_is_absence_evidence_not_a_delay_match():
    retriever = KeywordRetriever(CORPUS)
    results = retriever.retrieve(
        "none Earlier weather delay cleared at hub — back on schedule"
    )
    assert _ids(results)[0] == "DOC-ROUTINE"


def test_query_analysis_denies_only_what_the_text_denies():
    base, denied = _query_analysis(
        "damage Crushed cartons at the dock, no leak found, seals intact"
    )
    assert "leak" in denied
    assert "crushed" not in denied
    # The denied signal's tokens leave the base set; affirmed
    # content stays.
    assert "leak" not in base
    assert "crushed" in base


def test_leading_none_label_does_not_negate_the_shipments_own_clause():
    # Without the cue-view rule, the label "none" (itself a cue
    # word) would negate "on time" in the shipment's first clause
    # and the routine document would lose its own vocabulary.
    base, denied = _query_analysis("none Delivered on time, signed by receiver")
    assert "time" in base
    assert denied == set()


def test_absence_credit_goes_only_to_signal_free_documents():
    # POL-COMM-01 affirms a signal ("claim or delay review"), so a
    # denied-damage query must not credit it; POL-NONE-01 affirms
    # none and earns the credit on the real corpus.
    retriever = KeywordRetriever(POLICIES)
    results = retriever.retrieve(
        "none Terminal inspection complete: no damage, no leak found, "
        "seals intact and undamaged"
    )
    assert "POL-NONE-01" in _ids(results)
