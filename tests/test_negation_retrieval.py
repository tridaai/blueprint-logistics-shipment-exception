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

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.classifier import signal_phrase_states
from shipment_agent.policies_data import POLICIES
from shipment_agent.retriever import (
    KeywordRetriever,
    SemanticRetriever,
    _embedding_query,
    _query_analysis,
)

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


# ---------------------------------------------------------------------------
# The semantic half: denied signals are struck before embedding
# ---------------------------------------------------------------------------
#
# Round 9 taught the keyword ranker polarity and RQ-34 still missed
# in the hybrid ranking: the semantic half embedded the raw query,
# denials included, and a vector cannot hear the "no". The strike
# below is the same classifier discipline applied to the text the
# semantic retriever embeds (retriever._embedding_query).


def test_embedding_query_strikes_denied_signal_phrases():
    struck = _embedding_query(
        "none Terminal inspection complete: no damage, no leak found, "
        "seals intact and undamaged customer update claim packet escalation"
    )
    tokens = struck.split()
    assert "damage" not in tokens
    assert "leak" not in tokens
    # The rest of the query — including the look-alike "undamaged",
    # a different token the classifier never denied — survives.
    assert "undamaged" in tokens
    assert "inspection" in tokens
    assert "escalation" in tokens


def test_embedding_query_leaves_affirmed_queries_byte_identical():
    query = "damage Two cartons crushed, contents leaking at inspection"
    assert _embedding_query(query) == query
    plain = "delay Shipment held at the hub, revised estimate tomorrow"
    assert _embedding_query(plain) == plain


def test_embedding_query_strikes_only_what_is_denied():
    # "crushed" is affirmed in the same breath that denies "leak":
    # the affirmed signal's vocabulary must survive the strike.
    struck = _embedding_query("damage Crushed cartons at the dock, no leak found")
    assert "crushed" in struck.lower().split()
    assert "leak" not in struck.lower().split()


def test_embedding_query_recovery_strikes_the_delay_vocabulary():
    struck = _embedding_query(
        "none Earlier weather delay cleared at hub — back on schedule"
    )
    assert "delay" not in struck.split()


_DIMS = ("damage", "update", "customer", "inspection")


def _count_vec(text: str) -> list[float]:
    lowered = text.lower()
    return [float(lowered.count(word)) for word in _DIMS]


class _CountEmbeddings:
    def create(self, model=None, input=None):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=_count_vec(text))
                for i, text in enumerate(input)
            ]
        )


class _CountOpenAI:
    def __init__(self, **kwargs):
        self.embeddings = _CountEmbeddings()


@pytest.fixture
def count_embeddings(monkeypatch):
    for var in ("MODEL_BACKEND", "OPENAI_BASE_URL", "OPENAI_EMBEDDING_MODEL",
                "DATABASE_URL", "RETRIEVER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "shipment_agent.model_backends.load_dotenv", lambda *a, **k: None
    )
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_CountOpenAI))
    monkeypatch.setitem(sys.modules, "chromadb", None)  # in-memory cosine path
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_semantic_ranking_reads_polarity(count_embeddings):
    # With raw embedding the doubled "damage" dominates the count
    # vector and the damage document wins; struck, the query's
    # remaining vocabulary is the routine document's.
    corpus = [
        {
            "policy_id": "DOC-DAMAGE",
            "title": "Damage documentation",
            "text": "damage damage damage claim packet",
        },
        {
            "policy_id": "DOC-ROUTINE",
            "title": "Routine in-transit update",
            "text": "customer update inspection update",
        },
    ]
    retriever = SemanticRetriever(policies=corpus)
    results = retriever.retrieve(
        "inspection no damage and no damage found, customer update", top_k=2
    )
    assert _ids(results)[0] == "DOC-ROUTINE"
    assert results[-1].policy_id == "DOC-DAMAGE"
    assert results[-1].score == 0.0
