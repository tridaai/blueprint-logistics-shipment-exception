"""Retrieval-query regression tests.

The customer-simulation finding: the retrieve node built its query only
from the exception type + the classifier's rationale, so a customer SOP
written in operational language — the vocabulary of the shipment's own
event text and condition notes — never ranked for the case it described.
The query now carries the shipment's own content. These tests pin that.
"""

from __future__ import annotations

from shipment_agent.graph import run_shipment
from shipment_agent.policies_data import POLICIES
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import load_sample_shipments
from shipment_agent.schemas import ShipmentInput

# A realistic customer SOP in operational language. Its vocabulary comes
# from SYN-1002's event/condition text ("cartons crushed", "contents
# leaking", "terminal inspection") — it shares almost nothing with the
# classifier's rationale ("Condition/event text reports physical damage
# to the shipment"), which is exactly the adaptation-path failure mode.
OPS_SOP = {
    "policy_id": "POL-OPS-77",
    "title": "Crushed cartons at terminal inspection",
    "text": (
        "When cartons arrive crushed with contents leaking at the terminal "
        "inspection point, quarantine the freight at the terminal, photograph "
        "the crushed cartons, and record the leaking contents in the "
        "inspection log before the shipment moves again."
    ),
}


def _sample(shipment_id: str) -> dict:
    return next(s for s in load_sample_shipments() if s["shipment_id"] == shipment_id)


def test_operational_sop_ranks_for_the_damage_case_it_describes():
    retriever = KeywordRetriever(policies=[*POLICIES, OPS_SOP])
    result = run_shipment(
        ShipmentInput.model_validate(_sample("SYN-1002")), retriever=retriever
    )
    assert result.classification.exception_type.value == "damage"
    retrieved_ids = [p.policy_id for p in result.policies]
    assert "POL-OPS-77" in retrieved_ids, retrieved_ids


def test_shipment_content_alone_finds_the_operational_sop():
    """Direct retriever check: a query carrying the shipment's own event
    text retrieves the SOP; the old type+rationale-only query did not
    rank it in the top-3."""
    retriever = KeywordRetriever(policies=[*POLICIES, OPS_SOP])
    sample = _sample("SYN-1002")
    content_query = (
        f"damage {sample['latest_event']} {sample['condition_notes']}"
    )
    top = retriever.retrieve(content_query, top_k=3)
    assert "POL-OPS-77" in [p.policy_id for p in top]
