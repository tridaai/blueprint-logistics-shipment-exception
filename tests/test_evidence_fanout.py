"""Evidence fan-out equivalence: parallel schedule, identical results.

The evidence phase fans out in the default topology (extract ∥
ingest; retrieval ∥ extraction cross-check ∥ history evidence after
classification) and chains linearly in ``evidence_mode="sequential"``.
The node functions and the merge are the same code — this test pins
that the RESULTS are the same too, field by field, for every bundled
sample, with and without stored history. Only timing differs, so
telemetry and trace durations are excluded from the comparison.
"""

from __future__ import annotations

import pytest

from shipment_agent.graph import build_graph, run_shipment
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import sample_shipment_models

PRIORS = [
    {
        "shipment_id": "OLD-2", "consignee": "Synthetic Retail Co",
        "carrier": "Synthetic Carrier", "origin": "Memphis, TN",
        "destination": "Charlotte, NC", "lane": "Memphis, TN -> Charlotte, NC",
        "exception_type": "damage", "severity": "high",
    },
    {
        "shipment_id": "OLD-1", "consignee": "Synthetic Retail Co",
        "carrier": "Synthetic Carrier", "origin": "Memphis, TN",
        "destination": "Charlotte, NC", "lane": "Memphis, TN -> Charlotte, NC",
        "exception_type": "delay", "severity": "medium",
    },
]

HISTORY = {
    "consignee": "Synthetic Retail Co",
    "consignee_count": 2,
    "consignee_recent_types": ["damage", "delay"],
    "lane": "Memphis, TN -> Charlotte, NC",
    "lane_count": 2,
    "lane_recent_types": ["damage", "delay"],
    "carrier": "Synthetic Carrier",
    "carrier_count": 2,
    "carrier_exception_count": 2,
    "carrier_type_counts": {"damage": 1, "delay": 1},
    "feedback": [
        {"decision": "rejected", "reason": "draft promised a call we cannot staff", "match": "lane"}
    ],
}


def _comparable(result) -> dict:
    dump = result.model_dump(mode="json", exclude={"telemetry"})
    # The claim packet carries a copy of the telemetry — timing too.
    dump["draft"]["claim_packet"]["telemetry"] = None
    for step in dump["trace"]:
        step["duration_ms"] = None  # the one thing allowed to differ
    return dump


def _run_pair(shipment, **kwargs):
    parallel = run_shipment(
        shipment,
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        evidence_mode="parallel",
        **kwargs,
    )
    sequential = run_shipment(
        shipment,
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        evidence_mode="sequential",
        **kwargs,
    )
    return _comparable(parallel), _comparable(sequential)


@pytest.mark.parametrize("index", range(len(sample_shipment_models())))
def test_parallel_matches_sequential_for_every_sample(index):
    shipment = sample_shipment_models()[index]
    parallel, sequential = _run_pair(shipment)
    assert parallel == sequential, (
        f"{shipment.shipment_id}: parallel and sequential evidence "
        "modes produced different results"
    )


def test_parallel_matches_sequential_with_history_and_priors():
    shipment = sample_shipment_models()[0]  # SYN-1001, matches the history above
    parallel, sequential = _run_pair(shipment, history=HISTORY, priors=PRIORS)
    assert parallel == sequential
    # And the history evidence genuinely flowed through the fan-out:
    evidence = parallel["diagnosis"]["evidence"]
    assert any(line.startswith("memory:") for line in evidence)
    assert any(line.startswith("reviewer feedback:") for line in evidence)


def test_parallel_matches_sequential_with_extraction_discrepancies():
    """The cross-check branch contributes too: with an extracting
    backend whose values disagree with the provided fields, the
    discrepancy lines reach the diagnosis identically in both modes."""

    class ExtractingBackend(MockModelBackend):
        name = "fake-extract"

        def extract_document_fields(self, doc_type, document_id, raw_text, fields):
            return {
                "quantity_units": {"value": "999", "confidence": 0.9},
                "weight_kg": {"value": "840", "confidence": 0.9},
            }

    shipment = sample_shipment_models()[0]
    kwargs = dict(history=HISTORY, priors=PRIORS)
    parallel = run_shipment(
        shipment, backend=ExtractingBackend(), retriever=KeywordRetriever(),
        evidence_mode="parallel", **kwargs,
    )
    sequential = run_shipment(
        shipment, backend=ExtractingBackend(), retriever=KeywordRetriever(),
        evidence_mode="sequential", **kwargs,
    )
    assert _comparable(parallel) == _comparable(sequential)
    evidence = parallel.diagnosis.evidence
    assert any(line.startswith("extraction cross-check:") for line in evidence)


def test_unknown_evidence_mode_is_rejected():
    with pytest.raises(ValueError, match="evidence_mode"):
        build_graph(evidence_mode="sideways")
