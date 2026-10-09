"""Web UI, samples endpoint, reject flow, and demo smoke tests."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from shipment_agent.api import app
from shipment_agent.demo import print_trace
from shipment_agent.samples import load_sample_shipments

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_index_page_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Shipment Exception Agent" in response.text
    assert "#4F46E5" in response.text  # Trida indigo accent


def test_index_is_demo_console_with_brand():
    html = client.get("/").text
    # Brand: Trida logo + favicon wired up.
    assert "/static/brand/tridaai-logo.png" in html
    assert 'href="/favicon.svg"' in html
    # Demo console panels: pipeline trace, architecture, evals, decision log.
    assert 'id="trace"' in html
    assert 'id="evals-body"' in html
    assert 'id="decision-log"' in html
    assert "Architecture" in html
    # Light-maintenance rules: no external assets, no webfonts, system stack.
    assert "fonts.googleapis" not in html
    assert "@font-face" not in html
    assert "system-ui" in html
    # Architecture panel is the diagram + one line per step + a pointer to
    # the architecture doc — the long per-layer prose lives in docs/ only.
    assert "docs/architecture.md" in html
    assert "Hallucination class eliminated" not in html
    assert "the graph ends here by design" not in html


def test_brand_assets_served():
    logo = client.get("/static/brand/tridaai-logo.png")
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"
    icon = client.get("/static/brand/tridaai-icon.png")
    assert icon.status_code == 200
    favicon = client.get("/favicon.svg")
    assert favicon.status_code == 200
    assert "image/svg+xml" in favicon.headers["content-type"]
    ico = client.get("/favicon.ico")
    assert ico.status_code == 200


def test_analyze_returns_pipeline_trace():
    samples = client.get("/samples").json()
    body = client.post("/shipments/analyze", json=samples[0]).json()
    trace = body["trace"]
    assert [s["name"] for s in trace] == [
        "extract", "ingest", "classify", "retrieve", "draft", "validate",
        "human_approval",
    ]
    classify = trace[2]
    assert body["classification"]["exception_type"] in classify["summary"]
    assert classify["details"], "classify step must expose evidence details"
    validate = trace[5]
    assert validate["status"] == "passed"
    assert any("references_shipment_id" in d for d in validate["details"])
    assert trace[6]["status"] == "awaiting"


def test_evals_results_endpoint():
    response = client.get("/evals/results")
    assert response.status_code == 200
    results = response.json()
    assert results["total_cases"] >= 30
    assert results["passed"] is True
    assert "regression gate" in results["framing"]
    assert set(results["per_type"]) >= {"delay", "damage", "document_mismatch"}


def test_samples_endpoint_lists_at_least_ten():
    response = client.get("/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 10
    ids = {s["shipment_id"] for s in samples}
    assert len(ids) == len(samples)


def test_ui_flow_analyze_then_reject():
    samples = client.get("/samples").json()
    shipment = samples[0]
    analyzed = client.post("/shipments/analyze", json=shipment)
    assert analyzed.status_code == 200
    body = analyzed.json()
    assert body["validation"]["checks"], "guardrail checks must be exposed for the UI"
    check_names = {c["name"] for c in body["validation"]["checks"]}
    assert "references_shipment_id" in check_names
    assert "no_prohibited_promises" in check_names

    rejected = client.post(
        f"/shipments/{shipment['shipment_id']}/reject",
        json={"reviewer": "ops-lead", "reason": "Needs a phone call first"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["approval_status"] == "rejected"
    assert rejected.json()["external_action_taken"] is False

    # A decided shipment cannot be re-decided.
    again = client.post(
        f"/shipments/{shipment['shipment_id']}/reject",
        json={"reviewer": "ops-lead"},
    )
    assert again.status_code == 422


def test_reject_unknown_shipment_404():
    response = client.post("/shipments/NOPE/reject", json={"reviewer": "x"})
    assert response.status_code == 404


def test_bundled_samples_match_repo_copy():
    """The packaged samples and data/sample/ must never drift apart."""
    bundled = load_sample_shipments()
    repo_copy = json.loads((REPO_ROOT / "data" / "sample" / "sample_shipments.json").read_text())
    assert bundled == repo_copy
    assert len(bundled) >= 10


def test_sample_edge_cases_classify_as_intended():
    from shipment_agent.classifier import classify_shipment
    from shipment_agent.schemas import ShipmentInput
    from shipment_agent.tools import compare_documents

    expected = {
        "SYN-1007": "damage",              # partial damage
        "SYN-1008": "none",                # delay that recovered
        "SYN-1009": "document_mismatch",   # quantity-only mismatch
        "SYN-1010": "missed_appointment",  # missed + rescheduled
        "SYN-1011": "none",                # clean, on time, "no damage reported"
        "SYN-1012": "none",                # inspection negations + "undamaged"
    }
    by_id = {s["shipment_id"]: s for s in load_sample_shipments()}
    for shipment_id, expected_type in expected.items():
        shipment = ShipmentInput.model_validate(by_id[shipment_id])
        result = classify_shipment(shipment, compare_documents(shipment.documents))
        assert result.exception_type.value == expected_type, shipment_id
    # Quantity-only: exactly one mismatch field, and it is quantity_units.
    syn1009 = ShipmentInput.model_validate(by_id["SYN-1009"])
    mismatches = compare_documents(syn1009.documents)
    assert [m.field for m in mismatches] == ["quantity_units"]


def test_demo_trace_runs(capsys):
    print_trace(0)
    out = capsys.readouterr().out
    assert "PENDING_HUMAN_APPROVAL" in out
    assert "GUARDRAIL CHECKS" in out
    assert "RETRIEVED POLICY CONTEXT" in out
