"""Retrieval relevance evaluation — is retrieval quality measured or assumed?

The golden evals score the classifier; the shape tests score the
retrievers' mechanics. This pack scores the thing an approver
actually depends on: **for a query phrased the way shipments are
phrased, do the governing policies land in the top-3 the diagnosis
cites?** Cases live in ``retrieval_golden.jsonl`` — a labelled
query → expected-policies set, the queries derived from the sample
shipments' own event/condition/document text and the graph's
query shape (exception type + shipment content + the standing
trailer), including the customer-simulation case (RQ-04): a
customer's own operational SOP, written in operational vocabulary,
must rank for the case it describes.

Cases may also carry ``forbidden`` ids — documents that must NOT
surface (another tenant's SOP in this tenant's ranking). A
forbidden hit zeroes the case's score whatever the expected recall
was: surfacing the wrong tenant's SOP is not a partial success.
Cases with an empty ``expected`` and only ``forbidden`` ids score
1.0 exactly when nothing forbidden surfaces.

Three corpora shapes are measured:

- ``standard`` / ``ops_sop`` — the bundled shared corpus (plus the
  customer-SOP case's extra document).
- ``tenant_acme`` / ``tenant_globex`` — one tenant's view of the
  corpus (shared + that tenant's bundled SOPs), built by
  ``policies_data.policies_for_tenant``: the acme reefer query
  must rank SOP-ACME-01 for acme and must never surface it for
  globex.
- ``stored`` — a document supplied at runtime through the service's
  corpus management (the POST /policies path), measured in two
  phases: with the document stored it is the expected hit; after
  it is removed (the DELETE path) it must vanish from the ranking.

Metric: recall@3 per case (|retrieved ∩ expected| / |expected|,
with the forbidden rule above), averaged. Two rankings are
reported:

- ``keyword`` — the deterministic default retriever. This is the
  gated number (threshold below): it runs offline, always.
- ``hybrid`` — keyword + semantic merged by the production RRF
  rerank. Offline it runs over a **stand-in embedder** (hashed
  bag-of-words vectors, in this file): that measures the merge and
  the harness, NOT real embedding quality — the output says so.
  Pass ``--real`` to measure the configured embeddings instead
  (``RETRIEVER=hybrid`` stack: OpenAI/Ollama per the environment);
  that fails loudly when no embeddings are configured.

Run from the repo root:

    python evals/run_retrieval_evals.py [--real]

Exit code is 0 when keyword mean recall@3 >= 0.85, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shipment_agent.policies_data import POLICIES, policies_for_tenant  # noqa: E402
from shipment_agent.retriever import (  # noqa: E402
    HybridRetriever,
    KeywordRetriever,
    SemanticRetriever,
    _tokens,
    rerank_fused,
)

GOLDEN = Path(__file__).resolve().parent / "retrieval_golden.jsonl"
TOP_K = 3
KEYWORD_THRESHOLD = 0.85

# The customer's-own-SOP case (see test_retrieval_query.py): an
# operational-language SOP whose vocabulary comes from a shipment's
# event text, not from the classifier's rationale.
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

CORPORA = {
    "standard": POLICIES,
    "ops_sop": [*POLICIES, OPS_SOP],
    "tenant_acme": policies_for_tenant("acme"),
    "tenant_globex": policies_for_tenant("globex"),
}


# ---------------------------------------------------------------------------
# Stand-in embeddings (offline hybrid measurement)
# ---------------------------------------------------------------------------

_STANDIN_DIMS = 512


def _standin_vector(text: str) -> list[float]:
    """A deterministic bag-of-words vector: token counts hashed
    into fixed buckets. Cosine over these ranks by shared vocabulary
    with length normalisation — a stand-in for a real embedding
    model, good enough to exercise the hybrid merge honestly, and
    labelled as a stand-in wherever its numbers appear."""
    vector = [0.0] * _STANDIN_DIMS
    for token in _tokens(text):
        vector[zlib.crc32(token.encode("utf-8")) % _STANDIN_DIMS] += 1.0
    return vector


class _StandinEmbeddings:
    def create(self, model=None, input=None):
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=i, embedding=_standin_vector(text))
                for i, text in enumerate(input)
            ]
        )


class _StandinClient:
    def __init__(self) -> None:
        self.embeddings = _StandinEmbeddings()


class _StandinSemanticRetriever(SemanticRetriever):
    """The production semantic retriever, embeddings stood in for.

    Only the client builder is overridden; ranking, caching, and
    scores are the shipped code paths. The pgvector/Chroma probes
    are pre-answered 'no' so the in-memory cosine path serves —
    this harness measures ranking, not storage.
    """

    def _build_embeddings_client(self):
        return _StandinClient(), "standin-bow-512", "standin", "local"

    def __init__(self, policies):
        super().__init__(policies)
        self._pgvector_checked = True
        self._pgvector_ready = False
        self._chroma_checked = True
        self._chroma_collection = None


class _StandinHybrid:
    """HybridRetriever's merge, over the stand-in semantic half."""

    name = "hybrid (stand-in embeddings)"

    def __init__(self, policies):
        self._keyword = KeywordRetriever(policies)
        self._semantic = _StandinSemanticRetriever(policies)

    def retrieve(self, query: str, top_k: int = TOP_K):
        pool = top_k * 2
        return rerank_fused(
            self._keyword.retrieve(query, top_k=pool),
            self._semantic.retrieve(query, top_k=pool),
            top_k=top_k,
        )


def build_retrievers(policies: list[dict], *, real: bool) -> dict:
    """The two rankings under measurement, for one corpus."""
    retrievers = {"keyword": KeywordRetriever(policies)}
    if real:
        hybrid = HybridRetriever(policies)
        retrievers["hybrid (configured embeddings)"] = hybrid
    else:
        retrievers[_StandinHybrid.name] = _StandinHybrid(policies)
    return retrievers


def _score(retrieved: list[str], expected: list[str], forbidden: list[str]) -> dict:
    """One ranking's result for one case: expected recall, the
    forbidden hits, and the case score (a forbidden hit zeroes it —
    see the module docstring)."""
    hits = [pid for pid in expected if pid in retrieved]
    forbidden_hits = [pid for pid in forbidden if pid in retrieved]
    recall = len(hits) / len(expected) if expected else 1.0
    return {
        "retrieved": retrieved,
        "hits": hits,
        "expected_recall_at_3": round(recall, 4),
        "forbidden_hits": forbidden_hits,
        "recall_at_3": round(0.0 if forbidden_hits else recall, 4),
    }


def _evaluate_stored(case: dict, *, real: bool) -> dict:
    """A stored-document case, in two phases through the real
    service path (upsert → corpus → retrieve; remove → corpus →
    retrieve). Phase 1: the stored document is the expected hit.
    Phase 2: after removal it must be absent from the ranking —
    a delete that leaves the document retrievable is a corpus
    leak, scored 0 like any forbidden hit."""
    from shipment_agent.model_backends import MockModelBackend
    from shipment_agent.service import ShipmentService
    from shipment_agent.store import InMemoryStore

    document = case["document"]
    tenant = case["tenant"]
    service = ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(POLICIES),
        store=InMemoryStore(),
        checkpointer=False,
    )
    service.upsert_tenant_policy(
        tenant, document["policy_id"], document["title"], document["text"]
    )
    row: dict = {
        "case_id": case["case_id"],
        "expected": list(case["expected"]),
        "results": {},
    }
    expected = list(case["expected"])
    for name, retriever in build_retrievers(
        service.corpus_policies(tenant), real=real
    ).items():
        retrieved = [
            p.policy_id for p in retriever.retrieve(case["query"], top_k=TOP_K)
        ]
        row["results"][name] = _score(retrieved, expected, [])
    service.remove_tenant_policy(tenant, document["policy_id"])
    for name, retriever in build_retrievers(
        service.corpus_policies(tenant), real=real
    ).items():
        retrieved_after = [
            p.policy_id for p in retriever.retrieve(case["query"], top_k=TOP_K)
        ]
        result = row["results"][name]
        result["after_delete_retrieved"] = retrieved_after
        result["after_delete_clear"] = document["policy_id"] not in retrieved_after
        if not result["after_delete_clear"]:
            result["recall_at_3"] = 0.0
    return row


def evaluate(cases: list[dict], *, real: bool = False) -> dict:
    """Run every case against both rankings; return the report."""
    by_corpus: dict[str, dict] = {}
    for case in cases:
        corpus = case.get("corpus", "standard")
        if corpus == "stored" or corpus in by_corpus:
            continue
        by_corpus[corpus] = build_retrievers(CORPORA[corpus], real=real)
    totals: dict[str, list[float]] = {}
    rows: list[dict] = []
    for case in cases:
        if case.get("corpus") == "stored":
            row = _evaluate_stored(case, real=real)
        else:
            retrievers = by_corpus[case.get("corpus", "standard")]
            expected = list(case["expected"])
            forbidden = list(case.get("forbidden", []))
            row = {"case_id": case["case_id"], "expected": expected, "results": {}}
            for name, retriever in retrievers.items():
                retrieved = [
                    p.policy_id
                    for p in retriever.retrieve(case["query"], top_k=TOP_K)
                ]
                row["results"][name] = _score(retrieved, expected, forbidden)
        for name, result in row["results"].items():
            totals.setdefault(name, []).append(result["recall_at_3"])
        rows.append(row)
    summary = {
        name: {
            "cases": len(values),
            "mean_recall_at_3": round(sum(values) / len(values), 4),
        }
        for name, values in totals.items()
    }
    return {
        "dataset": "retrieval_golden.jsonl (synthetic, labelled)",
        "top_k": TOP_K,
        "embeddings": "configured" if real else "stand-in (hashed bag-of-words)",
        "summary": summary,
        "cases": rows,
        "keyword_threshold": KEYWORD_THRESHOLD,
        "passed": summary["keyword"]["mean_recall_at_3"] >= KEYWORD_THRESHOLD,
        "framing": (
            "Labelled relevance set used as a regression gate for retrieval "
            "ranking. Offline hybrid numbers use stand-in embeddings and "
            "measure the merge, not embedding quality."
        ),
    }


def _write_results(payload: dict) -> None:
    targets = [Path(__file__).resolve().parent / "retrieval_results.json"]
    packaged = (
        Path(__file__).resolve().parents[1]
        / "src" / "shipment_agent" / "data" / "retrieval_eval_results.json"
    )
    if packaged.parent.is_dir():
        targets.append(packaged)
    for target in targets:
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="measure the configured embeddings for the hybrid ranking "
        "instead of the offline stand-in",
    )
    args = parser.parse_args()
    cases = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        report = evaluate(cases, real=args.real)
    except Exception as exc:  # --real without embeddings configured: loud
        print(f"Retrieval eval could not run: {exc}")
        return 2
    from datetime import datetime, timezone

    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_results(report)

    print(f"Retrieval relevance: {len(cases)} labelled cases, recall@{TOP_K}")
    for name, stats in report["summary"].items():
        print(f"  {name:<32} mean recall@{TOP_K} = {stats['mean_recall_at_3']:.1%}")
    for row in report["cases"]:
        for name, result in row["results"].items():
            if result["recall_at_3"] < 1.0:
                missing = [p for p in row["expected"] if p not in result["hits"]]
                detail = f"missed {missing}; retrieved {result['retrieved']}"
                if result.get("forbidden_hits"):
                    detail += f"; FORBIDDEN hit {result['forbidden_hits']}"
                if result.get("after_delete_clear") is False:
                    detail += (
                        f"; still retrieved after delete: "
                        f"{result['after_delete_retrieved']}"
                    )
                print(
                    f"  {row['case_id']} [{name}]: recall {result['recall_at_3']:.2f} "
                    f"— {detail}"
                )
    print(
        "PASS" if report["passed"] else f"FAIL (keyword threshold {KEYWORD_THRESHOLD:.0%})"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
