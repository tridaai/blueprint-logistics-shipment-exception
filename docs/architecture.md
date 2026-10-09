# Architecture — Logistics Shipment Exception Agent

> **Status: reference prototype.** Built by Trida AI as an open blueprint for
> forward-deployed AI engineering in logistics operations. It runs entirely
> on synthetic data, takes no external action, and is not a production
> system. Nothing in this document describes a real client engagement.

## 1. Problem framing

Shipment exceptions — delays, damage, document mismatches, missed delivery
appointments — are where logistics operations burn skilled human time.
Detecting one is usually easy; the expensive part is the follow-through:
checking the documents, finding the governing policy, drafting a correct
customer update, and assembling a claim packet, for every exception, every
day, consistently.

This blueprint automates the *drafting and assembly* and stops there. No
output is acted on without human approval. That boundary is the design,
not a limitation to apologise for: customer communication and claims are
exactly where an unreviewed agent does reputational and financial damage.

## 2. System overview

```
Shipment input (events + documents, JSON)
        │
        ▼
┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│   ingest    │──►│   classify   │──►│   retrieve   │
│ validate,   │   │ deterministic│   │ policy       │
│ compute     │   │ rules first  │   │ context      │
│ facts       │   │              │   │              │
└─────────────┘   └──────────────┘   └──────────────┘
                                            │
        ┌──────────────┐   ┌──────────────┐   ▼
        │   human_   │◄──│   validate   │◄── draft
        │   approval │   │ guardrails   │   (mock or
        │   (GATE)   │   │ as code      │    LLM backend)
        └──────────────┘   └──────────────┘
              │
              ▼
            END — no external action exists in this graph
```

Implemented as a LangGraph `StateGraph` (`src/shipment_agent/graph.py`).
Surfaced through FastAPI (`api.py`), a traced demo (`demo.py`), a batch
CLI (`cli.py`), and a service layer that records approvals (`service.py`).

## 3. Graph and state design

State is a `TypedDict`. Node outputs are dictionaries produced from
Pydantic models, so values crossing node boundaries have a defined,
serialisable shape:

| Field | Produced by | Contents |
|---|---|---|
| `shipment` | caller | `ShipmentInput` — events, documents, schedule |
| `delay_hours` | ingest | computed exactly, never estimated by a model |
| `document_mismatches` | ingest | field-level BOL vs invoice diffs |
| `classification` | classify | type, severity, confidence, signals, rationale |
| `policies` | retrieve | top-3 policy snippets with scores |
| `draft` | draft | subject, body, claim packet, citations |
| `validation` | validate | guardrail pass/fail, errors, warnings |
| `approval_status` | human_approval | always `awaiting_approval` at graph exit |

Why a graph instead of a chain? Each node is independently testable, the
intermediate state is inspectable (an approver can see *why* a shipment was
classified), and production extensions — a supervisor node, a retry loop on
retrieval, parallel document processors — slot in without rewriting the
pipeline.

## 4. Key design decisions and trade-offs

**Deterministic classification before any LLM.** Exception type drives
customer messaging and claim handling, so it must be auditable and stable
from run to run and in production. Rules are prioritised deliberately:
damage > missed appointment > document mismatch > delay > none, so specific,
time-sensitive signals are never swallowed by a generic "late" keyword.
Trade-off: rules miss novel phrasings. Mitigation in production: route
low-confidence cases to an LLM classifier *as a suggestion*, log
disagreements, and promote recurring patterns into rules + golden cases.

**Facts are computed, never generated.** Delay hours and document mismatches
come from functions (`tools.py`), not from a model reading prose. That
keeps those facts out of the model's hands.

**Retrieval behind an interface.** The agent depends on a `Retriever`
protocol. The in-repo `KeywordRetriever` is transparent and deterministic
(stable for tests and demos); a production deployment swaps in LlamaIndex
over a vector store fed by the client's SOP/claims document systems,
without touching the graph.

**Model backend behind an interface, mock by default.** The default
backend renders drafts from deterministic templates offline. The API
service can select the OpenAI/Anthropic backends with `MODEL_BACKEND`;
they fail loudly without the optional SDKs or a key. Trade-off:
template drafts are less fluent than LLM drafts; they are also incapable of
inventing a delivery time or a compensation promise, which is the failure
mode that matters here.

**Guardrails as code, not prompts.** `guardrails.py` rejects drafts that
lack the shipment ID, lack policy citations on exception drafts, or contain
prohibited promise language ("we guarantee", "full refund", …). A draft
that fails validation cannot be approved through the service layer.

**The approval gate is structural.** There is no send/file/act node in the
graph at all. Approval in `service.py` records *who* approved and marks the
packet ready — and still performs no external action in this prototype.

## 5. Data model

Inputs are deliberately boring JSON: a shipment, its schedule, its latest
event text, condition notes, and documents with structured extracted fields
(`quantity_units`, `weight_kg`, `consignee`, `sku`). In production, the
document fields come from an OCR/extraction step (a blueprint of its own);
this prototype accepts them as given so the exception logic is isolated
and testable.

## 6. Failure modes

| Failure | Behaviour |
|---|---|
| Missing schedule data | delay_hours = None; classification falls back to event keywords; severity defaults to medium |
| BOL or invoice absent | mismatch check returns empty; classification relies on event text |
| Novel exception phrasing | may classify as `none` — visible in evals as a miss; golden set grows from these |
| Retrieved policies irrelevant | draft still carries citations; approver sees policy titles and can reject |
| LLM backend unavailable | explicit RuntimeError at construction; mock remains the default |
| Draft fails guardrails | approval is blocked with the exact errors surfaced |
| Conflicting signals (damage + delay) | priority order resolves deterministically; rationale records the winning signal |

## 7. Security notes

- No credentials are required for the default path. Optional LLM keys are
  read from the process environment; `.env.example` lists the variable
  names, but the application does not load a `.env` file automatically.
  A local `.env` is git-ignored.
- All sample data is synthetic; the repo must never contain real shipment,
  customer, or carrier data.
- The API has no authentication — acceptable for a local prototype, and
  called out here because it is the first thing to fix for any deployment.
- Prompt-injection surface: free-text event/condition fields flow into the
  LLM prompt in LLM mode. In production, treat them as untrusted data
  (delimit, never concatenate into system instructions) and keep the
  deterministic guardrails after generation — as this blueprint does.

## 8. What to change first (prototyping order)

Ordered by value when adapting this blueprint to your own operation:

1. **Policy corpus** — swap the synthetic SOPs (`policies_data.py`) for
   your real exception policies; drafts immediately speak your language.
2. **Model backend** — set `MODEL_BACKEND=openai|anthropic` and compare
   draft quality against the mock on the golden set.
3. **Intake shape** — map one real source (a TMS export or webhook
   payload) onto `ShipmentInput`; keep the rest of the graph untouched.
4. **Retriever** — replace keyword retrieval with embeddings over your
   full policy library behind the same `Retriever` protocol.
5. **Exception types** — add the exceptions your operation actually sees
   (customs hold, address issue, …) as rules + golden cases.
6. **Action layer** — wire approve → your messaging/claims system, behind
   the existing gate, with an audit log.

## 9. How you would productionise this

1. **Integrations:** TMS/carrier event feeds (webhooks or EDI) replace
   manual JSON input; document fields come from an OCR + extraction
   pipeline with confidence scores.
2. **Knowledge:** LlamaIndex + vector store over real SOPs, carrier claim
   rules, and customer contracts; retrieval evaluated on a labelled set.
3. **Approval UX:** queue UI with side-by-side evidence (classification
   signals, source documents, policy text), one-click edit/approve/reject,
   and full audit log of who approved what, when.
4. **Action layer:** after approval, send via the client's messaging system
   and file claims via carrier portals/APIs — behind feature flags, with
   idempotency keys and rate limits.
5. **Observability:** LangGraph tracing (e.g. Langfuse / OpenTelemetry),
   per-node latency and cost, classification drift dashboards.
6. **Evals as a regression gate:** the golden dataset grows from real
   (anonymised, consented) corrections made by approvers; every rule or
   prompt change runs against it locally (pytest + evals + demo via the
   Makefile) before merge.
7. **Access control & tenancy:** authenticated API, per-client policy
   corpora and data isolation, PII handling per the client's policy.

## 10. Limitations of this prototype

- Keyword/rule classification covers the phrasings in its golden set; it is
  a regression gate, not a real-world benchmark.
- The retriever is keyword-based; semantic retrieval is an interface swap,
  not implemented here.
- Approvals live in memory and disappear on restart.
- No OCR, no carrier/TMS integration, no sending — by design.
