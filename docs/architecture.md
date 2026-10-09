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
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   extract   │──►│    ingest    │──►│   classify   │──►│   retrieve   │
│ LLM fields +│   │ validate,    │   │ rules × LLM  │   │ keyword /    │
│ confidence, │   │ compute      │   │ cross-check  │   │ semantic /   │
│ code-checked│   │ facts        │   │              │   │ hybrid+rerank│
└─────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                              │
        ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  ▼
        │   human_   │◄──│   validate   │◄──│    draft     │◄─ diagnose ─► options
        │   approval │   │ guardrails   │   │ grounded on  │   (root cause;  (proposed,
        │   (GATE)   │   │ as code      │   │ recommended  │    cited)        scored by code)
        └──────────────┘   └──────────────┘   └──────────────┘
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
| `extractions` | extract | per-document extracted fields, per-field confidence, code cross-check status |
| `delay_hours` | ingest | computed exactly, never estimated by a model |
| `document_mismatches` | ingest | field-level BOL vs invoice diffs |
| `classification` | classify | type, severity, confidence, signals, rationale |
| `cross_check` | classify | rules vs LLM outcome + resolution (provider mode only) |
| `policies` | retrieve | top-3 policy snippets with scores (+ how each was retrieved) |
| `retrieval_info` | retrieve | mode, hybrid pool stats, vector store that served |
| `diagnosis` | diagnose | root cause, summary, evidence list, policy citations |
| `recovery_options` | options | proposed options with code-computed scores |
| `recommended_option_id` | options | the highest-scoring option; the draft grounds on it |
| `draft` | draft | subject, body, claim packet (incl. diagnosis + options), citations |
| `validation` | validate | guardrail pass/fail, errors, warnings |
| `approval_status` | human_approval | always `awaiting_approval` at graph exit |

Why a graph instead of a chain? Each node is independently testable, the
intermediate state is inspectable (an approver can see *why* a shipment was
classified), and production extensions — a supervisor node, a retry loop on
retrieval, parallel document processors — slot in without rewriting the
pipeline.

## 4. Key design decisions and trade-offs

**Extraction before classification, cross-checked by code.** Documents
arrive as text plus whatever fields the caller already has. In provider
mode the extract node asks the LLM for the typed fields
(`quantity_units`, `weight_kg`, `consignee`, `sku`) with a per-field
confidence, via structured output; deterministic code then compares
extracted vs provided values field by field (`match` / `mismatch` /
`missing_in_text` / `extracted_only`) and the discrepancies join the
evidence the rest of the pipeline — and the approver — can see. In
default mode nothing is extracted: provided fields pass through with
status `source_provided` and no confidence, so the offline path never
pretends a model ran. A failed or malformed extraction falls back to
the provided fields and the run continues; extraction can degrade the
evidence, never break the pipeline.

**Classification is a cross-check, not a suggestion.** Exception type
drives customer messaging and claim handling, so it must be auditable.
The deterministic rules run first and independently — prioritised
deliberately: damage > missed appointment > document mismatch > delay >
none, so specific, time-sensitive signals are never swallowed by a
generic "late" keyword. In provider mode the LLM then classifies the
same facts *independently* (it is never shown the rule result) with a
structured type/severity/confidence/rationale. The two outcomes and
their resolution are recorded in `cross_check` and the trace, under one
explicit policy:

> **Resolution policy.** Agreement → the rule result stands. Disagreement
> → the rules are authoritative (`rules_authoritative`), *except* when
> the rules landed on `none` or below 0.85 confidence while the LLM is at
> 0.85 confidence or above on a concrete type — then the LLM result is
> adopted (`llm_adopted`) and flagged in the adopted classification's
> signals and rationale, so the approver always knows a model overrode
> the rules, and why. No usable LLM result → `rules_only`, the default
> mode's behaviour for every run.

Trade-off: rules still miss novel phrasings in default mode; the
cross-check is the mitigation, and recurring disagreements in production
are what get promoted into rules + golden cases. A failed LLM call
records `rules_only` with the failure in the note and changes nothing
else. (`AgentResult.llm_suggestion` remains as a compatibility mirror of
the cross-check's LLM side.)

**Facts are computed, never generated.** Delay hours and document mismatches
come from functions (`tools.py`), not from a model reading prose. That
keeps those facts out of the model's hands.

**Diagnosis and recovery options are separate nodes.** After retrieval,
a diagnose node assembles the evidence deterministically — computed
delay, document diffs, extraction discrepancies, classification signals,
retrieved policy IDs — and produces a root-cause note over it: the LLM
composes the prose in provider mode, a template composes from the same
evidence structure in default mode, and a failed composition falls back
to the template. An options node then proposes 2–3 recovery options
(template or LLM), but **every number is computed by code**: per-kind
formulas turn delay hours and severity into ETA improvement, added cost,
and an SLA score, combined as `0.6·SLA + 0.4·ETA − 0.25·cost`
(normalised). The highest score is the recommendation; the draft and
the claim packet ground on it. Keeping "propose" (language) and
"score" (arithmetic) in different hands is the point — a model that
invents a plausible-looking cost is a production incident, not a quirk.

**Retrieval behind an interface, hybrid with an explicit rerank.** The
agent depends on a `Retriever` protocol, selected with `RETRIEVER`:
`KeywordRetriever` (transparent token overlap, deterministic),
`SemanticRetriever` (embedding cosine, same return shape), and
`HybridRetriever`. Hybrid retrieves a candidate pool from both (2×
top-k each), deduplicates by policy ID, and **reranks by reciprocal-rank
score fusion** (RRF, k=60) down to the cited top-3 — the trace shows
retrieve → merge → rerank explicitly. This is score-fusion reranking,
named honestly: no cross-encoder ships in this repo. Semantic search
runs on a **local Chroma store** when the `vectordb` extra is installed
— embedded persistent directory (`CHROMA_DIR`) or a server
(`CHROMA_HOST`, as in the compose local stack), corpus indexed on first
use under a content-fingerprinted collection name, embeddings always
supplied by our own provider client. Without the extra, in-memory
cosine serves behind the same interface, and `vector_store` in the
result says which path served. Embeddings follow the backend: OpenAI
(or any OpenAI-compatible endpoint via `OPENAI_BASE_URL`) — or Ollama
when the backend is Ollama. Anthropic has no embeddings API, so an
Anthropic backend still needs an OpenAI key (or Ollama) for semantic
modes, and the retriever fails loudly saying exactly that rather than
quietly degrading. A production deployment can still swap the
implementation for LlamaIndex over a managed store without touching
the graph.

**Model backend behind an interface, mock as offline fallback only.**
The default backend renders from deterministic templates offline; it is
the fallback for no-key environments, labelled as such in the console
and the README — never presented as the agent itself. All surfaces
(API, CLI, traced demo) select a provider with `MODEL_BACKEND`:
`openai`, `anthropic`, or `ollama` — a preset over the
OpenAI-compatible path pointing at a local Ollama server
(`OLLAMA_BASE_URL`, default `http://localhost:11434/v1`, placeholder
key), which makes a fully local real-LLM run a configuration change,
not a code change. `OPENAI_BASE_URL` is the general switch for any
other OpenAI-compatible hosted provider (LiteLLM, Together, Groq,
Azure, …). Configuration is environment variables only (the app loads
a repo-root `.env` at startup, real environment wins). The backends
fail loudly — with the fix in the message — without the optional SDKs
or a key; nothing silently falls back to the mock. In provider mode
there is no single-prompt path: every run visibly performs extraction →
cross-check → retrieval → diagnosis → options → draft, each as its own
call with its own prompt, all versioned in `prompts.py`. Whichever
backend drafts, the guardrails run on its output afterwards.
Trade-off: template drafts are less fluent than LLM drafts; they are
also incapable of inventing a delivery time, which is the failure mode
that matters here. Note the template *can* carry a compensation
promise — it quotes the source record verbatim, and a carrier agent's
note can contain one (sample SYN-1013 does). That is exactly what the
guardrail layer is for, in both modes.

**Approvals persist; the API can be gated.** Analyses and decisions are
stored through a small store interface (`store.py`): SQLite on disk by
default (`STATE_DB_PATH`, default `.data/state.db`, git-ignored), so
the approval queue survives restarts; the in-memory implementation
remains as the test double (`STATE_DB_PATH=:memory:`). Setting
`API_KEY` turns on a shared-key check (`X-API-Key` header) for all data
endpoints; unset, the API is open and documented as a local-dev default.
Neither changes the gate semantics: approval records a decision and
still performs no external action.

**Guardrails as code, not prompts.** `guardrails.py` rejects drafts that
lack the shipment ID, lack policy citations on exception drafts, or contain
prohibited promise language ("we guarantee", "full refund", …). A draft
that fails validation cannot be approved through the service layer.

**The approval gate is structural.** There is no send/file/act node in the
graph at all. Approval in `service.py` records *who* approved and marks the
packet ready — and still performs no external action in this prototype.

## 5. Data model

Inputs are deliberately boring JSON: a shipment, its schedule, its latest
event text, condition notes, and documents — each document carrying its
text (`raw_text`) plus whatever structured fields the caller already has
(`quantity_units`, `weight_kg`, `consignee`, `sku`). The extract node
produces the typed fields from the text in provider mode and cross-checks
them against the provided ones. What this prototype does *not* do is OCR:
scanned documents need an OCR step in front of it in production, and
photo damage assessment (a VLM step) is out of scope entirely.

## 6. Failure modes

| Failure | Behaviour |
|---|---|
| Missing schedule data | delay_hours = None; classification falls back to event keywords; severity defaults to medium |
| BOL or invoice absent | mismatch check returns empty; classification relies on event text |
| Novel exception phrasing | may classify as `none` — visible in evals as a miss; golden set grows from these |
| Retrieved policies irrelevant | draft still carries citations; approver sees policy titles and can reject |
| LLM backend unavailable | explicit RuntimeError at construction naming the fix (missing key or missing `llm` extra); no silent fallback to the mock |
| LLM classification malformed/unavailable | cross-check records `rules_only` with the failure in its note; the rule result stands and the run continues |
| LLM extraction / diagnosis / options failure | that node falls back to provided fields / the evidence template / template options; the run continues |
| Chroma installed but store broken | semantic retrieval falls back to in-memory cosine; `vector_store` in the result reports `memory` |
| `RETRIEVER=semantic` with no embeddings route | explicit RuntimeError at construction: embeddings need OpenAI(-compatible) or the Ollama backend |
| Draft fails guardrails | approval is blocked with the exact errors surfaced (sample SYN-1013 demonstrates this deterministically) |
| Conflicting signals (damage + delay) | priority order resolves deterministically; rationale records the winning signal |

## 7. Security notes

- No credentials are required for the default path. All provider
  configuration is environment variables. The application loads a `.env`
  file from the repo root at startup (parser in `config.py`); variables
  set in the real environment take precedence over the file. `.env` is
  git-ignored — never commit a filled-in one. `.env.example` lists every
  variable.
- All sample data is synthetic; the repo must never contain real shipment,
  customer, or carrier data.
- API authentication is optional and coarse: set `API_KEY` and every
  data endpoint requires the `X-API-Key` header; unset, the API is open
  by design for local development (the console says so). A shared key
  is not per-user identity — production needs real auth and per-client
  tenancy on top.
- Approval records persist in a local SQLite file (see §4); treat it
  like any operational data store in a deployment (backups, access).
- Prompt-injection surface: free-text event/condition fields flow into the
  LLM prompt in LLM mode. In production, treat them as untrusted data
  (delimit, never concatenate into system instructions) and keep the
  deterministic guardrails after generation — as this blueprint does.

## 8. What to change first (prototyping order)

Ordered by value when adapting this blueprint to your own operation:

1. **Policy corpus** — swap the synthetic SOPs (`policies_data.py`) for
   your real exception policies; drafts immediately speak your language.
2. **Model backend** — install the `llm` extra, copy `.env.example` to
   `.env`, set `MODEL_BACKEND=openai|anthropic|ollama` (plus the key for
   the cloud providers), and compare draft quality against the fallback
   on the golden set — then run the LLM-judge pack
   (`evals/run_llm_evals.py`) for groundedness. Point `OPENAI_BASE_URL`
   at a compatible endpoint if you host your own.
3. **Intake shape** — map one real source (a TMS export or webhook
   payload) onto `ShipmentInput`; keep the rest of the graph untouched.
4. **Retriever** — set `RETRIEVER=hybrid` (and install the `vectordb`
   extra for the local Chroma store) for embedding-based ranking of the
   shipped corpus, or replace the retriever with embeddings over your
   full policy library behind the same `Retriever` protocol.
5. **Exception types** — add the exceptions your operation actually sees
   (customs hold, address issue, …) as rules + golden cases.
6. **Action layer** — wire approve → your messaging/claims system, behind
   the existing gate, with an audit log.

## 9. How you would productionise this

1. **Integrations:** TMS/carrier event feeds (webhooks or EDI) replace
   manual JSON input; an OCR step feeds the in-graph extraction for
   scanned documents.
2. **Knowledge:** scale the shipped local Chroma store to the real SOP
   library (or LlamaIndex + a managed vector store over real SOPs,
   carrier claim rules, and customer contracts); retrieval evaluated on
   a labelled set.
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
7. **Access control & tenancy:** per-user identity on top of the shipped
   optional API key, per-client policy corpora and data isolation, PII
   handling per the client's policy.

## 10. Limitations of this prototype

- Keyword/rule classification covers the phrasings in its golden set; it
  is a regression gate, not a real-world benchmark. The provider-mode
  cross-check can adopt a confident LLM classification (flagged), but
  default mode keeps the rules' blind spots.
- No retrieval eval set: hybrid ranking and the rerank are tested for
  shape and determinism, not measured for relevance quality.
- The LLM-judge eval pack is a model judging a model — a groundedness
  regression signal, not a human evaluation, and it only runs with a
  real provider configured.
- API auth is a single optional shared key; approvals persist locally
  in SQLite — a real deployment needs identity, tenancy, and a managed
  database.
- Extraction consumes document text; there is no OCR engine and no
  photo/VLM damage assessment.
- The Docker local stack (agent + Ollama + Chroma) is reviewed but not
  build-verified — no Docker daemon in the development environment.
- No carrier/TMS integration, no sending, no claims filing — by design.
