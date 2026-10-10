# Architecture — Logistics Shipment Exception Agent

> **Status: reference prototype.** Built by Trida AI as an open blueprint for
> forward-deployed AI engineering in logistics operations. It runs entirely
> on synthetic data, takes no external action by default (the one opt-in
> exception is the operator-configured approval webhook, §4), and is not a
> production system. Nothing in this document describes a real client
> engagement.

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
        │   human_   │◄──│   validate   │◄──│    verify    │◄─ draft ◄─ diagnose ─► options
        │   approval │   │ guardrails   │   │ self-critique│   (grounded on  (root cause;  (proposed,
        │   (GATE)   │   │ as code +    │   │ vs facts     │    recommended)  cited)        scored by code)
        │            │   │ repair loop  │   │              │
        └──────────────┘   └──────────────┘   └──────────────┘
              │
              ▼
            END — no external action exists in this graph
            (an approval may POST to an operator-configured webhook — §4)
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
| `document_check_warning` | ingest | set when no BOL/invoice pair exists — the skipped check is announced, never silent |
| `classification` | classify | type, severity, confidence, signals, rationale |
| `cross_check` | classify | rules vs LLM outcome + resolution (provider mode only) |
| `classify_note` | classify | the translated provider error when the LLM cross-check degraded to rules-only |
| `policies` | retrieve | top-3 policy snippets with scores (+ how each was retrieved) |
| `retrieval_info` | retrieve | mode, hybrid pool stats, vector store that served |
| `history` | service (memory) | prior-shipment summary for this consignee + lane, or `None` |
| `diagnosis` | diagnose | root cause, summary, evidence list (incl. memory lines), policy citations |
| `recovery_options` | options | proposed options with code-computed scores |
| `recommended_option_id` | options | the highest-scoring option; the draft grounds on it |
| `draft` | draft | subject, body, claim packet (incl. diagnosis + options), citations |
| `verification` | verify | self-critique verdict: grounded?, issues, source (`llm` \| `checklist`) |
| `validation` | validate | guardrail pass/fail, errors, warnings |
| `repair_attempted` / `repaired` / `repair_attempts` | validate | bounded-repair bookkeeping |
| `original_validation` | validate | the first (failed) validation, preserved when repair ran |
| `autonomy` | human_approval | deterministic routing recommendation + reasons (never acted on) |
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
to the template. In provider mode the diagnosis is **agentic**: before
composing, the model runs a bounded tool loop (`tools_agent.py`) — it
may call `search_policies` (the retriever, its own wording),
`lane_history` (store history for the consignee/lane), `shipment_facts`
(the computed facts), and `carrier_history` (exception counts by type
for the carrier) — capped by `DIAGNOSIS_MAX_TOOL_CALLS` (default 4,
hard cap 6). Every executed call lands in the trace with a one-line
summary; a tool that fails degrades the diagnosis, never the run. The
default mode runs no loop and says so in the trace. An options node then proposes 2–3 recovery options
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

The retrieval **query is built from the shipment's own content** —
exception type and rationale, yes, but also the latest event text, the
condition notes, and the document field values. An earlier version
queried with type + rationale only, and a customer SOP written in
operational language ("cartons crushed at terminal inspection") never
ranked for the damage case it described, because its vocabulary shares
nothing with the classifier's rationale. The query now carries the
case's own words; a regression test pins the operational-vocabulary
scenario.

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

**Provider errors are translated; degradation is recorded, not silent.**
Every provider call (chat backends and the embeddings client) is wrapped
(`errors.py`): a transport failure becomes a `ProviderError` in the
product's own style naming the backend, the endpoint it tried, and the
likely fix — an unreachable Ollama says to start Ollama or check
`OLLAMA_BASE_URL`. SDK retries are disabled (`max_retries=0`) and the
timeout is `LLM_TIMEOUT_SECONDS`, so a dead endpoint fails in seconds;
retrying is a deployment decision in front of the service, not inside
it. Nodes that can degrade — extraction, the classification cross-check,
diagnosis, options, verification — fall back and record the translated
reason in the trace; drafting cannot degrade, so its failure is the
clean error on every surface (CLI/demo exit 1, API 502), never a raw
SDK traceback. Embeddings errors name the `RETRIEVER` value actually
set (`hybrid` says `hybrid`).

**Self-verification, then guardrails, then a bounded repair.** Drafting
is where generated language can drift from computed facts, so a verify
node critiques the draft against the verified facts and cited policies
before the guardrails run: an LLM critique call in provider mode
(versioned prompt in `prompts.py`, structured grounded/issues verdict),
a deterministic evidence checklist in default mode — same verdict
shape, honestly labelled `source="checklist"` (it checks cited policy
IDs against the retrieved set, hour figures against the computed delay,
shipment references against the case, and flags monetary figures no
fact supports). The verdict never edits the draft; it is evidence for
the approver and input to repair. On guardrail failure, a **bounded
repair loop** (`GUARDRAIL_REPAIR`, default on;
`GUARDRAIL_REPAIR_MAX_ATTEMPTS`, default 1, hard cap 3) redrafts with
the failure reasons and self-verification issues fed back into the
drafting prompt, then re-verifies and re-validates. The original
failure is preserved as `original_validation` and the attempt flagged
(`repair_attempted` / `repaired`). The guardrail rules themselves do
not change, and with repair off a failed draft behaves exactly as a
no-repair pipeline's does. Note the honest default-mode consequence:
the deterministic template redrafts identical words, so SYN-1013's
repair attempt fails identically — repair pays off with a real model
behind it, and the flags make which case happened unmistakable.

**Memory: prior shipments as diagnosis evidence.** Before diagnosis,
the service asks the store for prior analysed shipments
(`prior_shipments`, both stores; SQLite gained a `shipment_json`
column by additive migration) and summarises matches for the same
consignee and the same lane (origin → destination): counts of priors
that themselves had exceptions, plus their most recent types. The
summary reaches the diagnosis as evidence lines ("memory: 2 prior
exception(s) for this consignee in the stored history…"). A shipment
never counts itself; "none" priors do not count; empty history adds
no line. Scope is deliberately small — counts and recent types, not a
case-retrieval system; the store is the seam where a customer's real
history source would plug in.

**Autonomy recommendation: deterministic, printed, never acted on.**
Every result carries a routing recommendation computed by policy in
code (`autonomy.py`): eligible for auto-approval only when the
exception is `none` or severity `low`, the guardrails passed, the
cross-check did not disagree, and no repair was needed — each
criterion's outcome listed in `reasons`. It is printed on the result,
the console, the demo, and the claim packet, and nothing reads it to
skip the gate. Its job is to make the autonomy conversation concrete
for a customer: which of *their* case classes could move, and why the
rest stay human.

**Approvals persist; the API can be gated.** Analyses and decisions are
stored through a small store interface (`store.py`): SQLite on disk by
default (`STATE_DB_PATH`, default `.data/state.db`, git-ignored), so
the approval queue survives restarts; the in-memory implementation
remains as the test double (`STATE_DB_PATH=:memory:`). Setting
`API_KEY` turns on a shared-key check (`X-API-Key` header) for all data
endpoints; unset, the API is open and documented as a local-dev default.
Approve and reject take one decision-maker field, `actor` (legacy
`approver`/`reviewer` accepted), and the result returns `decided_by`.

**Guardrails as code, not prompts.** `guardrails.py` rejects drafts that
lack the shipment ID, lack policy citations on exception drafts, or contain
prohibited promise language ("we guarantee", "full refund", …). A draft
that fails validation cannot be approved through the service layer.

**The approval gate is structural.** There is no send/file/act node in the
graph at all. Approval in `service.py` records *who* approved and marks the
packet ready. The one opt-in exception lives in the service layer, not
the graph: **output routing**. When the operator sets
`ACTION_WEBHOOK_URL`, a successful approval POSTs the approved packet
JSON to that endpoint — the thin adapter to the customer's system of
choice — with a short timeout (`ACTION_WEBHOOK_TIMEOUT_SECONDS`,
default 5s). The outcome is recorded as `dispatch_status`
(`sent`/`failed`) on the result; a failed dispatch never undoes the
approval. Unset (the default), approval performs no external action.

## 5. Data model

Inputs are deliberately boring JSON: a shipment, its schedule, its latest
event text, condition notes, and documents — each document carrying its
text (`raw_text`) plus whatever structured fields the caller already has
(`quantity_units`, `weight_kg`, `consignee`, `sku`). The extract node
produces the typed fields from the text in provider mode and cross-checks
them against the provided ones. What this prototype does *not* do is OCR:
scanned documents need an OCR step in front of it in production, and
photo damage assessment (a VLM step) is out of scope entirely.

### Intake contract and normalisation

`ShipmentInput` fields (the README carries the same table):

| Field | Type | Required | Notes |
|---|---|---|---|
| `shipment_id` | string | yes | results, approvals, and history key on it |
| `origin` / `destination` | string | yes | the memory lane key is `origin -> destination` |
| `carrier` | string | no | default `Synthetic Carrier` |
| `customer_name` | string | no | consignee name for memory matching (fallback: a document's `consignee` field) |
| `service_level` | string | no | `standard` \| `priority` \| `critical` |
| `status` | string | no | default `in_transit` |
| `scheduled_delivery` / `estimated_delivery` | datetime | no | drive the computed delay |
| `latest_event` / `condition_notes` | string | no | feed classification AND the retrieval query |
| `documents` | array | no | each: `doc_type`, `document_id`, optional `raw_text`, optional `fields` object |

Canonical document types: `bol`, `invoice`, `tracking`, `delivery_note`,
`other`. Intake normalises caller dialects onto that vocabulary
(`bill_of_lading`/`BOL`/`Bill of Lading` → `bol`; `commercial_invoice` →
`invoice`; `tracking_note`/`tracking_update` → `tracking`;
`delivery_receipt`/`POD`/`proof_of_delivery` → `delivery_note`). An
unknown type is kept as supplied and flagged on the document record
(`doc_type_provided`, `doc_type_flagged`) — visible, never silently
reinterpreted. Document `fields` values coerce numbers to strings
(`120` → `"120"`) instead of failing validation; `null` values are
dropped. And the BOL↔invoice mismatch check announces a skip: with no
pair present, the result and trace carry *"no bill_of_lading/invoice
pair found — document mismatch check skipped"* — an empty mismatch list
always means "compared and agreed".

## 6. Failure modes

| Failure | Behaviour |
|---|---|
| Missing schedule data | delay_hours = None; classification falls back to event keywords; severity defaults to medium |
| BOL or invoice absent | mismatch check is skipped AND the result/trace carry the explicit warning ("no bill_of_lading/invoice pair found — document mismatch check skipped"); classification relies on event text |
| Unknown / aliased doc_type | aliases normalise to the canonical vocabulary; unknown types are kept and flagged (`doc_type_flagged`), never dropped |
| Numeric document field values | coerced to strings at intake — no 422 for a well-formed TMS payload |
| Novel exception phrasing | may classify as `none` — visible in evals as a miss; golden set grows from these |
| Retrieved policies irrelevant | draft still carries citations; approver sees policy titles and can reject |
| LLM backend unavailable at startup | explicit RuntimeError at construction naming the fix (missing key or missing `llm` extra); no silent fallback to the mock |
| Provider unreachable mid-run (e.g. Ollama down) | translated `ProviderError` naming backend, endpoint, likely fix; degradable nodes fall back with the reason in the trace; a drafting failure ends the run cleanly (CLI/demo exit 1, API 502). No SDK retries — fails within `LLM_TIMEOUT_SECONDS` |
| Provider client cannot even be constructed (e.g. a `NO_PROXY` entry like `[::1]` the HTTP library cannot parse) | translated `ProviderError` from the construction site itself — naming the backend, the endpoint, and the proxy variables to check — surfaced through the same CLI/API error paths, never a raw SDK traceback |
| LLM classification malformed/unavailable | cross-check records `rules_only` with the failure in its note; the rule result stands and the run continues |
| LLM extraction / diagnosis / options / verification failure | that node falls back to provided fields / the evidence template / template options / the deterministic checklist; the fallback is recorded in the trace and the run continues |
| Chroma installed but store broken | semantic retrieval falls back to in-memory cosine; `vector_store` in the result reports `memory` |
| `RETRIEVER=semantic`/`hybrid` with no embeddings route | explicit RuntimeError at construction naming the RETRIEVER value set: embeddings need OpenAI(-compatible) or the Ollama backend |
| Draft fails guardrails, repair on (default) | one bounded redraft with the failures fed back, re-verified + re-validated; result flags the attempt and preserves the original failure; if the redraft still fails, approval is blocked (SYN-1013 demonstrates the deterministic case) |
| Draft fails guardrails, `GUARDRAIL_REPAIR=off` | no attempt: approval is blocked with the exact errors surfaced — the pre-repair behaviour, unchanged |
| Approval webhook unreachable / non-2xx | `dispatch_status=failed` on the result; the approval itself stands |
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
   payload) onto `ShipmentInput` (§5 has the full field table); keep
   the rest of the graph untouched. Concretely: (a) point your payload
   at `POST /shipments/analyze` and map your identifiers to
   `shipment_id`, your lane fields to `origin`/`destination`, your
   latest status text to `latest_event`, and your condition remarks to
   `condition_notes` — those two text fields drive both classification
   and the retrieval query, so do not summarise them away; (b) attach
   documents with your own type names — the normaliser maps
   `bill_of_lading`/`BOL`/`Invoice`/`POD`-style dialects onto the
   canonical vocabulary (§5), and anything unrecognised is kept and
   flagged rather than rejected; (c) send document `fields` as your
   system has them — numbers are coerced to strings at intake; the
   mismatch check reads `quantity_units` and `weight_kg` from the
   `bol` and `invoice` documents; (d) if your source cannot supply a
   BOL/invoice pair, expect the explicit skipped-check warning on every
   result — that is the contract working, not an error.
4. **Retriever** — set `RETRIEVER=hybrid` (and install the `vectordb`
   extra for the local Chroma store) for embedding-based ranking of the
   shipped corpus, or replace the retriever with embeddings over your
   full policy library behind the same `Retriever` protocol.
5. **Exception types** — add the exceptions your operation actually sees
   (customs hold, address issue, …) as rules + golden cases.
6. **Action layer** — the seam already exists: set `ACTION_WEBHOOK_URL`
   to POST approved packets to your system, then grow it into the full
   approve → messaging/claims wiring behind the existing gate, with an
   audit log.

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
4. **Action layer:** the shipped approval webhook is the first adapter;
   production grows it into send-via-the-client's-messaging-system and
   claim filing via carrier portals/APIs — behind feature flags, with
   idempotency keys and rate limits, and `dispatch_status` grown into
   full delivery bookkeeping.
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
- Default-mode self-verification is a deterministic checklist (citations,
  figures, references, invented money) — it does not judge tone or subtle
  overclaiming; the provider-mode LLM critique is broader but is a model
  judging a draft.
- Memory is counts and recent exception types from this store — not a
  case-similarity search, and it starts empty on a fresh deployment.
- The autonomy recommendation is policy, not learning: its band
  (none/low) is a starting posture a customer tunes, and it never acts.
- The Docker local stack (agent + Ollama + Chroma) is reviewed but not
  build-verified — no Docker daemon in the development environment.
- No carrier/TMS integration and no claims filing — by design; the only
  outbound call that exists is the opt-in approval webhook.
