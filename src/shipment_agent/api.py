"""FastAPI surface for the Shipment Exception Agent prototype.

Run:  uvicorn shipment_agent.api:app        (or `make serve`)
UI:   http://localhost:8000/                (minimal demo page, no build step)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import env_str, load_dotenv, silence_langchain_deprecation_warnings
from .db import database_url, ensure_migrated
from .db import ping as db_ping
from .observability import configure_json_logging, new_request_id, request_id_ctx

# Must precede the service import (which loads the graph/langgraph).
silence_langchain_deprecation_warnings()

from .audit import audit_rows, render_audit_csv
from .errors import ProviderError
from .events import CallbackSink, RunEvent
from .metrics import compute_metrics, render_prometheus
from .samples import load_sample_shipments
from .schemas import AgentResult, ShipmentInput
from .wiring import build_service_from_env

# Load the repo-root .env at startup (real environment variables win), so
# MODEL_BACKEND / RETRIEVER / API keys can live in the file — see .env.example.
load_dotenv()


def require_api_key(
    x_api_key: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> None:
    """The API-key gate: two models, chosen by configuration.

    **Shared-key model** (no per-tenant keys configured): when
    ``API_KEY`` is set, every data endpoint requires the matching
    ``X-API-Key`` header; when it is unset the API is open — the
    local-dev default, stated here and in the docs rather than
    implied. Under this model the ``X-Tenant-ID`` header is a
    *trusted claim*: the key authenticates the caller, not the
    tenant, so any key holder can name any partition. That is the
    single-client deployment shape, and it is only as strong as the
    network segment the API sits on.

    **Per-tenant model** (``TENANT_API_KEYS`` names a tenant, or any
    ``API_KEY_<TENANT>`` variable is set — see ``config.py``): a key
    opens only its own tenant's partition. The claimed tenant is
    resolved exactly like everywhere else (header, then ``TENANT_ID``,
    then the default tenant) and the presented key must be *that
    tenant's* key. The shared ``API_KEY`` keeps working for the
    **default tenant only**. A real key presented for the wrong
    tenant is a 403 (authenticated, wrong partition); a missing or
    unrecognised key is a 401.

    The console page and /health stay open either way; the console
    carries API-key and tenant fields for the gated calls.
    """
    from .config import (
        DEFAULT_TENANT_ID,
        known_api_keys,
        per_tenant_keys_configured,
        tenant_api_key,
    )
    from .service import resolve_tenant_id

    shared = env_str("API_KEY")
    if not per_tenant_keys_configured():
        if not shared:
            return
        if x_api_key != shared:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key (send the X-API-Key header).",
            )
        return
    tenant = resolve_tenant_id(x_tenant_id)
    expected = tenant_api_key(tenant)
    if expected is not None and x_api_key == expected:
        return
    if (
        expected is None
        and tenant == DEFAULT_TENANT_ID
        and shared
        and x_api_key == shared
    ):
        return  # the shared key's one remaining home: the default tenant
    if x_api_key and x_api_key in known_api_keys():
        raise HTTPException(
            status_code=403,
            detail=(
                f"That API key is not valid for tenant {tenant!r} — a key "
                "opens only its own tenant partition."
            ),
        )
    raise HTTPException(
        status_code=401,
        detail=(
            "Invalid or missing API key — send the X-API-Key issued for "
            "the tenant named by X-Tenant-ID."
        ),
    )


_AUTH = [Depends(require_api_key)]

logger = logging.getLogger("shipment_agent.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: JSON logging, schema migrations. Shutdown: drain.

    With ``DATABASE_URL`` set, migrations run before the first
    request is served — a production process never starts against a
    half-migrated database. Shutdown closes the checkpointer's
    connection pool; uvicorn's own SIGTERM handling (stop accepting,
    finish in-flight requests, then this lifespan exit) provides the
    graceful drain.
    """
    configure_json_logging(env_str("LOG_LEVEL", "INFO") or "INFO")
    if database_url():
        ensure_migrated()
        logger.info("startup: migrations current, database configured")
    else:
        logger.info("startup: no DATABASE_URL — in-memory test doubles")
    yield
    try:
        # Only an already-resolved checkpointer can hold a pool —
        # never construct one just to shut down.
        if service._checkpointer_resolved:
            pool = getattr(service._resolved_checkpointer, "conn", None)
            if pool is not None and hasattr(pool, "close"):
                pool.close()
    except Exception:  # shutdown must not fail on cleanup
        pass
    logger.info("shutdown: drained")


app = FastAPI(
    title="Trida AI Blueprint — Logistics Shipment Exception Agent",
    description="Reference prototype. Synthetic data only. Drafts stop at a human-approval gate; no external action is ever taken.",
    version="0.1.0",
    lifespan=lifespan,
)

service = build_service_from_env()


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign/propagate X-Request-ID; one JSON access line per request."""
    request_id = request.headers.get("x-request-id") or new_request_id()
    token = request_id_ctx.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "event": "http_request",
            },
        )
        return response
    finally:
        request_id_ctx.reset(token)


@app.exception_handler(ProviderError)
def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    """A provider failure is a clean 502 with the actionable, translated
    message (backend, endpoint, likely fix) — never a stack dump."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"
_BRAND_DIR = _STATIC_DIR / "brand"

# Branded static assets (logo, favicon) for the demo console.
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> FileResponse:
    return FileResponse(_BRAND_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> FileResponse:
    return FileResponse(_BRAND_DIR / "favicon.ico", media_type="image/x-icon")


@app.get("/evals/results", dependencies=_AUTH)
def eval_results() -> dict:
    """Latest eval summary, as emitted by ``evals/run_evals.py``.

    Looked up in the packaged copy first (present in installs/Docker),
    then in a repo checkout's ``evals/results.json``. 404 when no eval
    run has been recorded yet — the UI shows a 'run evals' state.
    """
    candidates = [
        Path(__file__).parent / "data" / "eval_results.json",
        Path.cwd() / "evals" / "results.json",
        Path(__file__).resolve().parents[2] / "evals" / "results.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise HTTPException(
        status_code=404,
        detail="No eval results yet. Run: python evals/run_evals.py",
    )


class ApproveRequest(BaseModel):
    # One name for the decision-maker across both endpoints: `actor`.
    # `approver` stays accepted as a legacy alias.
    actor: str | None = None
    approver: str | None = None
    reason: str = ""  # optional — stored, and surfaces as feedback on later cases


class RejectRequest(BaseModel):
    actor: str | None = None
    reviewer: str | None = None  # legacy alias of `actor`
    reason: str = ""


def _decision_actor(request: ApproveRequest | RejectRequest) -> str:
    actor = request.actor or getattr(request, "approver", None) or getattr(request, "reviewer", None)
    if not actor:
        raise HTTPException(
            status_code=422,
            detail="Provide 'actor' — the decision-maker's name ('approver'/'reviewer' are accepted legacy aliases).",
        )
    return actor


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "prototype": "trida-blueprint-logistics-shipment-exception"}


@app.get("/readiness")
def readiness() -> JSONResponse:
    """Readiness = the process can serve real work.

    With ``DATABASE_URL`` configured that means the system of record
    answers a ping — a replica that cannot reach its database is not
    ready and says so (503), so a load balancer drains it. Without a
    database the process serves on in-memory test doubles and is
    ready by definition; the payload names that mode honestly.
    """
    if database_url():
        if db_ping():
            return JSONResponse(
                status_code=200,
                content={"status": "ready", "database": "postgres"},
            )
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "database": "unreachable"},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "database": "in-memory test doubles"},
    )


# Tenancy: every data endpoint accepts an ``X-Tenant-ID`` header
# naming the tenant partition it operates on (resolution: header,
# else the TENANT_ID environment default, else the default tenant —
# see service.resolve_tenant_id). A read for a shipment that lives
# in another tenant's partition is a 404, exactly as if it did not
# exist: partitions are invisible to each other, not just filtered.
# Whether the header is a trusted claim or an authenticated one is
# the auth model's business — per-tenant keys (TENANT_API_KEYS /
# API_KEY_<TENANT>) bind the partition to the caller's credential;
# see require_api_key above.


@app.get("/metrics")
def metrics(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> PlainTextResponse:
    """Prometheus text exposition, computed live from the store:
    runs, decisions, guardrail failures, latency, tokens, estimated
    cost (see ``metrics.py``). Open like /health — aggregates only,
    no shipment content; scrapers live on a trusted network segment
    in any real deployment. The aggregates cover the resolved
    tenant's partition (``X-Tenant-ID``), so a multi-tenant
    deployment scrapes per tenant."""
    from .service import resolve_tenant_id

    records = service._get_store().records(tenant_id=resolve_tenant_id(x_tenant_id))
    payload = render_prometheus(compute_metrics(records))
    return PlainTextResponse(payload, media_type="text/plain; version=0.0.4")


@app.get("/audit/export", dependencies=_AUTH)
def audit_export(
    format: str = Query(default="json"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
):
    """The audit trail: one row per analysed shipment — conclusion,
    decider, reason, timestamps — as JSON (default) or CSV
    (``?format=csv``). A projection of the caller's tenant partition
    of the store, the system of record; see ``audit.py``."""
    from .service import resolve_tenant_id

    rows = audit_rows(
        service._get_store().records(tenant_id=resolve_tenant_id(x_tenant_id))
    )
    if format == "json":
        return {"count": len(rows), "decisions": rows}
    if format == "csv":
        return PlainTextResponse(render_audit_csv(rows), media_type="text/csv")
    raise HTTPException(
        status_code=422, detail="format must be 'json' or 'csv'."
    )


@app.get("/queue", dependencies=_AUTH)
def approval_queue(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict:
    """The approval queue: shipments awaiting a human decision,
    sorted by severity then age (oldest first), each with the flags
    an approver scans for — cross-check disagreement, guardrail
    repair, reviewer block, failing guardrails, information needed,
    auto-approval eligibility — plus its age bucket and SLA view
    (``sla_hours`` budget for its severity, ``sla_breach`` when the
    wait has blown it). The response also carries the queue's own
    ``summary`` (depth, breaches, age/severity mix) and the active
    ``sla_hours`` budgets (``QUEUE_SLA_HOURS_<SEVERITY>``). The
    approver's worklist, computed from the caller's tenant partition
    of the store (see ``insights.py``). Each item also carries its
    ladder stage (``sla_stage``: within_budget / breach / escalated,
    against ``sla_escalation_factor`` × its budget), and the summary
    counts escalations beside breaches."""
    from .insights import (
        queue_summary,
        sla_escalation_factor_from_env,
        sla_thresholds_from_env,
    )

    queue_items = service.approval_queue(tenant_id=x_tenant_id)
    return {
        "count": len(queue_items),
        "queue": queue_items,
        "summary": queue_summary(queue_items),
        "sla_hours": sla_thresholds_from_env(),
        "sla_escalation_factor": sla_escalation_factor_from_env(),
    }


@app.post("/queue/sla-sweep", dependencies=_AUTH)
def sla_breach_sweep(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict:
    """Run one SLA breach sweep over the caller's tenant queue.

    The sweep is the documented observer of the queue's SLA flags:
    for every awaiting shipment whose wait has blown its severity's
    budget and which has not yet fired, it delivers one signed
    ``sla_breach`` webhook event (opt-in: ``SLA_BREACH_WEBHOOK=on``,
    to ``SLA_BREACH_WEBHOOK_URL`` or the approval webhook URL) and
    ledgers the attempt on the record. The ladder climbs in order:
    a breach reported in rung-1 territory that keeps aging past
    ``QUEUE_SLA_ESCALATION_FACTOR`` × its budget re-fires on a later
    sweep as a signed ``sla_escalation`` event with the wait
    duration (each entry names its ``rung``). Dedupe is per rung
    per shipment per analysis — a second sweep fires nothing for a
    rung already fired. The response lists what this sweep
    observed: ``outcome`` is ``sent`` / ``failed`` for fired
    events, ``disabled`` / ``not_configured`` when the channel is
    off (observed, marked nothing). Operators running the
    multi-tenant sweep use the CLI (``shipment-agent sla-sweep``);
    this endpoint is the single-tenant shape of the same sweep."""
    entries = service.sla_breach_sweep(tenant_id=x_tenant_id)
    return {"count": len(entries), "events": entries}


@app.get("/carriers/scorecards", dependencies=_AUTH)
def carrier_scorecards(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict:
    """Carrier scorecards: per-carrier aggregates over the caller's
    tenant partition — shipments, exception mix and rate, damage
    rate, and the human decision record (approvals / rejections /
    approval rate), busiest carrier first."""
    cards = service.carrier_scorecards(tenant_id=x_tenant_id)
    return {"count": len(cards), "carriers": cards}


@app.get("/carriers/{carrier}/scorecard", dependencies=_AUTH)
def carrier_scorecard(
    carrier: str,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict:
    """One carrier's scorecard (404 when the carrier has no stored
    history in the caller's tenant partition)."""
    card = service.carrier_scorecard(carrier, tenant_id=x_tenant_id)
    if card is None:
        raise HTTPException(
            status_code=404, detail=f"No stored history for carrier: {carrier}"
        )
    return card


@app.get("/policies", dependencies=_AUTH)
def list_policies(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> list[dict[str, str]]:
    """The policy corpus the caller's tenant retrieves from: the
    shared documents plus that tenant's own tagged SOPs (their
    ``tenant_id`` field says which). Another tenant's documents are
    not listed — the listing is the corpus, and the corpus is the
    partition."""
    from .policies_data import policies_for_tenant
    from .service import resolve_tenant_id

    return policies_for_tenant(resolve_tenant_id(x_tenant_id))


@app.get("/samples", dependencies=_AUTH)
def list_samples() -> list[dict]:
    return load_sample_shipments()


@app.post("/shipments/analyze", response_model=AgentResult, dependencies=_AUTH)
def analyze(
    shipment: ShipmentInput,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> AgentResult:
    """Analyse one shipment. Send an ``Idempotency-Key`` header to
    make the call safe to retry: a repeat with the same key (and the
    same shipment id) returns the stored run — flagged
    ``idempotent_replay`` in the body and with an
    ``X-Idempotent-Replay: true`` response header — instead of
    running the pipeline (and spending model calls) again. The run
    is recorded in the caller's tenant partition (``X-Tenant-ID``)."""
    result = service.analyze(
        shipment, idempotency_key=idempotency_key, tenant_id=x_tenant_id
    )
    if result.idempotent_replay:
        response.headers["X-Idempotent-Replay"] = "true"
    return result


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/shipments/analyze/stream", dependencies=_AUTH)
def analyze_stream(
    shipment: ShipmentInput,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> StreamingResponse:
    """The same analysis as ``POST /shipments/analyze``, streamed.

    The run executes on a worker thread; its structured events (see
    ``events.py``) are forwarded as Server-Sent Events as they happen.
    The final event is ``run_completed`` and carries the full result
    JSON under ``result`` — the same payload the non-stream endpoint
    returns. A failed run ends the stream with a ``run_failed`` event
    carrying the clean, translated error message (no stack dump).

    With an ``Idempotency-Key`` that already has a stored run, there
    is nothing to stream: the response is the single ``run_completed``
    event carrying the stored (replay-flagged) result.
    """
    replay = service.idempotent_result(
        idempotency_key, shipment.shipment_id, tenant_id=x_tenant_id
    )
    if replay is not None:

        def replay_stream():
            final = {
                "type": "run_completed",
                "shipment_id": replay.shipment_id,
                "node": None,
                "duration_ms": None,
                "detail": {"status": replay.approval_status},
                "result": replay.model_dump(mode="json"),
            }
            yield _sse(final)

        return StreamingResponse(replay_stream(), media_type="text/event-stream")

    events_queue: queue.Queue = queue.Queue()
    sink = CallbackSink(events_queue.put)
    outcome: dict = {}

    def work() -> None:
        try:
            outcome["result"] = service.analyze(
                shipment,
                event_sink=sink,
                idempotency_key=idempotency_key,
                tenant_id=x_tenant_id,
            )
        except Exception as exc:  # surfaced as the run_failed event below
            outcome["error"] = exc
        finally:
            events_queue.put(None)  # sentinel: the run is over

    threading.Thread(target=work, daemon=True).start()

    def generate():
        completed: RunEvent | None = None
        while True:
            event = events_queue.get()
            if event is None:
                break
            if event.type == "run_completed":
                completed = event  # held back: it goes out last, with the result
                continue
            yield _sse(event.to_dict())
        if "error" in outcome:
            yield _sse(
                {
                    "type": "run_failed",
                    "shipment_id": shipment.shipment_id,
                    "node": None,
                    "duration_ms": None,
                    "detail": {"error": str(outcome["error"])},
                }
            )
            return
        final = (
            completed.to_dict()
            if completed is not None
            else {
                "type": "run_completed",
                "shipment_id": shipment.shipment_id,
                "node": None,
                "duration_ms": None,
                "detail": {"status": outcome["result"].approval_status},
            }
        )
        final["result"] = outcome["result"].model_dump(mode="json")
        yield _sse(final)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/shipments/{shipment_id}", response_model=AgentResult, dependencies=_AUTH)
def get_result(
    shipment_id: str,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> AgentResult:
    """One shipment's stored run, from the caller's tenant partition
    — a shipment that exists only under another tenant is a 404."""
    result = service.get(shipment_id, tenant_id=x_tenant_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Shipment not analyzed yet.")
    return result


@app.post("/shipments/{shipment_id}/approve", response_model=AgentResult, dependencies=_AUTH)
def approve(
    shipment_id: str,
    request: ApproveRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> AgentResult:
    """Record the human approval. Send an ``Idempotency-Key``
    header to make the decision safe to retry: a repeat approval
    under the same key returns the recorded decision (flagged
    ``idempotent_replay``, with an ``X-Idempotent-Replay: true``
    header) instead of erroring — and never dispatches the webhook
    a second time. The *opposite* decision under a spent key is a
    409; a second decision under a different (or no) key stays a
    422, as before."""
    from .service import DecisionConflictError

    try:
        result = service.approve(
            shipment_id,
            approver=_decision_actor(request),
            reason=request.reason,
            tenant_id=x_tenant_id,
            idempotency_key=idempotency_key,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DecisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result.idempotent_replay:
        response.headers["X-Idempotent-Replay"] = "true"
    return result


@app.post("/shipments/{shipment_id}/reject", response_model=AgentResult, dependencies=_AUTH)
def reject(
    shipment_id: str,
    request: RejectRequest,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> AgentResult:
    """Record the human rejection — the mirror of approve,
    including its ``Idempotency-Key`` contract (replay returns the
    recorded rejection; an approval attempted under the rejection's
    spent key is a 409)."""
    from .service import DecisionConflictError

    try:
        result = service.reject(
            shipment_id,
            reviewer=_decision_actor(request),
            reason=request.reason,
            tenant_id=x_tenant_id,
            idempotency_key=idempotency_key,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DecisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result.idempotent_replay:
        response.headers["X-Idempotent-Replay"] = "true"
    return result


@app.get("/shipments/{shipment_id}/dispatch", dependencies=_AUTH)
def dispatch_ledger(
    shipment_id: str,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> dict:
    """The approval webhook's delivery ledger for one shipment:
    every attempt (timestamp, outcome, HTTP status, signature id,
    error), the current dispatch status, and the retry bookkeeping
    (attempts used/remaining, next-retry due time). Empty attempts
    when no webhook is configured — the default no-external-action
    mode ledgers nothing because nothing was attempted."""
    ledger = service.dispatch_ledger(shipment_id, tenant_id=x_tenant_id)
    if ledger is None:
        raise HTTPException(status_code=404, detail="Shipment not analyzed yet.")
    return ledger


class DispatchRetryRequest(BaseModel):
    # Operator override: retry now even though the recorded backoff
    # has not elapsed (the endpoint was down, it is fixed, send it).
    force: bool = False


@app.post(
    "/shipments/{shipment_id}/dispatch/retry",
    response_model=AgentResult,
    dependencies=_AUTH,
)
def retry_dispatch(
    shipment_id: str,
    request: DispatchRetryRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
) -> AgentResult:
    """Retry a failed approval-webhook delivery — one bounded attempt.

    Refused (422) when the shipment is not approved, no webhook is
    configured, the delivery already succeeded, the attempt budget
    is exhausted, or the backoff has not elapsed and ``force`` is
    not set. The attempt joins the delivery ledger either way."""
    try:
        return service.retry_dispatch(
            shipment_id, force=request.force, tenant_id=x_tenant_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
