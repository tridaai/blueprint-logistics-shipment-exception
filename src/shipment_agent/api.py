"""FastAPI surface for the Shipment Exception Agent prototype.

Run:  uvicorn shipment_agent.api:app        (or `make serve`)
UI:   http://localhost:8000/                (minimal demo page, no build step)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .policies_data import POLICIES
from .samples import load_sample_shipments
from .schemas import AgentResult, ShipmentInput
from .service import ShipmentService

app = FastAPI(
    title="Trida AI Blueprint — Logistics Shipment Exception Agent",
    description="Reference prototype. Synthetic data only. Drafts stop at a human-approval gate; no external action is ever taken.",
    version="0.1.0",
)

service = ShipmentService()

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


@app.get("/evals/results")
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
    approver: str


class RejectRequest(BaseModel):
    reviewer: str
    reason: str = ""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "prototype": "trida-blueprint-logistics-shipment-exception"}


@app.get("/policies")
def list_policies() -> list[dict[str, str]]:
    return POLICIES


@app.get("/samples")
def list_samples() -> list[dict]:
    return load_sample_shipments()


@app.post("/shipments/analyze", response_model=AgentResult)
def analyze(shipment: ShipmentInput) -> AgentResult:
    return service.analyze(shipment)


@app.get("/shipments/{shipment_id}", response_model=AgentResult)
def get_result(shipment_id: str) -> AgentResult:
    result = service.get(shipment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Shipment not analyzed yet.")
    return result


@app.post("/shipments/{shipment_id}/approve", response_model=AgentResult)
def approve(shipment_id: str, request: ApproveRequest) -> AgentResult:
    try:
        return service.approve(shipment_id, approver=request.approver)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/shipments/{shipment_id}/reject", response_model=AgentResult)
def reject(shipment_id: str, request: RejectRequest) -> AgentResult:
    try:
        return service.reject(shipment_id, reviewer=request.reviewer, reason=request.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
