"""Access to the bundled synthetic sample shipments.

The samples ship *inside* the package (``shipment_agent/data/``) so the
CLI demo, the API, and the web UI work identically from a source checkout
and from a pip-installed copy. ``data/sample/sample_shipments.json`` at
the repo root is the same file for browsing on GitHub; a test asserts the
two copies never drift.
"""

from __future__ import annotations

import json
from importlib import resources

from .schemas import ShipmentInput


def load_sample_shipments() -> list[dict]:
    """Return the bundled synthetic sample shipments as raw dicts."""
    packaged = resources.files("shipment_agent").joinpath("data", "sample_shipments.json")
    raw = packaged.read_text(encoding="utf-8")
    return json.loads(raw)


def sample_shipment_models() -> list[ShipmentInput]:
    return [ShipmentInput.model_validate(item) for item in load_sample_shipments()]
