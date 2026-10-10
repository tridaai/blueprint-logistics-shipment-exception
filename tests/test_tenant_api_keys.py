"""Per-tenant API keys: a key opens only its own tenant partition.

Round 5 partitioned the data but left ``X-Tenant-ID`` a trusted
header: under the single shared ``API_KEY``, any key holder could
name any tenant. Per-tenant keys (``TENANT_API_KEYS`` pairs, or one
``API_KEY_<TENANT>`` variable per tenant) close that gap: once any
per-tenant key is configured, the key presented must be the claimed
tenant's own — another tenant's key (or the shared key aimed at a
named tenant) is a 403, a missing/unknown key a 401, and the shared
``API_KEY`` keeps working for the default tenant only. With no
per-tenant configuration the legacy model is unchanged: one optional
shared key, the header trusted — the docs say both parts plainly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shipment_agent import api as api_module
from shipment_agent.config import tenant_api_key, tenant_api_keys
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.service import ShipmentService
from shipment_agent.store import InMemoryStore

SHIPMENT = {
    "shipment_id": "KEY-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

_KEY_VARS = (
    "API_KEY",
    "TENANT_API_KEYS",
    "TENANT_ID",
    "API_KEY_ACME",
    "API_KEY_GLOBEX",
    "API_KEY_ACME_RETAIL",
)


@pytest.fixture(autouse=True)
def clean_key_env(monkeypatch):
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.service.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.api.load_dotenv", lambda *a, **k: None)


@pytest.fixture()
def client(monkeypatch):
    service = ShipmentService(
        backend=MockModelBackend(),
        retriever=KeywordRetriever(),
        store=InMemoryStore(),
        checkpointer=False,
    )
    monkeypatch.setattr(api_module, "service", service)
    return TestClient(api_module.app)


def _samples(client, key=None, tenant=None):
    headers = {}
    if key is not None:
        headers["X-API-Key"] = key
    if tenant is not None:
        headers["X-Tenant-ID"] = tenant
    return client.get("/samples", headers=headers)


# ---------------------------------------------------------------------------
# Configuration parsing
# ---------------------------------------------------------------------------


def test_tenant_api_keys_parsing(monkeypatch):
    monkeypatch.setenv("TENANT_API_KEYS", "acme:key-one, globex : key-two")
    assert tenant_api_keys() == {"acme": "key-one", "globex": "key-two"}
    # Malformed pairs are skipped, never half-trusted.
    monkeypatch.setenv("TENANT_API_KEYS", "acme:key-one,no-colon,:orphan,globex:")
    assert tenant_api_keys() == {"acme": "key-one"}


def test_per_tenant_env_var_names_come_from_the_tenant_id(monkeypatch):
    monkeypatch.setenv("API_KEY_ACME_RETAIL", "retail-key")
    assert tenant_api_key("acme-retail") == "retail-key"
    assert tenant_api_key("acme") is None
    # TENANT_API_KEYS wins over the per-tenant variable.
    monkeypatch.setenv("TENANT_API_KEYS", "acme-retail:map-key")
    assert tenant_api_key("acme-retail") == "map-key"


# ---------------------------------------------------------------------------
# The per-tenant model
# ---------------------------------------------------------------------------


def test_a_tenants_own_key_opens_its_partition(client, monkeypatch):
    monkeypatch.setenv("TENANT_API_KEYS", "acme:acme-key,globex:globex-key")
    assert _samples(client, key="acme-key", tenant="acme").status_code == 200
    assert _samples(client, key="globex-key", tenant="globex").status_code == 200


def test_another_tenants_key_is_a_403_not_a_401(client, monkeypatch):
    monkeypatch.setenv("TENANT_API_KEYS", "acme:acme-key,globex:globex-key")
    response = _samples(client, key="globex-key", tenant="acme")
    assert response.status_code == 403
    assert "acme" in response.json()["detail"]


def test_missing_and_unknown_keys_are_401(client, monkeypatch):
    monkeypatch.setenv("TENANT_API_KEYS", "acme:acme-key")
    assert _samples(client, tenant="acme").status_code == 401
    assert _samples(client, key="not-a-key", tenant="acme").status_code == 401


def test_the_shared_key_keeps_working_for_the_default_tenant_only(
    client, monkeypatch
):
    monkeypatch.setenv("API_KEY", "shared-key")
    monkeypatch.setenv("TENANT_API_KEYS", "acme:acme-key")
    # Default tenant (no header): the shared key still works.
    assert _samples(client, key="shared-key").status_code == 200
    # A named tenant: the shared key is a real key in the wrong place.
    assert _samples(client, key="shared-key", tenant="acme").status_code == 403
    # And the default tenant's door is not opened by acme's key either.
    assert _samples(client, key="acme-key").status_code == 403


def test_per_tenant_env_var_form_binds_the_tenant(client, monkeypatch):
    monkeypatch.setenv("API_KEY_ACME", "acme-key")
    assert _samples(client, key="acme-key", tenant="acme").status_code == 200
    assert _samples(client, key="acme-key", tenant="globex").status_code == 403
    # Per-tenant mode is on: an unkeyed tenant is closed, not open.
    assert _samples(client, tenant="globex").status_code == 401


def test_per_tenant_keys_gate_the_data_not_just_the_door(client, monkeypatch):
    """End to end: acme's key writes and reads acme's partition; the
    same key cannot even claim globex's partition to look for it."""
    monkeypatch.setenv("TENANT_API_KEYS", "acme:acme-key,globex:globex-key")
    created = client.post(
        "/shipments/analyze",
        json=SHIPMENT,
        headers={"X-API-Key": "acme-key", "X-Tenant-ID": "acme"},
    )
    assert created.status_code == 200
    own = client.get(
        "/shipments/KEY-1",
        headers={"X-API-Key": "acme-key", "X-Tenant-ID": "acme"},
    )
    assert own.status_code == 200
    wrong_key = client.get(
        "/shipments/KEY-1",
        headers={"X-API-Key": "globex-key", "X-Tenant-ID": "acme"},
    )
    assert wrong_key.status_code == 403
    right_key_wrong_claim = client.get(
        "/shipments/KEY-1",
        headers={"X-API-Key": "globex-key", "X-Tenant-ID": "globex"},
    )
    assert right_key_wrong_claim.status_code == 404  # globex's own door, empty room


# ---------------------------------------------------------------------------
# The legacy models are unchanged without per-tenant configuration
# ---------------------------------------------------------------------------


def test_shared_key_model_still_trusts_the_tenant_header(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "shared-key")
    assert _samples(client, key="shared-key", tenant="acme").status_code == 200
    assert _samples(client, key="shared-key", tenant="globex").status_code == 200
    assert _samples(client, tenant="acme").status_code == 401
    assert _samples(client, key="wrong", tenant="acme").status_code == 401


def test_no_keys_configured_stays_open(client):
    assert _samples(client).status_code == 200
    assert _samples(client, tenant="acme").status_code == 200


def test_health_stays_open_under_per_tenant_keys(client, monkeypatch):
    monkeypatch.setenv("TENANT_API_KEYS", "acme:acme-key")
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200
