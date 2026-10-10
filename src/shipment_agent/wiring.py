"""Composition root: build the service and its collaborators from config.

Every entry point (API, CLI, demo) obtains its collaborators here —
this module is the ONE place that decides how environment
configuration becomes concrete objects:

- backend / retriever come from the existing factories
  (``MODEL_BACKEND`` / ``RETRIEVER`` + ``.env``), exposed as
  :func:`build_backend` / :func:`build_retriever` so callers keep
  their current resolution timing (the CLI and demo resolve once per
  invocation; the service resolves per analysis, which keeps batch
  runs free of shared usage counters — see ``service.py``);
- the store comes from :func:`default_store` (``STATE_DB_PATH``);
- :func:`build_service_from_env` assembles the service the API runs,
  and accepts a store override for callers that must not persist
  (the CLI batch path passes its in-memory store).

The ports these objects satisfy are declared in ``ports.py``.
"""

from __future__ import annotations

from .model_backends import get_backend
from .ports import ModelBackend, Retriever, Store
from .retriever import get_retriever
from .service import ShipmentService
from .store import default_store


def build_backend() -> ModelBackend:
    """The model backend the environment configures (see model_backends)."""
    return get_backend()


def build_retriever() -> Retriever:
    """The policy retriever the environment configures (see retriever)."""
    return get_retriever()


def build_store() -> Store:
    """The approval store the environment configures (see store)."""
    return default_store()


def build_service_from_env(store: Store | None = None) -> ShipmentService:
    """Assemble the ShipmentService from configuration.

    ``store`` overrides the configured store — used by the CLI batch
    path, which persists nothing. Backend and retriever deliberately
    stay lazily resolved inside the service (its documented behaviour:
    fresh per analysis unless a caller injects doubles).
    """
    return ShipmentService(store=store if store is not None else build_store())
