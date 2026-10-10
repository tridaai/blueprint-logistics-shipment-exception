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
- the store comes from :func:`default_store` (``DATABASE_URL`` for
  the Postgres system of record; ``STATE_DB_PATH`` selects the
  SQLite test double);
- the document object store comes from
  :func:`~shipment_agent.object_store.get_object_store`
  (``S3_BUCKET``) — resolved lazily by the service, like the
  checkpointer;
- the gate checkpointer comes from
  :func:`~shipment_agent.checkpoints.get_checkpointer`
  (``CHECKPOINTS`` / ``DATABASE_URL``) — resolved lazily by the
  service, like the backend;
- :func:`build_service_from_env` assembles the service the API runs,
  and accepts overrides for callers that must not persist (the CLI
  batch path passes its in-memory store and disables checkpoints).

The ports these objects satisfy are declared in ``ports.py``.
"""

from __future__ import annotations

from .checkpoints import get_checkpointer
from .model_backends import get_backend
from .object_store import get_object_store
from .ports import Checkpointer, ModelBackend, ObjectStore, Retriever, Store
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


def build_checkpointer() -> Checkpointer | None:
    """The gate checkpointer the environment configures (see checkpoints)."""
    return get_checkpointer()


def build_object_store() -> ObjectStore | None:
    """The document object store the environment configures (see object_store)."""
    return get_object_store()


def build_service_from_env(
    store: Store | None = None,
    checkpointer: Checkpointer | bool | None = None,
    object_store: ObjectStore | bool | None = None,
) -> ShipmentService:
    """Assemble the ShipmentService from configuration.

    ``store`` / ``checkpointer`` / ``object_store`` override the
    configured collaborators — used by the CLI batch path, which
    persists nothing (in-memory store, checkpoints off). Backend,
    retriever, and (by default) the checkpointer and object store
    stay lazily resolved inside the service (its documented
    behaviour: fresh per analysis unless a caller injects doubles).
    """
    return ShipmentService(
        store=store if store is not None else build_store(),
        checkpointer=checkpointer,
        object_store=object_store,
    )
