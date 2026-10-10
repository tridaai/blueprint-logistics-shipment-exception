"""PostgreSQL: connection handling and the schema migration runner.

PostgreSQL is the system of record for a production-shaped run: the
approval store (``store.py``), the LangGraph checkpointer
(``checkpoints.py``), and the pgvector policy embeddings
(``retriever.py``) all live in the one database ``DATABASE_URL``
points at. The compose stack ships the ``pgvector/pgvector`` image so
the ``vector`` extension is available.

Schema is owned by numbered SQL files in the repo-root ``migrations/``
directory, applied in order by :func:`run_migrations` and recorded in
a ``schema_migrations`` table, so a database is never half-migrated
silently. The runner executes at startup: the API lifespan calls it,
and the Postgres-backed collaborators call :func:`ensure_migrated`
(a per-process, per-URL once-guard) when they are constructed, so the
CLI and demo take the same path.

Without ``DATABASE_URL`` there is no database at all: the service
falls back to in-memory test doubles (see ``store.default_store``).
That mode exists for tests and offline smoke runs — nothing persists,
and the docs say so wherever it is mentioned.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .config import env_str, load_dotenv

_MIGRATED_URLS: set[str] = set()
_MIGRATE_LOCK = threading.Lock()


def database_url() -> str | None:
    """The configured ``DATABASE_URL``, or None (test-double mode)."""
    load_dotenv()
    return env_str("DATABASE_URL")


def migrations_dir() -> Path:
    """Locate the ``migrations/`` directory (repo root, or the image's
    copy — the Dockerfile places it next to ``src/``)."""
    configured = env_str("MIGRATIONS_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "migrations"


def connect(url: str | None = None, *, autocommit: bool = False):
    """One psycopg 3 connection to the configured database.

    Imported lazily so the offline/test-double paths never pay for —
    or fail on — the driver. Callers own the connection and close it
    (the store/checkpointer pattern is one connection per call, like
    the SQLite doubles they replace).
    """
    import psycopg

    resolved = url or database_url()
    if not resolved:
        raise RuntimeError(
            "DATABASE_URL is not set — there is no PostgreSQL to connect to. "
            "Set DATABASE_URL (see .env.example) or run on the in-memory "
            "test doubles."
        )
    return psycopg.connect(resolved, autocommit=autocommit)


def ping(url: str | None = None) -> bool:
    """True when the configured database answers ``SELECT 1``."""
    try:
        with connect(url) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _split_statements(sql: str) -> list[str]:
    """Split a migration file into statements.

    Migration files are deliberately plain: one statement per ``;``,
    no semicolons inside string literals or procedural bodies, so this
    stays a line-level split rather than a SQL parser.
    """
    statements: list[str] = []
    buffer: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        buffer.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(buffer).strip().rstrip(";"))
            buffer = []
    if buffer:  # trailing statement without a semicolon still runs
        statements.append("\n".join(buffer).strip())
    return [s for s in statements if s]


def _migration_files() -> list[tuple[str, Path]]:
    files = []
    for path in sorted(migrations_dir().glob("[0-9]*.sql")):
        version = path.name.split("_", 1)[0]
        files.append((version, path))
    return files


def run_migrations(url: str | None = None) -> list[str]:
    """Apply every unapplied migration, in version order.

    Returns the versions applied by this call (empty when the schema
    is already current — the runner is idempotent and safe to call at
    every startup). The LangGraph checkpointer's own tables are not
    here: the official Postgres saver creates them via its ``setup()``
    (see ``checkpoints.py``). What lives here is this application's
    schema: the approvals store, the pgvector embeddings table.
    """
    resolved = url or database_url()
    if not resolved:
        raise RuntimeError("DATABASE_URL is not set — nothing to migrate.")
    applied_now: list[str] = []
    with connect(resolved) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version TEXT PRIMARY KEY, "
            "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        already = {
            row[0] for row in conn.execute("SELECT version FROM schema_migrations")
        }
        for version, path in _migration_files():
            if version in already:
                continue
            for statement in _split_statements(path.read_text(encoding="utf-8")):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (version,)
            )
            applied_now.append(version)
        conn.commit()
    return applied_now


def ensure_migrated(url: str | None = None) -> None:
    """Run migrations at most once per process per database URL.

    The startup path for the Postgres-backed collaborators: whichever
    is constructed first migrates; the rest find it done. Failures are
    loud — a production run against an unmigratable database should
    not start half-configured.
    """
    resolved = url or database_url()
    if not resolved:
        return
    with _MIGRATE_LOCK:
        if resolved in _MIGRATED_URLS:
            return
        run_migrations(resolved)
        _MIGRATED_URLS.add(resolved)
