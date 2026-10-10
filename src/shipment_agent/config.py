"""Runtime configuration: environment variables + an optional ``.env`` file.

All provider/API configuration comes from environment variables — nothing
is hardcoded beyond the defaults documented in ``.env.example`` and the
README's configuration table.

At startup the entry points (API, CLI, demo) call :func:`load_dotenv`,
which reads a ``.env`` file when one exists — first in the current working
directory, then at the repository root. The parser is deliberately small
and built in (no third-party dependency): ``KEY=VALUE`` lines, ``#``
comments, optional surrounding quotes, optional ``export`` prefix.

Precedence rule: a variable already set in the real process environment
always wins. ``.env`` only fills in variables that are not set, so CI and
production environments override the file.
"""

from __future__ import annotations

import os
from pathlib import Path


def _candidate_paths() -> list[Path]:
    return [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",  # repository root
    ]


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    if not key:
        return None
    return key, value


def load_dotenv(path: Path | None = None) -> Path | None:
    """Load ``.env`` values into ``os.environ`` without overriding it.

    Returns the path that was loaded, or ``None`` when no file exists.
    Safe to call more than once: keys already present in the environment
    (from the process or an earlier load) are never overwritten.
    """
    candidates = [path] if path is not None else _candidate_paths()
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            parsed = _parse_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if key not in os.environ:
                os.environ[key] = value
        return candidate
    return None


def silence_langchain_deprecation_warnings() -> None:
    """Entry-point hygiene: hide langgraph's `allowed_objects` warning.

    LangGraph currently emits a LangChainPendingDeprecationWarning about
    a default that will change in a future version. A CLI/demo user can
    do nothing about it, so the entry points (CLI, demo, API) filter it
    by message. Two quirks make this less trivial than it looks: the
    warning fires while langgraph imports (so entry points call this
    before importing the graph), and langchain_core re-registers its own
    "default" filters at import time, jumping ahead of any filter set
    earlier — so langchain_core is imported first here and the ignore
    is registered after it. Library code never touches global warning
    state; this runs only where a human is looking at the output.
    """
    import warnings

    try:
        import langchain_core  # noqa: F401 — surfaces its filters first
    except ImportError:
        pass
    warnings.filterwarnings("ignore", message=r".*allowed_objects.*")


# Every record belongs to exactly one tenant (store.py partitions on
# it). Requests name theirs with the ``X-Tenant-ID`` header; a process
# serving a single client sets ``TENANT_ID`` instead; with neither,
# everything lives in this default tenant — the single-tenant
# deployment shape the earlier rounds shipped.
DEFAULT_TENANT_ID = "default"


def tenant_api_keys() -> dict[str, str]:
    """The per-tenant API keys from ``TENANT_API_KEYS``: tenant → key.

    Format: comma-separated ``tenant:key`` pairs —
    ``TENANT_API_KEYS=acme:key-one,globex:key-two``. Tenant ids
    therefore cannot contain ``:`` or ``,`` (they are partition
    names, not prose); whitespace around pairs is ignored, and a
    malformed pair (no colon, an empty side) is skipped rather than
    half-trusted. An empty dict means the variable is not configured.
    """
    load_dotenv()
    keys: dict[str, str] = {}
    raw = env_str("TENANT_API_KEYS")
    if not raw:
        return keys
    for pair in raw.split(","):
        tenant, sep, key = pair.partition(":")
        tenant, key = tenant.strip(), key.strip()
        if sep and tenant and key:
            keys[tenant] = key
    return keys


def _tenant_env_var(tenant_id: str) -> str:
    """The per-tenant key variable for one tenant:
    ``API_KEY_<TENANT>``, the tenant id uppercased with every
    non-alphanumeric character folded to ``_`` (``acme-retail`` →
    ``API_KEY_ACME_RETAIL``). Computed from the tenant id, never
    parsed back from a variable name, so the mapping is exact."""
    import re

    return "API_KEY_" + re.sub(r"[^A-Za-z0-9]", "_", tenant_id).upper()


def tenant_api_key(tenant_id: str) -> str | None:
    """The API key issued for one tenant, or None when it has none.

    ``TENANT_API_KEYS`` wins over the per-tenant ``API_KEY_<TENANT>``
    variable when both name the tenant.
    """
    load_dotenv()
    key = tenant_api_keys().get(tenant_id)
    if key:
        return key
    return env_str(_tenant_env_var(tenant_id))


def per_tenant_keys_configured() -> bool:
    """Whether any per-tenant key configuration exists at all.

    True when ``TENANT_API_KEYS`` names a tenant or any
    ``API_KEY_<TENANT>`` variable is set. This is the switch between
    the API's two auth models (see ``api.require_api_key``): with no
    per-tenant configuration, the tenant header stays a trusted
    partition claim under the single shared key; with any, a key
    must belong to the tenant it is presented for.
    """
    load_dotenv()
    if tenant_api_keys():
        return True
    return any(
        name.startswith("API_KEY_") and value
        for name, value in os.environ.items()
    )


def known_api_keys() -> set[str]:
    """Every key value the deployment recognises: the shared
    ``API_KEY``, every ``TENANT_API_KEYS`` value, every
    ``API_KEY_<TENANT>`` value. Used to tell "a real key aimed at
    the wrong tenant" (403) apart from "no such key" (401)."""
    load_dotenv()
    keys = set(tenant_api_keys().values())
    shared = env_str("API_KEY")
    if shared:
        keys.add(shared)
    for name, value in os.environ.items():
        if name.startswith("API_KEY_") and value:
            keys.add(value)
    return keys


def env_str(name: str, default: str | None = None) -> str | None:
    """Read a string variable, treating an empty value as unset."""
    value = os.environ.get(name)
    return value if value else default


def env_float(name: str, default: float) -> float:
    """Read a float variable, falling back to ``default`` when unset/invalid."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Read an int variable, falling back to ``default`` when unset/invalid."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
