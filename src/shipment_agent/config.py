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


def _tenant_pairs(raw: str | None) -> dict[str, str]:
    """Parse comma-separated ``tenant:value`` pairs into a dict.

    The shared shape of the ``TENANT_*`` map variables
    (``TENANT_API_KEYS``, ``TENANT_PREVIOUS_API_KEYS``,
    ``TENANT_KEY_ROTATED_AT``). Tenant ids therefore cannot contain
    ``:`` or ``,`` (they are partition names, not prose); whitespace
    around pairs is ignored, and a malformed pair (no colon, an
    empty side) is skipped rather than half-trusted.
    """
    pairs: dict[str, str] = {}
    if not raw:
        return pairs
    for pair in raw.split(","):
        tenant, sep, value = pair.partition(":")
        tenant, value = tenant.strip(), value.strip()
        if sep and tenant and value:
            pairs[tenant] = value
    return pairs


def tenant_api_keys() -> dict[str, str]:
    """The per-tenant API keys from ``TENANT_API_KEYS``: tenant → key.

    Format: comma-separated ``tenant:key`` pairs —
    ``TENANT_API_KEYS=acme:key-one,globex:key-two``. An empty dict
    means the variable is not configured.
    """
    load_dotenv()
    return _tenant_pairs(env_str("TENANT_API_KEYS"))


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


# ---------------------------------------------------------------------------
# Key rotation: a tenant's outgoing key, accepted for a bounded window
# ---------------------------------------------------------------------------
#
# Rotating a tenant's key used to be a hard cutover: the moment the
# new key was configured, the old one 401'd — including for requests
# already in flight from the tenant's own systems. A rotation is now
# a three-variable ceremony:
#
# - ``TENANT_PREVIOUS_API_KEYS`` (or ``API_KEY_PREVIOUS_<TENANT>``)
#   holds the outgoing key beside the new current one;
# - ``TENANT_KEY_ROTATED_AT`` records when the rotation happened
#   (per tenant, ISO-8601);
# - ``TENANT_KEY_GRACE_HOURS`` bounds how long the outgoing key keeps
#   working after that moment (per tenant
#   ``TENANT_KEY_GRACE_HOURS_<TENANT>`` wins; default 72 hours).
#
# The outgoing key authenticates as the *previous* generation: the
# API records which generation each analysis arrived under (an id
# like ``acme:previous`` — never the secret), so an operator can see
# the old key still in use and know when the window can close. With
# no rotation timestamp there is no window at all: a "previous" key
# without a recorded rotation is just an unknown key.

DEFAULT_KEY_GRACE_HOURS = 72.0


def tenant_previous_api_keys() -> dict[str, str]:
    """The outgoing (previous-generation) keys from
    ``TENANT_PREVIOUS_API_KEYS``: tenant → key, same pair format as
    :func:`tenant_api_keys`."""
    load_dotenv()
    return _tenant_pairs(env_str("TENANT_PREVIOUS_API_KEYS"))


def _tenant_previous_env_var(tenant_id: str) -> str:
    """The per-tenant previous-key variable: ``API_KEY_PREVIOUS_<TENANT>``,
    the tenant id folded exactly as in :func:`_tenant_env_var`."""
    import re

    return "API_KEY_PREVIOUS_" + re.sub(r"[^A-Za-z0-9]", "_", tenant_id).upper()


def tenant_previous_api_key(tenant_id: str) -> str | None:
    """The outgoing key issued for one tenant, or None.

    ``TENANT_PREVIOUS_API_KEYS`` wins over the per-tenant
    ``API_KEY_PREVIOUS_<TENANT>`` variable, mirroring the current
    key's precedence (:func:`tenant_api_key`)."""
    load_dotenv()
    key = tenant_previous_api_keys().get(tenant_id)
    if key:
        return key
    return env_str(_tenant_previous_env_var(tenant_id))


def tenant_key_rotated_at(tenant_id: str):
    """When this tenant's key last rotated (aware datetime), or None.

    Read from ``TENANT_KEY_ROTATED_AT`` (a ``tenant:<ISO-8601>``
    pair map) or the per-tenant ``TENANT_KEY_ROTATED_AT_<TENANT>``
    variable. An unparseable value is no rotation — a typo must not
    silently open a grace window. Naive timestamps are read as UTC.
    """
    from datetime import datetime, timezone

    load_dotenv()
    raw = _tenant_pairs(env_str("TENANT_KEY_ROTATED_AT")).get(tenant_id)
    if not raw:
        import re

        var = "TENANT_KEY_ROTATED_AT_" + re.sub(
            r"[^A-Za-z0-9]", "_", tenant_id
        ).upper()
        raw = env_str(var)
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def tenant_key_grace_hours(tenant_id: str) -> float:
    """How long the outgoing key keeps working after rotation.

    ``TENANT_KEY_GRACE_HOURS_<TENANT>`` over the global
    ``TENANT_KEY_GRACE_HOURS`` over :data:`DEFAULT_KEY_GRACE_HOURS`.
    A negative value is a configuration error, not an instant
    cutover: it clamps to zero (the window closes at the rotation
    moment itself)."""
    import re

    var = "TENANT_KEY_GRACE_HOURS_" + re.sub(
        r"[^A-Za-z0-9]", "_", tenant_id
    ).upper()
    hours = env_float(var, env_float("TENANT_KEY_GRACE_HOURS", DEFAULT_KEY_GRACE_HOURS))
    return max(0.0, hours)


def key_grace_deadline(tenant_id: str):
    """When this tenant's previous key stops working, or None when
    no rotation is recorded (no window exists)."""
    from datetime import timedelta

    rotated = tenant_key_rotated_at(tenant_id)
    if rotated is None:
        return None
    return rotated + timedelta(hours=tenant_key_grace_hours(tenant_id))


def tenant_key_generation(
    tenant_id: str, presented_key: str | None, now=None
) -> str | None:
    """Which generation of the tenant's key was presented:
    ``"current"``, ``"previous"``, or None (neither key, or the
    previous key outside its grace window).

    ``now`` injects the clock (tests drive a fixed one); the default
    is real UTC now. The previous generation validates only inside
    its window: rotation timestamp recorded, and ``now`` at or
    before the grace deadline.
    """
    if not presented_key:
        return None
    current = tenant_api_key(tenant_id)
    if current is not None and presented_key == current:
        return "current"
    previous = tenant_previous_api_key(tenant_id)
    if previous is not None and presented_key == previous:
        deadline = key_grace_deadline(tenant_id)
        if deadline is not None:
            from datetime import datetime, timezone

            moment = now or datetime.now(timezone.utc)
            if moment <= deadline:
                return "previous"
    return None


def tenant_rotation_status(tenant_id: str, now=None) -> dict:
    """The non-secret facts of one tenant's key rotation, for the
    operator view (``GET /auth/rotation``) and /metrics.

    Booleans and timestamps only — never key material. ``grace_open``
    is True while the outgoing key still authenticates."""
    from datetime import datetime, timezone

    rotated = tenant_key_rotated_at(tenant_id)
    deadline = key_grace_deadline(tenant_id)
    moment = now or datetime.now(timezone.utc)
    return {
        "tenant_id": tenant_id,
        "current_key_configured": tenant_api_key(tenant_id) is not None,
        "previous_key_configured": tenant_previous_api_key(tenant_id) is not None,
        "rotated_at": rotated.isoformat() if rotated else None,
        "grace_hours": tenant_key_grace_hours(tenant_id),
        "grace_deadline": deadline.isoformat() if deadline else None,
        "grace_open": bool(deadline is not None and moment <= deadline),
    }


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
    if tenant_api_keys() or tenant_previous_api_keys():
        return True
    return any(
        name.startswith("API_KEY_") and value
        for name, value in os.environ.items()
    )


def known_api_keys() -> set[str]:
    """Every key value the deployment recognises: the shared
    ``API_KEY``, every current and previous ``TENANT_*_API_KEYS``
    value, every ``API_KEY_<TENANT>`` / ``API_KEY_PREVIOUS_<TENANT>``
    value. Used to tell "a real key aimed at the wrong tenant"
    (403) apart from "no such key" (401). An expired previous key
    stays *known* — aimed at its own tenant it earns the precise
    "grace window closed" answer, not a generic 401."""
    load_dotenv()
    keys = set(tenant_api_keys().values())
    keys.update(tenant_previous_api_keys().values())
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
