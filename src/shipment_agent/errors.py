"""Provider error translation.

Raw SDK exceptions (``openai.APIConnectionError``, an httpx stack, …)
are what a library should raise and what a product must never show.
Every provider call in the backends and the embeddings client is wrapped
so a transport failure becomes a :class:`ProviderError` in the product's
own ``error: …`` style: it names the backend, the endpoint it tried,
and the likely fix. Callers that degrade (extract / classify / diagnose
/ options) record the translated message in the trace; the drafting
call, which cannot degrade, surfaces it as a clean error on every
surface (CLI/demo exit 1, API 502) instead of a stack dump.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    """A provider call failed; the message says what to do about it."""


def _failure_kind(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (401, 403) or "unauthorized" in message or "authentication" in message:
        return "auth"
    if "connect" in name or "connection" in message or "refused" in message or "unreachable" in message:
        return "connection"
    if "timeout" in name or "timed out" in message or "timeout" in message:
        return "timeout"
    return "provider"


_FIX_HINTS = {
    ("ollama", "connection"): (
        "start Ollama (`ollama serve`), pull the model (`ollama pull llama3.1`), "
        "or point OLLAMA_BASE_URL at a running server"
    ),
    ("openai", "connection"): (
        "check OPENAI_BASE_URL and this machine's network/proxy settings"
    ),
    ("anthropic", "connection"): (
        "check ANTHROPIC_BASE_URL and this machine's network/proxy settings"
    ),
}


def _fix_hint(backend: str, kind: str) -> str:
    specific = _FIX_HINTS.get((backend, kind))
    if specific:
        return specific
    if kind == "auth":
        return "check that the API key in .env / the environment is valid for this endpoint"
    if kind == "timeout":
        return (
            "the endpoint did not answer within LLM_TIMEOUT_SECONDS — raise it, "
            "or check the endpoint's health"
        )
    if kind == "connection":
        return f"check the {backend} endpoint address and this machine's network connection"
    return "see the provider detail above; check the backend configuration in .env"


def translate_provider_error(
    exc: Exception, *, backend: str, base_url: str | None
) -> ProviderError:
    """Turn a raw provider/SDK exception into an actionable ProviderError."""
    if isinstance(exc, ProviderError):
        return exc
    kind = _failure_kind(exc)
    detail = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    if len(detail) > 240:
        detail = detail[:237] + "..."
    endpoint = base_url or "the provider default endpoint"
    return ProviderError(
        f"{backend} provider call failed ({kind}): {detail} — endpoint: {endpoint}. "
        f"Likely fix: {_fix_hint(backend, kind)}."
    )
