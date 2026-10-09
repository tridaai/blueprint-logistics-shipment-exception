"""Retrieval behind a small interface — the LlamaIndex-style layer.

Two implementations ship in the repo, selected with the ``RETRIEVER``
environment variable (see :func:`get_retriever`):

- ``keyword`` (default) — transparent token overlap, deterministic, offline.
- ``semantic`` — embeds the policy corpus and the query with an embeddings
  API and ranks by cosine similarity. Embeddings always run through
  OpenAI (or an OpenAI-compatible endpoint via ``OPENAI_BASE_URL``):
  Anthropic has no embeddings API, so ``MODEL_BACKEND=anthropic`` still
  needs ``OPENAI_API_KEY`` for this retriever — the failure message says
  exactly that when the key is missing.

Production deployments would swap either for LlamaIndex over a vector
store (pgvector, Qdrant, …) fed by the client's document systems. The
agent only depends on the ``Retriever`` protocol, so that swap touches
one file, not the graph.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from .config import env_float, env_str, load_dotenv
from .model_backends import DEFAULT_TIMEOUT_SECONDS, _missing_sdk_error
from .policies_data import POLICIES
from .schemas import RetrievedPolicy

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "was",
    "by", "with", "when", "if", "it", "its", "are", "be", "at", "as",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]: ...


class KeywordRetriever:
    """Transparent keyword-overlap retriever over the policy corpus.

    Deliberately simple and deterministic so evals and tests are stable
    offline. Score = shared tokens normalised by policy length.
    """

    def __init__(self, policies: list[dict[str, str]] | None = None) -> None:
        self._policies = policies if policies is not None else POLICIES
        self._index = [
            (p, set(_tokens(f"{p['title']} {p['text']}"))) for p in self._policies
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return []
        scored: list[RetrievedPolicy] = []
        for policy, tokens in self._index:
            overlap = len(query_tokens & tokens)
            if overlap == 0:
                continue
            score = round(overlap / math.sqrt(len(tokens)), 4)
            scored.append(
                RetrievedPolicy(
                    policy_id=policy["policy_id"],
                    title=policy["title"],
                    snippet=policy["text"],
                    score=score,
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class SemanticRetriever:
    """Cosine-similarity retriever over the policy corpus, via embeddings.

    The corpus is embedded once (lazily, on first use) and cached; each
    query is embedded per call. Same return shape as the keyword
    retriever: ``RetrievedPolicy`` snippets with similarity scores.
    """

    def __init__(self, policies: list[dict[str, str]] | None = None) -> None:
        self._policies = policies if policies is not None else POLICIES
        self._client, self._model = self._build_embeddings_client()
        self._corpus_vectors: list[list[float]] | None = None

    @staticmethod
    def _build_embeddings_client():
        """Resolve the embeddings client, failing loudly when unusable.

        Embeddings come from OpenAI regardless of the drafting backend —
        Anthropic offers no embeddings API. That is stated, not hidden:
        with ``MODEL_BACKEND=anthropic`` (or ``mock``) an ``OPENAI_API_KEY``
        is still required, and the error says so.
        """
        load_dotenv()
        if not env_str("OPENAI_API_KEY"):
            backend = (env_str("MODEL_BACKEND") or "mock").lower()
            hint = (
                f" MODEL_BACKEND={backend} has no embeddings API to use instead —"
                " drafting and embeddings are separate providers here."
                if backend != "openai"
                else ""
            )
            raise RuntimeError(
                "RETRIEVER=semantic needs embeddings, which run through OpenAI, "
                "but OPENAI_API_KEY is not set." + hint
                + " Add OPENAI_API_KEY=<your key> to the .env file in the repo root "
                "(copy .env.example to .env) or export it, or set RETRIEVER=keyword."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise _missing_sdk_error("openai (embeddings)", "OpenAI") from exc
        client_kwargs: dict = {
            "api_key": env_str("OPENAI_API_KEY"),
            "timeout": env_float("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        }
        base_url = env_str("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        model = env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAI(**client_kwargs), model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda d: getattr(d, "index", 0))
        return [list(d.embedding) for d in ordered]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        if self._corpus_vectors is None:
            self._corpus_vectors = self._embed(
                [f"{p['title']} {p['text']}" for p in self._policies]
            )
        query_vector = self._embed([query])[0]
        scored = [
            RetrievedPolicy(
                policy_id=policy["policy_id"],
                title=policy["title"],
                snippet=policy["text"],
                score=round(_cosine(query_vector, vector), 4),
            )
            for policy, vector in zip(self._policies, self._corpus_vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def get_retriever(name: str | None = None) -> Retriever:
    """Select the retriever by name or the RETRIEVER env var (default: keyword).

    Loads the repo-root ``.env`` first (real environment variables win), so
    every surface — API, CLI, traced demo — selects the same way.
    """
    load_dotenv()
    selected = (name or env_str("RETRIEVER") or "keyword").lower()
    if selected == "keyword":
        return KeywordRetriever()
    if selected == "semantic":
        return SemanticRetriever()
    raise ValueError(f"Unknown RETRIEVER: {selected!r} (expected keyword | semantic)")
