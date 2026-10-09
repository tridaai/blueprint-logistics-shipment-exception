"""Retrieval behind a small interface — the LlamaIndex-style layer.

Production deployments would swap the in-repo retriever for LlamaIndex over
a vector store (pgvector, Qdrant, …) fed by the client's document systems.
The agent only depends on the ``Retriever`` protocol, so that swap touches
one file, not the graph.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

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
