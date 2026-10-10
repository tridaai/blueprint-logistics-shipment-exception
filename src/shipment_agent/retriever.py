"""Retrieval behind a small interface — the LlamaIndex-style layer.

Three implementations ship in the repo, selected with the ``RETRIEVER``
environment variable (see :func:`get_retriever`):

- ``keyword`` (default) — transparent token overlap, deterministic, offline.
- ``semantic`` — embeds the policy corpus and the query with an embeddings
  API and ranks by cosine similarity. Embeddings come from the local
  Ollama server when ``MODEL_BACKEND=ollama``, otherwise from OpenAI
  (or an OpenAI-compatible endpoint via ``OPENAI_BASE_URL``): Anthropic
  has no embeddings API, so ``MODEL_BACKEND=anthropic`` still needs
  ``OPENAI_API_KEY`` for this retriever — the failure message says
  exactly that when the key is missing. Where the vectors live:
  **pgvector** in the application's PostgreSQL when ``DATABASE_URL``
  is set (the production path), else a Chroma store when the
  ``vectordb`` extra / a Chroma server is configured (the documented
  alternative), else in-memory cosine.
- ``hybrid`` — runs both, merges the candidate pools, and applies an
  explicit rerank step: reciprocal-rank score fusion in code (see
  :func:`rerank_fused`). This is score-fusion reranking, not a
  cross-encoder — the repo ships no cross-encoder and claims none.

A production corpus fed by the client's own document systems would
swap the corpus source (and could put LlamaIndex in front of the
same pgvector tables); the agent only depends on the ``Retriever``
protocol, so that swap touches one file, not the graph.

**Tenancy.** The corpus is tagged: shared documents carry no
``tenant_id``, a tenant's own SOPs carry theirs (see
``policies_data.TENANT_POLICIES``). Every implementation scopes its
corpus with :func:`corpus_for_tenant` and offers ``for_tenant`` —
the service resolves a scoped view per run, so one tenant's runs
retrieve its own SOPs on top of the shared corpus while another
tenant's documents are absent from the corpus entirely (never
ranked, never cited — not filtered from the results after the
fact). An unscoped retriever (the CLI / demo default) sees the
shared corpus only. A retriever built over an explicitly injected
corpus is the injector's own: ``for_tenant`` returns it unchanged,
which is what the eval harnesses and tests rely on.
"""

from __future__ import annotations

import math
import re

from .config import env_float, env_str, load_dotenv
from .errors import translate_construction_error, translate_provider_error
from .model_backends import DEFAULT_TIMEOUT_SECONDS, _missing_sdk_error
from .policies_data import full_corpus
from .schemas import RetrievedPolicy
from .tracing import provider_span

_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "was",
    "by", "with", "when", "if", "it", "its", "are", "be", "at", "as",
}


def corpus_for_tenant(
    policies: list[dict[str, str]], tenant_id: str | None
) -> list[dict[str, str]]:
    """The slice of a tagged corpus one tenant may retrieve from.

    A document is visible when it is shared (no ``tenant_id``) or
    belongs to this tenant. ``tenant_id=None`` (no tenant claimed)
    sees shared documents only — tenant documents never surface for
    an anonymous run, and never for another tenant's.
    """
    return [
        policy
        for policy in policies
        if not policy.get("tenant_id") or policy.get("tenant_id") == tenant_id
    ]


def merge_corpus(
    base: list[dict[str, str]], extra: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Merge runtime-supplied documents over a base corpus.

    An extra document whose ``policy_id`` already exists in the base
    replaces that entry in place (position kept); a new id appends.
    This is how a tenant's stored documents (see
    ``service.upsert_tenant_policy``) join — and can revise — its
    bundled corpus without a restart: the merged corpus is what the
    retriever views are then scoped from, so tenant scoping applies
    to stored documents exactly as to bundled ones.
    """
    merged = list(base)
    positions = {p["policy_id"]: i for i, p in enumerate(merged)}
    for policy in extra:
        policy_id = policy["policy_id"]
        if policy_id in positions:
            merged[positions[policy_id]] = policy
        else:
            positions[policy_id] = len(merged)
            merged.append(policy)
    return merged


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


# The Retriever protocol is declared in ports.py (the seam registry);
# it is re-exported here so existing imports keep working.
from .ports import Retriever  # noqa: E402,F401


class KeywordRetriever:
    """Transparent keyword-overlap retriever over the policy corpus.

    Deliberately simple and deterministic so evals and tests are stable
    offline. Score = shared tokens normalised by policy length.

    With no ``policies`` argument the corpus is the deployment's
    tagged corpus (shared + every tenant's documents), scoped to
    ``tenant_id`` — ``None`` (the default) sees the shared corpus
    only, and :meth:`for_tenant` returns the view for one tenant. An
    explicitly injected corpus is used exactly as given (the eval
    harnesses own their corpora), and ``for_tenant`` then returns
    this retriever unchanged.
    """

    name = "keyword"

    def __init__(
        self,
        policies: list[dict[str, str]] | None = None,
        *,
        tenant_id: str | None = None,
        _corpus: list[dict[str, str]] | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        if _corpus is not None:
            # A scoped view over the deployment corpus (for_tenant).
            self._explicit = False
            self._corpus = _corpus
            self._policies = corpus_for_tenant(_corpus, tenant_id)
        elif policies is not None:
            self._explicit = True
            self._corpus = list(policies)
            self._policies = self._corpus
        else:
            self._explicit = False
            self._corpus = full_corpus()
            self._policies = corpus_for_tenant(self._corpus, tenant_id)
        self._index = [
            (p, set(_tokens(f"{p['title']} {p['text']}"))) for p in self._policies
        ]

    @property
    def tenant_id(self) -> str | None:
        """The tenant this view is scoped to (None = shared only)."""
        return self._tenant_id

    def for_tenant(self, tenant_id: str | None) -> "KeywordRetriever":
        """The view of this retriever scoped to one tenant's corpus."""
        if self._explicit or tenant_id == self._tenant_id:
            return self
        return KeywordRetriever(tenant_id=tenant_id, _corpus=self._corpus)

    def with_extra_policies(
        self, extra: list[dict[str, str]]
    ) -> "KeywordRetriever":
        """A view whose corpus also carries runtime-supplied documents.

        The stored tenant documents (see
        ``service.upsert_tenant_policy``) merged over the deployment
        corpus with :func:`merge_corpus`, this view's tenant scoping
        unchanged — the service resolves extras per tenant, and the
        scoping here still decides what the view may see. An
        explicitly injected corpus is the injector's own and is
        returned unchanged, exactly as with :meth:`for_tenant`."""
        if self._explicit or not extra:
            return self
        return KeywordRetriever(
            tenant_id=self._tenant_id, _corpus=merge_corpus(self._corpus, extra)
        )

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
                    retrieval="keyword",
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _vector_literal(vector: list[float]) -> str:
    """A pgvector text literal (``[0.1,0.2,...]``) for a Python vector.

    Embeddings cross the SQL boundary as literals cast with
    ``::vector`` — deliberately no driver-level type registration, so
    any psycopg connection serves the pgvector path unchanged.
    """
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


class SemanticRetriever:
    """Cosine-similarity retriever over the policy corpus, via embeddings.

    The corpus is embedded once (lazily, on first use) and cached; each
    query is embedded per call. Same return shape as the keyword
    retriever: ``RetrievedPolicy`` snippets with similarity scores.
    """

    name = "semantic"

    def __init__(
        self,
        policies: list[dict[str, str]] | None = None,
        *,
        mode_label: str = "semantic",
        tenant_id: str | None = None,
        _corpus: list[dict[str, str]] | None = None,
    ) -> None:
        # Corpus scoping follows the keyword retriever's model (see
        # its docstring): the default corpus is the deployment's
        # tagged corpus scoped to ``tenant_id``; an injected corpus
        # is the injector's own and for_tenant leaves it alone.
        self._tenant_id = tenant_id
        if _corpus is not None:
            self._explicit = False
            self._corpus = _corpus
            self._policies = corpus_for_tenant(_corpus, tenant_id)
        elif policies is not None:
            self._explicit = True
            self._corpus = list(policies)
            self._policies = self._corpus
        else:
            self._explicit = False
            self._corpus = full_corpus()
            self._policies = corpus_for_tenant(self._corpus, tenant_id)
        # The RETRIEVER value this instance serves ("semantic", or
        # "hybrid" when the hybrid retriever owns it) — error messages
        # name the value the operator actually set, never a sibling mode.
        self._mode_label = mode_label
        (
            self._client,
            self._model,
            self._embeddings_backend,
            self._embeddings_base_url,
        ) = self._build_embeddings_client()
        self._corpus_vectors: list[list[float]] | None = None
        # Which store actually serves queries: "pgvector" when the
        # application's PostgreSQL is configured (DATABASE_URL — the
        # production path), else "chroma" when a Chroma server is
        # configured / the vectordb extra is installed, else "memory"
        # (in-repo cosine over freshly embedded vectors).
        self.vector_store = "memory"
        self._chroma_collection = None
        self._chroma_checked = False
        self._pgvector_checked = False
        self._pgvector_ready = False

    @property
    def tenant_id(self) -> str | None:
        """The tenant this view is scoped to (None = shared only)."""
        return self._tenant_id

    def for_tenant(self, tenant_id: str | None) -> "SemanticRetriever":
        """The view of this retriever scoped to one tenant's corpus.

        A shallow copy sharing the embeddings client (construction
        is the expensive part, and it is tenant-independent), with
        the corpus rescoped and the per-corpus caches reset: the
        pgvector availability answer carries over (it is a property
        of the database, not of the corpus), and the pgvector query
        itself filters on the tenant axis (see _retrieve_pgvector).
        """
        if self._explicit or tenant_id == self._tenant_id:
            return self
        import copy

        view = copy.copy(self)
        view._tenant_id = tenant_id
        view._policies = corpus_for_tenant(self._corpus, tenant_id)
        view._corpus_vectors = None
        view._chroma_collection = None
        view._chroma_checked = False
        return view

    def with_extra_policies(
        self, extra: list[dict[str, str]]
    ) -> "SemanticRetriever":
        """A view whose corpus also carries runtime-supplied documents.

        Same contract as the keyword retriever's: the extras merge
        over the deployment corpus (:func:`merge_corpus`), this
        view's tenant scoping is re-applied, and the per-corpus
        caches reset — so a newly stored document is embedded on
        first use wherever vectors live (the pgvector path embeds it
        through the same content-hash upsert as any corpus change;
        the Chroma path fingerprints the new corpus and re-indexes).
        An explicitly injected corpus is returned unchanged."""
        if self._explicit or not extra:
            return self
        import copy

        view = copy.copy(self)
        view._corpus = merge_corpus(self._corpus, extra)
        view._policies = corpus_for_tenant(view._corpus, view._tenant_id)
        view._corpus_vectors = None
        view._chroma_collection = None
        view._chroma_checked = False
        return view

    def _build_embeddings_client(self):
        """Resolve the embeddings client, failing loudly when unusable.

        Embeddings follow the configured stack:

        - ``MODEL_BACKEND=ollama`` → embeddings from the local Ollama
          server (``OLLAMA_EMBEDDING_MODEL``, default ``nomic-embed-text``)
          — fully local, no cloud key.
        - Otherwise → OpenAI (or an OpenAI-compatible endpoint via
          ``OPENAI_BASE_URL``), regardless of the drafting backend:
          Anthropic offers no embeddings API. That is stated, not hidden:
          with ``MODEL_BACKEND=anthropic`` (or ``mock``) an
          ``OPENAI_API_KEY`` is still required, and the error says so.

        Returns (client, model, provider_name, base_url).
        """
        load_dotenv()
        backend = (env_str("MODEL_BACKEND") or "mock").lower()
        timeout = env_float("LLM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
        if backend == "ollama":
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise _missing_sdk_error("ollama (embeddings)", "OpenAI") from exc
            base_url = env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            try:
                client = OpenAI(
                    api_key="ollama",  # placeholder — Ollama ignores it
                    base_url=base_url,
                    timeout=timeout,
                    max_retries=0,
                )
            except Exception as exc:  # construction failures are translated too
                raise translate_construction_error(
                    exc,
                    backend=f"ollama embeddings (RETRIEVER={self._mode_label})",
                    base_url=base_url,
                ) from exc
            return client, env_str("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"), "ollama", base_url
        if not env_str("OPENAI_API_KEY"):
            hint = (
                f" MODEL_BACKEND={backend} has no embeddings API to use instead —"
                " drafting and embeddings are separate providers here."
                if backend != "openai"
                else ""
            )
            raise RuntimeError(
                f"RETRIEVER={self._mode_label} needs embeddings, which run through OpenAI, "
                "but OPENAI_API_KEY is not set." + hint
                + " Add OPENAI_API_KEY=<your key> to the .env file in the repo root "
                "(copy .env.example to .env) or export it, set MODEL_BACKEND=ollama "
                "to embed locally with Ollama, or set RETRIEVER=keyword."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise _missing_sdk_error("openai (embeddings)", "OpenAI") from exc
        client_kwargs: dict = {
            "api_key": env_str("OPENAI_API_KEY"),
            "timeout": timeout,
            "max_retries": 0,
        }
        base_url = env_str("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        model = env_str("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        try:
            client = OpenAI(**client_kwargs)
        except Exception as exc:  # construction failures are translated too
            raise translate_construction_error(
                exc,
                backend=f"openai embeddings (RETRIEVER={self._mode_label})",
                base_url=base_url or "https://api.openai.com/v1",
            ) from exc
        return (client, model, "openai", base_url or "https://api.openai.com/v1")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        with provider_span(
            self._embeddings_backend, self._model, "embeddings"
        ) as handle:
            handle.set_counts(input_count=len(texts))
            try:
                response = self._client.embeddings.create(
                    model=self._model, input=texts
                )
            except Exception as exc:
                raise translate_provider_error(
                    exc,
                    backend=f"{self._embeddings_backend} embeddings (RETRIEVER={self._mode_label})",
                    base_url=self._embeddings_base_url,
                ) from exc
            ordered = sorted(response.data, key=lambda d: getattr(d, "index", 0))
            return [list(d.embedding) for d in ordered]

    # -- pgvector (the production store: the application's Postgres) --

    def _pgvector_available(self) -> bool:
        """True when DATABASE_URL points at a migrated Postgres.

        Checked once; on any failure the retriever falls through to
        the Chroma / in-memory paths rather than breaking retrieval —
        the same posture as the Chroma path below.
        """
        if self._pgvector_checked:
            return self._pgvector_ready
        self._pgvector_checked = True
        try:
            from .db import connect, database_url, ensure_migrated

            if not database_url():
                return False
            ensure_migrated()
            with connect() as conn:
                conn.execute("SELECT 1 FROM policy_embeddings LIMIT 1")
            self._pgvector_ready = True
            self.vector_store = "pgvector"
        except Exception:
            self._pgvector_ready = False
        return self._pgvector_ready

    def _pg_ensure_corpus(self, conn) -> None:
        """Embed any missing/stale policies into the pgvector table.

        Rows are keyed by (policy_id, model) and carry a content
        hash: an edited policy is re-embedded, an unchanged corpus
        costs one SELECT per retrieval.
        """
        import hashlib

        def content_hash(policy: dict) -> str:
            return hashlib.sha256(
                f"{policy['policy_id']}|{policy['title']}|{policy['text']}".encode(
                    "utf-8"
                )
            ).hexdigest()

        existing = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT policy_id, content_hash FROM policy_embeddings "
                "WHERE model = %s",
                (self._model,),
            )
        }
        stale = [
            p for p in self._policies if existing.get(p["policy_id"]) != content_hash(p)
        ]
        if not stale:
            return
        vectors = self._embed([f"{p['title']} {p['text']}" for p in stale])
        for policy, vector in zip(stale, vectors):
            conn.execute(
                "INSERT INTO policy_embeddings "
                "(policy_id, model, content_hash, embedding, tenant_id) "
                "VALUES (%s, %s, %s, %s::vector, %s) "
                "ON CONFLICT (policy_id, model) DO UPDATE SET "
                "content_hash = EXCLUDED.content_hash, "
                "embedding = EXCLUDED.embedding, "
                "tenant_id = EXCLUDED.tenant_id",
                (
                    policy["policy_id"],
                    self._model,
                    content_hash(policy),
                    _vector_literal(vector),
                    policy.get("tenant_id"),
                ),
            )
        conn.commit()

    def _retrieve_pgvector(self, query: str, top_k: int) -> list[RetrievedPolicy]:
        from .db import connect

        query_vector = _vector_literal(self._embed([query])[0])
        with connect() as conn:
            self._pg_ensure_corpus(conn)
            # The tenant axis (migration 0005): a row serves when it
            # is shared (tenant_id NULL) or belongs to this view's
            # tenant. IS NOT DISTINCT FROM keeps the unscoped view
            # (tenant None) on shared rows only.
            rows = conn.execute(
                "SELECT policy_id, 1 - (embedding <=> %s::vector) AS score "
                "FROM policy_embeddings WHERE model = %s "
                "AND (tenant_id IS NULL OR tenant_id IS NOT DISTINCT FROM %s) "
                "ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_vector, self._model, self._tenant_id, query_vector, top_k),
            ).fetchall()
        by_id = {p["policy_id"]: p for p in self._policies}
        retrieved: list[RetrievedPolicy] = []
        for policy_id, score in rows:
            policy = by_id.get(policy_id)
            if policy is None:  # row from a different corpus — skip it
                continue
            retrieved.append(
                RetrievedPolicy(
                    policy_id=policy_id,
                    title=policy["title"],
                    snippet=policy["text"],
                    score=round(float(score), 4),
                    retrieval="semantic",
                )
            )
        return retrieved

    def _get_chroma_collection(self):
        """Lazily open the local Chroma store, indexing the corpus once.

        Returns the collection, or ``None`` when chromadb is not
        installed (the in-memory fallback then serves queries). The
        store is either a Chroma server (``CHROMA_HOST`` set — the
        docker-compose local stack runs one) or an embedded persistent
        directory (``CHROMA_DIR``, default ``<repo>/.chroma``,
        git-ignored): the policy corpus is embedded and added on first
        use, and survives restarts after that. Embeddings are always
        computed by our configured provider client and passed in
        explicitly — Chroma's own default embedding function (which
        downloads a model) is never used.
        """
        if self._chroma_checked:
            return self._chroma_collection
        self._chroma_checked = True
        try:
            import chromadb
        except ImportError:
            return None
        try:
            import hashlib
            from pathlib import Path

            chroma_host = env_str("CHROMA_HOST")
            if chroma_host:
                client = chromadb.HttpClient(
                    host=chroma_host, port=int(env_str("CHROMA_PORT") or "8000")
                )
            else:
                persist_dir = env_str("CHROMA_DIR") or str(
                    Path(__file__).resolve().parents[2] / ".chroma"
                )
                client = chromadb.PersistentClient(path=persist_dir)
            # The collection is named by a fingerprint of the corpus
            # contents: a changed corpus (edited policies, a different
            # corpus in tests) gets its own collection instead of
            # silently querying stale vectors.
            fingerprint = hashlib.sha256(
                "\n".join(
                    f"{p['policy_id']}|{p['title']}|{p['text']}" for p in self._policies
                ).encode("utf-8")
            ).hexdigest()[:12]
            collection = client.get_or_create_collection(
                f"policies-{fingerprint}", metadata={"hnsw:space": "cosine"}
            )
            if collection.count() != len(self._policies):
                if collection.count() > 0:  # partial/stale write — rebuild
                    collection.delete(
                        ids=[p["policy_id"] for p in self._policies]
                    )
                texts = [f"{p['title']} {p['text']}" for p in self._policies]
                collection.add(
                    ids=[p["policy_id"] for p in self._policies],
                    embeddings=self._embed(texts),
                    documents=texts,
                    metadatas=[{"title": p["title"]} for p in self._policies],
                )
            self._chroma_collection = collection
            self.vector_store = "chroma"
        except Exception:  # a broken local store must not break retrieval
            self._chroma_collection = None
        return self._chroma_collection

    def _retrieve_chroma(self, collection, query: str, top_k: int) -> list[RetrievedPolicy]:
        query_vector = self._embed([query])[0]
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, len(self._policies)),
        )
        by_id = {p["policy_id"]: p for p in self._policies}
        retrieved: list[RetrievedPolicy] = []
        for policy_id, distance in zip(result["ids"][0], result["distances"][0]):
            policy = by_id.get(policy_id)
            if policy is None:  # foreign id in a shared store — skip it
                continue
            retrieved.append(
                RetrievedPolicy(
                    policy_id=policy_id,
                    title=policy["title"],
                    snippet=policy["text"],
                    # hnsw:space=cosine -> distance is 1 - cosine similarity.
                    score=round(1.0 - float(distance), 4),
                    retrieval="semantic",
                )
            )
        return retrieved

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        if self._pgvector_available():
            return self._retrieve_pgvector(query, top_k)
        collection = self._get_chroma_collection()
        if collection is not None:
            return self._retrieve_chroma(collection, query, top_k)
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
                retrieval="semantic",
            )
            for policy, vector in zip(self._policies, self._corpus_vectors)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]


def rerank_fused(
    keyword_results: list[RetrievedPolicy],
    semantic_results: list[RetrievedPolicy],
    top_k: int = 3,
    *,
    rrf_k: int = 60,
) -> list[RetrievedPolicy]:
    """The hybrid rerank step: reciprocal-rank score fusion, in code.

    Keyword scores (token overlap) and semantic scores (cosine) live on
    incomparable scales, so the merge never compares them directly.
    Each list contributes ``1 / (rrf_k + rank)`` per policy; the fused
    score is the sum across lists, and policies found by both retrievers
    rank above single-list finds at similar ranks. The returned score
    IS the fused score — labelled as such via ``retrieval`` ("keyword",
    "semantic", or "keyword+semantic"). This is score-fusion reranking:
    transparent and deterministic, not a cross-encoder model.
    """
    fused: dict[str, dict] = {}
    for source, results in (("keyword", keyword_results), ("semantic", semantic_results)):
        for rank, result in enumerate(results):
            entry = fused.setdefault(
                result.policy_id, {"policy": result, "score": 0.0, "sources": []}
            )
            entry["score"] += 1.0 / (rrf_k + rank + 1)
            entry["sources"].append(source)
            if result.score > entry["policy"].score:
                entry["policy"] = result
    merged = sorted(
        fused.values(), key=lambda e: (-e["score"], e["policy"].policy_id)
    )
    reranked: list[RetrievedPolicy] = []
    for entry in merged[:top_k]:
        sources = [s for s in ("keyword", "semantic") if s in entry["sources"]]
        reranked.append(
            entry["policy"].model_copy(
                update={"score": round(entry["score"], 4), "retrieval": "+".join(sources)}
            )
        )
    return reranked


class HybridRetriever:
    """Keyword + semantic retrieval with an explicit rerank step.

    Pipeline: both retrievers produce a candidate pool → the pools are
    merged by policy ID → :func:`rerank_fused` reranks by reciprocal-rank
    fusion → the top-k are returned and cited exactly like the other
    retrievers' results. ``last_stats`` records the pool sizes of the
    most recent call so the trace can show merge → rerank honestly.
    """

    name = "hybrid"

    def __init__(self, policies: list[dict[str, str]] | None = None) -> None:
        self._keyword = KeywordRetriever(policies)
        self._semantic = SemanticRetriever(policies, mode_label="hybrid")
        self.last_stats: dict[str, int] = {}

    @property
    def vector_store(self) -> str:
        """The store serving the semantic half (chroma | memory)."""
        return self._semantic.vector_store

    @property
    def tenant_id(self) -> str | None:
        """The tenant this view is scoped to (None = shared only)."""
        return self._keyword.tenant_id

    def for_tenant(self, tenant_id: str | None) -> "HybridRetriever":
        """The view of this retriever scoped to one tenant's corpus:
        both halves rescoped, the merge unchanged."""
        keyword = self._keyword.for_tenant(tenant_id)
        semantic = self._semantic.for_tenant(tenant_id)
        if keyword is self._keyword and semantic is self._semantic:
            return self
        import copy

        view = copy.copy(self)
        view._keyword = keyword
        view._semantic = semantic
        view.last_stats = {}
        return view

    def with_extra_policies(
        self, extra: list[dict[str, str]]
    ) -> "HybridRetriever":
        """A view whose corpus also carries runtime-supplied
        documents: both halves gain the extras, the merge unchanged."""
        keyword = self._keyword.with_extra_policies(extra)
        semantic = self._semantic.with_extra_policies(extra)
        if keyword is self._keyword and semantic is self._semantic:
            return self
        import copy

        view = copy.copy(self)
        view._keyword = keyword
        view._semantic = semantic
        view.last_stats = {}
        return view

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedPolicy]:
        pool = top_k * 2
        keyword_results = self._keyword.retrieve(query, top_k=pool)
        semantic_results = self._semantic.retrieve(query, top_k=pool)
        self.last_stats = {
            "keyword_pool": len(keyword_results),
            "semantic_pool": len(semantic_results),
        }
        return rerank_fused(keyword_results, semantic_results, top_k=top_k)


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
    if selected == "hybrid":
        return HybridRetriever()
    raise ValueError(
        f"Unknown RETRIEVER: {selected!r} (expected keyword | semantic | hybrid)"
    )
