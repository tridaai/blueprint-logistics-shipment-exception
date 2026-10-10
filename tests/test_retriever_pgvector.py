"""pgvector retriever helpers (hermetic unit tests).

The live pgvector round-trip lives in test_postgres_integration.py,
gated on DATABASE_URL; what is pinned here is the SQL-boundary
formatting every pgvector query depends on.
"""

from __future__ import annotations

from shipment_agent.retriever import _vector_literal


def test_vector_literal_format():
    assert _vector_literal([0.1, 0.2, -3.0]) == "[0.1,0.2,-3.0]"
    assert _vector_literal([1, 2]) == "[1.0,2.0]"
    assert _vector_literal([]) == "[]"
