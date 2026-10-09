from shipment_agent.retriever import KeywordRetriever


def test_retriever_finds_delay_policy():
    retriever = KeywordRetriever()
    results = retriever.retrieve("delay customer notification revised estimated delivery", top_k=3)
    assert results
    assert results[0].policy_id == "POL-DELAY-01"


def test_retriever_finds_damage_policy():
    retriever = KeywordRetriever()
    results = retriever.retrieve("damage claim packet photographs delivery receipt", top_k=3)
    ids = [r.policy_id for r in results]
    assert "POL-DMG-01" in ids or "POL-CLAIM-01" in ids


def test_retriever_empty_query():
    assert KeywordRetriever().retrieve("", top_k=3) == []
