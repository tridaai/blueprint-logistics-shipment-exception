"""Node resilience policy: retries for language steps, timeouts, no more.

resilience.py wraps the backend: idempotent language steps (extract,
classify cross-check, diagnose, options, verify, review) retry once
on ProviderError with the retry recorded in the trace; drafting and
anything past the gate never retry; every provider call carries the
NODE_TIMEOUT_SECONDS ceiling. Everything here is a fake backend —
no network.
"""

from __future__ import annotations

import time

import pytest

from shipment_agent.errors import ProviderError
from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.samples import sample_shipment_models

CLASSIFICATION = {
    "exception_type": "delay",
    "severity": "high",
    "confidence": 0.88,
    "rationale": "Computed delay dominates.",
}


def _trace_details(result, step_name: str) -> list[str]:
    return next(s.details for s in result.trace if s.name == step_name)


class FlakyClassifyBackend(MockModelBackend):
    """classify_with_llm fails with ProviderError `failures` times,
    then answers. Counts every call."""

    name = "flaky"

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures
        self.classify_calls = 0

    def classify_with_llm(self, context):
        self.classify_calls += 1
        if self.classify_calls <= self.failures:
            raise ProviderError("flaky provider call failed (connection): boom")
        return dict(CLASSIFICATION)


def test_flaky_classify_succeeds_on_the_retry_with_a_trace_note():
    backend = FlakyClassifyBackend(failures=1)
    result = run_shipment(
        sample_shipment_models()[0], backend=backend, retriever=KeywordRetriever()
    )
    assert backend.classify_calls == 2
    # The cross-check ran with the LLM's answer (not rules_only).
    assert result.cross_check is not None
    assert result.cross_check.resolution == "agree"
    details = _trace_details(result, "classify")
    assert any("attempt 2 after provider error" in d for d in details)


def test_always_failing_classify_degrades_after_two_attempts():
    backend = FlakyClassifyBackend(failures=99)
    result = run_shipment(
        sample_shipment_models()[0], backend=backend, retriever=KeywordRetriever()
    )
    assert backend.classify_calls == 2  # max attempts, then today's degradation
    assert result.cross_check is not None
    assert result.cross_check.resolution == "rules_only"
    assert result.classification.exception_type.value == "delay"


def test_non_provider_errors_are_not_retried():
    class BoomBackend(MockModelBackend):
        name = "boom"

        def __init__(self) -> None:
            super().__init__()
            self.classify_calls = 0

        def classify_with_llm(self, context):
            self.classify_calls += 1
            raise RuntimeError("not a provider error")

    backend = BoomBackend()
    result = run_shipment(
        sample_shipment_models()[0], backend=backend, retriever=KeywordRetriever()
    )
    assert backend.classify_calls == 1
    assert result.cross_check.resolution == "rules_only"


def test_drafting_never_retries():
    class FailingDraftBackend(MockModelBackend):
        name = "failing-draft"

        def __init__(self) -> None:
            super().__init__()
            self.draft_calls = 0

        def draft_customer_update(self, context):
            self.draft_calls += 1
            raise ProviderError("failing-draft provider call failed (connection): down")

    backend = FailingDraftBackend()
    with pytest.raises(ProviderError, match="provider call failed"):
        run_shipment(
            sample_shipment_models()[0], backend=backend, retriever=KeywordRetriever()
        )
    assert backend.draft_calls == 1


def test_flaky_verify_retries_and_lands_the_verdict():
    class FlakyVerifyBackend(MockModelBackend):
        name = "flaky-verify"

        def __init__(self) -> None:
            super().__init__()
            self.verify_calls = 0

        def verify_draft(self, context):
            self.verify_calls += 1
            if self.verify_calls == 1:
                raise ProviderError("flaky-verify provider call failed (timeout): slow")
            return {"grounded": True, "issues": [], "summary": "Grounded."}

    backend = FlakyVerifyBackend()
    result = run_shipment(
        sample_shipment_models()[0], backend=backend, retriever=KeywordRetriever()
    )
    assert backend.verify_calls == 2
    assert result.verification is not None
    assert result.verification.grounded is True
    assert result.verification.source == "llm"
    details = _trace_details(result, "verify")
    assert any("attempt 2 after provider error" in d for d in details)


def test_node_timeout_turns_a_hung_call_into_a_clean_degradation(monkeypatch):
    monkeypatch.setenv("NODE_TIMEOUT_SECONDS", "1")

    class SlowClassifyBackend(MockModelBackend):
        name = "slow"

        def classify_with_llm(self, context):
            time.sleep(5)  # wedged endpoint; the policy must not wait for it
            return dict(CLASSIFICATION)

    started = time.perf_counter()
    result = run_shipment(
        sample_shipment_models()[0],
        backend=SlowClassifyBackend(),
        retriever=KeywordRetriever(),
    )
    elapsed = time.perf_counter() - started
    # Two timed-out attempts at 1s each + backoff — not two 5s sleeps.
    assert elapsed < 4.5
    assert result.cross_check is not None
    assert result.cross_check.resolution == "rules_only"
    details = _trace_details(result, "classify")
    assert any("NODE_TIMEOUT_SECONDS" in d for d in details)
