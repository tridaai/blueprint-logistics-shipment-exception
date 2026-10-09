"""LLM backend tests — provider SDKs are faked in ``sys.modules``.

No network, no real keys. The fakes record every call so tests can assert
which provider surface was used (draft vs classification suggestion),
which model was requested, and which client kwargs (api_key, base_url,
timeout) the backend constructed.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import (
    AnthropicBackend,
    MockModelBackend,
    OpenAIBackend,
    get_backend,
)
from shipment_agent.schemas import ShipmentInput

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "OPENAI_EMBEDDING_MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
    "ANTHROPIC_BASE_URL", "LLM_TIMEOUT_SECONDS", "RETRIEVER",
]

DELAY_SHIPMENT = {
    "shipment_id": "INT-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Synthetic Customer",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

KEYWORD_DELAY_SHIPMENT = {  # no dates -> keyword-only delay, confidence 0.8
    "shipment_id": "SUG-2",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "latest_event": "Delayed at hub",
    "documents": [],
}

CLEAN_SHIPMENT = {  # no signals at all -> rule result `none`
    "shipment_id": "SUG-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "latest_event": "In transit, on schedule",
    "documents": [],
}

CLEAN_DRAFT = (
    "Subject: Update on shipment {sid}: delay\n\n"
    "Dear Synthetic Customer,\n\n"
    "Your shipment {sid} is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)


# --------------------------------------------------------------------------
# Fake provider SDKs
# --------------------------------------------------------------------------

class _FakeCompletions:
    def __init__(self, client: "FakeOpenAI") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, messages=None):
        system = " ".join(messages[0]["content"].split())  # normalise line wraps
        kind = "classify" if "suggesting an exception classification" in system else "draft"
        self._client.calls.append({"kind": kind, "model": model})
        text = self._client.suggestion_text if kind == "classify" else self._client.draft_text
        message = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []
    draft_text = CLEAN_DRAFT
    suggestion_text = (
        '{"exception_type": "damage", "severity": "high", "confidence": 0.77,'
        ' "rationale": "The notes hint at handling damage the rules did not match."}'
    )

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))
        FakeOpenAI.instances.append(self)


class _FakeMessages:
    def __init__(self, client: "FakeAnthropic") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, system=None, messages=None):
        system = " ".join((system or "").split())  # normalise line wraps
        kind = "classify" if "suggesting an exception classification" in system else "draft"
        self._client.calls.append({"kind": kind, "model": model})
        text = self._client.suggestion_text if kind == "classify" else self._client.draft_text
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class FakeAnthropic:
    instances: list["FakeAnthropic"] = []
    draft_text = CLEAN_DRAFT
    suggestion_text = FakeOpenAI.suggestion_text

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[dict] = []
        self.messages = _FakeMessages(self)
        FakeAnthropic.instances.append(self)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    # Isolate backend/retriever selection from any repo-root .env; .env
    # loading itself is covered in test_config.py.
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_openai(monkeypatch):
    FakeOpenAI.instances = []
    FakeOpenAI.draft_text = CLEAN_DRAFT
    FakeOpenAI.suggestion_text = (
        '{"exception_type": "damage", "severity": "high", "confidence": 0.77,'
        ' "rationale": "The notes hint at handling damage the rules did not match."}'
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    return FakeOpenAI


@pytest.fixture
def fake_anthropic(monkeypatch):
    FakeAnthropic.instances = []
    FakeAnthropic.draft_text = CLEAN_DRAFT
    FakeAnthropic.suggestion_text = FakeOpenAI.suggestion_text
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    return FakeAnthropic


def _run(shipment: dict, backend) -> object:
    return run_shipment(ShipmentInput.model_validate(shipment), backend=backend)


# --------------------------------------------------------------------------
# Selection + construction
# --------------------------------------------------------------------------

def test_mock_is_the_default_backend():
    backend = get_backend()
    assert isinstance(backend, MockModelBackend)
    assert backend.name == "mock"


def test_missing_openai_key_fails_loudly(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "openai")
    with pytest.raises(RuntimeError) as excinfo:
        get_backend()
    message = str(excinfo.value)
    assert "OPENAI_API_KEY is not set" in message
    assert ".env" in message  # the message names the fix


def test_missing_anthropic_key_fails_loudly(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "anthropic")
    with pytest.raises(RuntimeError) as excinfo:
        get_backend()
    assert "ANTHROPIC_API_KEY is not set" in str(excinfo.value)


def test_missing_sdk_fails_loudly(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "openai", None)  # import now raises ImportError
    with pytest.raises(RuntimeError) as excinfo:
        get_backend()
    assert "uv sync --extra llm" in str(excinfo.value)


def test_unknown_backend_fails(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "bogus")
    with pytest.raises(ValueError):
        get_backend()


def test_openai_client_kwargs_base_url_timeout_key(monkeypatch, fake_openai):
    monkeypatch.setenv("MODEL_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12")
    backend = get_backend()
    assert isinstance(backend, OpenAIBackend)
    kwargs = fake_openai.instances[-1].kwargs
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["base_url"] == "https://gateway.example/v1"
    assert kwargs["timeout"] == 12.0


def test_openai_client_omits_base_url_by_default(monkeypatch, fake_openai):
    monkeypatch.setenv("MODEL_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    get_backend()
    kwargs = fake_openai.instances[-1].kwargs
    assert "base_url" not in kwargs
    assert kwargs["timeout"] == 60.0  # documented default


def test_anthropic_client_kwargs_base_url(monkeypatch, fake_anthropic):
    monkeypatch.setenv("MODEL_BACKEND", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")
    backend = get_backend()
    assert isinstance(backend, AnthropicBackend)
    kwargs = fake_anthropic.instances[-1].kwargs
    assert kwargs["api_key"] == "sk-ant-test"
    assert kwargs["base_url"] == "https://anthropic.example"


def test_model_override_is_used(monkeypatch, fake_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "my-hosted-model")
    backend = OpenAIBackend()
    _run(DELAY_SHIPMENT, backend)
    models = {call["model"] for call in fake_openai.instances[-1].calls}
    assert models == {"my-hosted-model"}


# --------------------------------------------------------------------------
# Drafting through the real pipeline (guardrails still run afterwards)
# --------------------------------------------------------------------------

def test_openai_draft_flows_through_pipeline_and_guardrails(monkeypatch, fake_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_openai.draft_text = CLEAN_DRAFT.format(sid="INT-1")
    result = _run(DELAY_SHIPMENT, OpenAIBackend())
    assert result.draft.subject == "Update on shipment INT-1: delay"
    assert "INT-1" in result.draft.body
    assert result.validation.passed
    assert result.approval_status == "awaiting_approval"
    assert result.external_action_taken is False


def test_anthropic_draft_flows_through_pipeline_and_guardrails(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    fake_anthropic.draft_text = CLEAN_DRAFT.format(sid="INT-1")
    result = _run(DELAY_SHIPMENT, AnthropicBackend())
    assert "INT-1" in result.draft.body
    assert result.validation.passed


def test_llm_draft_with_banned_phrase_fails_guardrails(monkeypatch, fake_openai):
    """Guardrails run AFTER generation — a fluent but non-compliant LLM
    draft is blocked exactly like a template draft would be."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_openai.draft_text = (
        "Subject: Update on shipment INT-1\n\n"
        "Dear Synthetic Customer, your shipment INT-1 is delayed and "
        "we guarantee a full refund. The next update will arrive tomorrow."
    )
    result = _run(DELAY_SHIPMENT, OpenAIBackend())
    assert result.validation.passed is False
    failed = {c.name for c in result.validation.checks if not c.passed}
    assert "no_prohibited_promises" in failed


# --------------------------------------------------------------------------
# Classification suggestion (LLM backends only, advisory)
# --------------------------------------------------------------------------

def test_suggestion_fires_on_none_result(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    fake_anthropic.draft_text = CLEAN_DRAFT.format(sid="SUG-1")
    result = _run(CLEAN_SHIPMENT, AnthropicBackend())
    # Rule result stands; the suggestion is recorded next to it.
    assert result.classification.exception_type.value == "none"
    assert result.llm_suggestion is not None
    assert result.llm_suggestion.exception_type == "damage"
    assert result.llm_suggestion.confidence == 0.77
    assert result.llm_suggestion.backend == "anthropic"
    assert result.llm_suggestion.agrees_with_rules is False
    classify_step = next(s for s in result.trace if s.name == "classify")
    assert any("llm suggestion" in d for d in classify_step.details)
    assert any("DISAGREEMENT" in d for d in classify_step.details)


def test_suggestion_fires_on_low_confidence_and_can_agree(monkeypatch, fake_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_openai.draft_text = CLEAN_DRAFT.format(sid="SUG-2")
    fake_openai.suggestion_text = (
        '{"exception_type": "delay", "severity": "medium", "confidence": 0.83,'
        ' "rationale": "The event text reports a delay and nothing else."}'
    )
    result = _run(KEYWORD_DELAY_SHIPMENT, OpenAIBackend())
    assert result.classification.exception_type.value == "delay"
    assert result.classification.confidence == 0.8  # below the 0.85 threshold
    assert result.llm_suggestion is not None
    assert result.llm_suggestion.agrees_with_rules is True


def test_suggestion_does_not_fire_on_confident_rule_result(monkeypatch, fake_openai):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    fake_openai.draft_text = CLEAN_DRAFT.format(sid="INT-1")
    result = _run(DELAY_SHIPMENT, OpenAIBackend())  # computed delay, confidence 0.94
    assert result.llm_suggestion is None
    kinds = [call["kind"] for call in fake_openai.instances[-1].calls]
    assert "classify" not in kinds
    assert kinds == ["draft"]


def test_suggestion_never_fires_with_mock_backend():
    result = _run(CLEAN_SHIPMENT, MockModelBackend())
    assert result.llm_suggestion is None
    result = _run(KEYWORD_DELAY_SHIPMENT, MockModelBackend())
    assert result.llm_suggestion is None


def test_malformed_suggestion_reply_is_dropped(monkeypatch, fake_anthropic):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    fake_anthropic.draft_text = CLEAN_DRAFT.format(sid="SUG-1")
    fake_anthropic.suggestion_text = "I think it might be damage, probably."
    result = _run(CLEAN_SHIPMENT, AnthropicBackend())
    assert result.classification.exception_type.value == "none"
    assert result.llm_suggestion is None
