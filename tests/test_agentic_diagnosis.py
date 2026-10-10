"""Agentic diagnosis tests — scripted tool-calling fakes, no network.

Provider mode upgrades the diagnosis from one composed call to a
bounded tool loop: the model may call search_policies / lane_history /
shipment_facts / carrier_history before composing. These tests script
fake SDKs that emit tool calls and assert: the calls execute through
the toolbox, the loop cap holds, a broken tool degrades gracefully,
both provider wire formats work, and the default (mock) mode is
completely unchanged.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.graph import run_shipment
from shipment_agent.model_backends import MockModelBackend, OpenAIBackend
from shipment_agent.retriever import KeywordRetriever
from shipment_agent.schemas import ShipmentInput
from shipment_agent.tools_agent import (
    TOOL_SPECS,
    DiagnosisToolBox,
    diagnosis_max_tool_calls,
)

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "DIAGNOSIS_MAX_TOOL_CALLS",
]

DELAY_SHIPMENT = {
    "shipment_id": "AGT-1",
    "origin": "Memphis, TN",
    "destination": "Charlotte, NC",
    "customer_name": "Acme Parts",
    "carrier": "Synthetic Carrier",
    "scheduled_delivery": "2026-10-10T09:00:00",
    "estimated_delivery": "2026-10-11T21:00:00",
    "latest_event": "Delayed at regional hub due to weather hold",
    "documents": [],
}

PRIORS = [
    {
        "shipment_id": "OLD-2", "consignee": "Acme Parts", "carrier": "Synthetic Carrier",
        "origin": "Memphis, TN", "destination": "Charlotte, NC",
        "lane": "Memphis, TN -> Charlotte, NC", "exception_type": "damage", "severity": "high",
    },
    {
        "shipment_id": "OLD-1", "consignee": "Acme Parts", "carrier": "Synthetic Carrier",
        "origin": "Memphis, TN", "destination": "Charlotte, NC",
        "lane": "Memphis, TN -> Charlotte, NC", "exception_type": "delay", "severity": "medium",
    },
]

CLASSIFY_TEXT = (
    '{"exception_type": "delay", "severity": "high", "confidence": 0.88,'
    ' "rationale": "Computed delay dominates."}'
)
DIAGNOSIS_TEXT = (
    '{"root_cause": "Weather hold consumed the buffer; delay is 36.0 hours.",'
    ' "summary": "Carrier weather hold, 36-hour delay."}'
)
OPTIONS_TEXT = (
    '[{"kind": "expedite", "title": "Upgrade", "description": "Fly the last leg."}]'
)
VERIFY_TEXT = '{"grounded": true, "issues": [], "summary": "Grounded."}'
DRAFT_TEXT = (
    "Subject: Update on shipment AGT-1: delay\n\n"
    "Dear Acme Parts,\n\n"
    "Your shipment AGT-1 is delayed. [POL-DELAY-01] applies. "
    "The next update will arrive within one business day."
)


def _route(system: str) -> str:
    if "extracting structured fields" in system:
        return "{}"
    if "classifying a shipment exception" in system:
        return CLASSIFY_TEXT
    if "proposing recovery options" in system:
        return OPTIONS_TEXT
    if "verifying whether a drafted" in system:
        return VERIFY_TEXT
    return DRAFT_TEXT


# --------------------------------------------------------------------------
# Scripted OpenAI fake: emits tool calls per the client's `script`
# --------------------------------------------------------------------------

class _ScriptedCompletions:
    def __init__(self, client: "ScriptedOpenAI") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, messages=None, tools=None):
        system = " ".join(messages[0]["content"].split())
        client = self._client
        if "diagnosing the root cause" not in system:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=_route(system)))]
            )
        # Diagnosis turns: follow the script until it runs out, then compose.
        tool_results = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
        step = len(tool_results)
        if tools is not None and step < len(client.script):
            name, arguments = client.script[step]
            tool_call = SimpleNamespace(
                id=f"call_{step}",
                function=SimpleNamespace(name=name, arguments=arguments),
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=DIAGNOSIS_TEXT))]
        )


class ScriptedOpenAI:
    instances: list["ScriptedOpenAI"] = []
    script: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_ScriptedCompletions(self))
        ScriptedOpenAI.instances.append(self)


# --------------------------------------------------------------------------
# Scripted Anthropic fake: one tool_use, then the composed answer
# --------------------------------------------------------------------------

class _ScriptedMessages:
    def __init__(self, client: "ScriptedAnthropic") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, system=None, messages=None, tools=None):
        system = " ".join((system or "").split())
        if "diagnosing the root cause" not in system:
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=_route(system))]
            )
        saw_tool_result = any(
            isinstance(m.get("content"), list) for m in messages if isinstance(m, dict)
        )
        if tools is not None and not saw_tool_result:
            block = SimpleNamespace(
                type="tool_use",
                id="tu_1",
                name="shipment_facts",
                input={},
            )
            return SimpleNamespace(content=[block])
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=DIAGNOSIS_TEXT)]
        )


class ScriptedAnthropic:
    instances: list["ScriptedAnthropic"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = _ScriptedMessages(self)
        ScriptedAnthropic.instances.append(self)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.tools_agent.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_openai(monkeypatch):
    ScriptedOpenAI.instances = []
    ScriptedOpenAI.script = []
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=ScriptedOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return ScriptedOpenAI


@pytest.fixture
def fake_anthropic(monkeypatch):
    ScriptedAnthropic.instances = []
    monkeypatch.setitem(
        sys.modules, "anthropic", SimpleNamespace(Anthropic=ScriptedAnthropic)
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from shipment_agent.model_backends import AnthropicBackend

    return AnthropicBackend


def _run(backend, priors=None):
    return run_shipment(
        ShipmentInput.model_validate(DELAY_SHIPMENT),
        backend=backend,
        retriever=KeywordRetriever(),
        priors=priors if priors is not None else PRIORS,
    )


# --------------------------------------------------------------------------
# The loop, end to end
# --------------------------------------------------------------------------

def test_openai_tool_loop_executes_tools_in_order(fake_openai):
    fake_openai.script = [
        ("search_policies", '{"query": "damage claim documentation"}'),
        ("carrier_history", '{"carrier": "Synthetic Carrier"}'),
    ]
    result = _run(OpenAIBackend())
    diagnosis = result.diagnosis
    assert diagnosis is not None and diagnosis.source == "llm"
    assert [t["name"] for t in diagnosis.tool_calls] == ["search_policies", "carrier_history"]
    assert "36.0" in diagnosis.root_cause
    diagnose_step = next(s for s in result.trace if s.name == "diagnose")
    assert any(d.startswith("tool call: search_policies") for d in diagnose_step.details)
    assert any(d.startswith("tool call: carrier_history") for d in diagnose_step.details)


def test_anthropic_tool_loop_executes_a_tool(fake_anthropic):
    result = _run(fake_anthropic())
    diagnosis = result.diagnosis
    assert diagnosis is not None and diagnosis.source == "llm"
    assert [t["name"] for t in diagnosis.tool_calls] == ["shipment_facts"]


def test_tool_call_cap_is_enforced(fake_openai, monkeypatch):
    monkeypatch.setenv("DIAGNOSIS_MAX_TOOL_CALLS", "2")
    fake_openai.script = [("shipment_facts", "{}")] * 6  # greedy model
    result = _run(OpenAIBackend())
    diagnosis = result.diagnosis
    assert diagnosis is not None and diagnosis.source == "llm"
    assert len(diagnosis.tool_calls) == 2  # budget spent, then it composed


def test_tool_error_degrades_to_composition_without_the_tool(fake_openai):
    # A dispatch that raises: the loop feeds the error back and the
    # model still composes; the run does not fail.
    backend = OpenAIBackend()

    def exploding_dispatch(name, args):
        raise RuntimeError("store is on fire")

    context = {
        "shipment_id": "AGT-1", "origin": "Memphis, TN", "destination": "Charlotte, NC",
        "carrier": "Synthetic Carrier", "exception_type": "delay", "severity": "high",
        "rationale": "r", "delay_hours": 36.0, "mismatches": [], "discrepancies": [],
        "latest_event": "", "condition_notes": "", "policy_details": [],
    }
    from shipment_agent.model_backends import DraftContext

    fake_openai.script = [("shipment_facts", "{}")]
    result = backend.diagnose_with_tools(
        DraftContext(context), TOOL_SPECS, exploding_dispatch, 4
    )
    assert result is not None
    assert result["tool_calls"][0]["name"] == "shipment_facts"
    assert "error" in result["tool_calls"][0]["summary"]


def test_mock_mode_has_no_tool_loop_and_says_so():
    result = run_shipment(
        ShipmentInput.model_validate(DELAY_SHIPMENT), backend=MockModelBackend()
    )
    assert result.diagnosis is not None
    assert result.diagnosis.tool_calls == []
    diagnose_step = next(s for s in result.trace if s.name == "diagnose")
    assert any("tool loop: provider-only" in d for d in diagnose_step.details)


# --------------------------------------------------------------------------
# The toolbox itself
# --------------------------------------------------------------------------

def _toolbox(priors=PRIORS) -> DiagnosisToolBox:
    return DiagnosisToolBox(
        shipment=ShipmentInput.model_validate(DELAY_SHIPMENT),
        classification={"exception_type": "delay", "severity": "high", "confidence": 0.94},
        delay_hours=36.0,
        mismatches=[],
        retriever=KeywordRetriever(),
        priors=priors,
    )


def test_toolbox_search_policies_returns_cited_policies():
    text, summary = _toolbox().dispatch("search_policies", {"query": "delay customer update"})
    assert "POL-" in text
    assert summary.startswith("search_policies(")


def test_toolbox_carrier_history_counts_exceptions_by_type():
    text, summary = _toolbox().dispatch("carrier_history", {"carrier": "Synthetic Carrier"})
    assert "damage×1" in text and "delay×1" in text
    assert "carrier_history" in summary


def test_toolbox_lane_history_reports_consignee_and_lane():
    text, _ = _toolbox().dispatch(
        "lane_history",
        {"consignee": "Acme Parts", "origin": "Memphis, TN", "destination": "Charlotte, NC"},
    )
    assert "2 prior exception(s) for consignee" in text
    assert "2 prior exception(s) on lane" in text


def test_toolbox_shipment_facts_are_the_computed_facts():
    text, _ = _toolbox().dispatch("shipment_facts", {})
    assert "computed delay_hours: 36.0" in text
    assert "exception_type: delay" in text


def test_toolbox_unknown_tool_raises_for_the_loop_to_absorb():
    with pytest.raises(ValueError, match="unknown tool"):
        _toolbox().dispatch("send_email", {})


def test_tool_call_budget_defaults_and_hard_cap(monkeypatch):
    assert diagnosis_max_tool_calls() == 4
    monkeypatch.setenv("DIAGNOSIS_MAX_TOOL_CALLS", "99")
    assert diagnosis_max_tool_calls() == 6  # hard cap
    monkeypatch.setenv("DIAGNOSIS_MAX_TOOL_CALLS", "1")
    assert diagnosis_max_tool_calls() == 1
