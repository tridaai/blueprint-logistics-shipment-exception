"""LLM-judge eval pack tests — the provider SDK is faked; no network,
no real key. These cover the pack's plumbing: loud failure without a
provider, per-case verdicts, the agreement summary, token/cost lines,
and a judge error failing the pack instead of passing silently.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "LLM_TIMEOUT_SECONDS", "RETRIEVER", "LLM_JUDGE_MODEL",
]

JUDGE_CLEAN = (
    '{"grounded": true, "invented_eta": false, "prohibited_promise": false,'
    ' "score": 0.95, "rationale": "Every claim traces to the verified facts."}'
)
JUDGE_DIRTY = (
    '{"grounded": false, "invented_eta": true, "prohibited_promise": false,'
    ' "score": 0.2, "rationale": "The draft states a delivery date not in the facts."}'
)


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "evals" / "run_llm_evals.py"
    spec = importlib.util.spec_from_file_location("run_llm_evals", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeCompletions:
    def __init__(self, client: "FakeEvalOpenAI") -> None:
        self._client = client

    def create(self, model=None, max_tokens=None, messages=None):
        system = " ".join(messages[0]["content"].split())
        user = messages[-1]["content"]
        client = self._client
        if "judging whether" in system:
            text = client.judge_text
        elif "extracting structured fields" in system:
            text = "{}"
        elif "classifying a shipment exception" in system:
            text = (
                '{"exception_type": "delay", "severity": "high",'
                ' "confidence": 0.9, "rationale": "Delay dominates."}'
            )
        elif "diagnosing the root cause" in system:
            text = '{"root_cause": "Schedule slip.", "summary": "Delayed."}'
        elif "proposing recovery options" in system:
            text = (
                '[{"kind": "expedite", "title": "Expedite",'
                ' "description": "Upgrade the remaining leg."},'
                ' {"kind": "wait_and_monitor", "title": "Hold",'
                '  "description": "Watch the next scan."}]'
            )
        else:
            sid = re.search(r"Shipment: (\S+)", user)
            shipment_id = sid.group(1) if sid else "UNKNOWN"
            text = (
                f"Subject: Update on shipment {shipment_id}: delay\n\n"
                f"Dear Synthetic Customer,\n\nYour shipment {shipment_id} is "
                "delayed. [POL-DELAY-01] applies. The next update will arrive "
                "within one business day."
            )
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=40)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=usage,
        )


class FakeEvalOpenAI:
    instances: list["FakeEvalOpenAI"] = []
    judge_text = JUDGE_CLEAN

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions(self))
        FakeEvalOpenAI.instances.append(self)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("shipment_agent.model_backends.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("shipment_agent.retriever.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def fake_openai(monkeypatch):
    FakeEvalOpenAI.instances = []
    FakeEvalOpenAI.judge_text = JUDGE_CLEAN
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeEvalOpenAI))
    monkeypatch.setenv("MODEL_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return FakeEvalOpenAI


def test_pack_fails_loudly_without_a_provider(capsys):
    module = _load_eval_module()
    exit_code = module.main(["--limit", "2"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "needs a real provider" in err
    assert "MODEL_BACKEND" in err


def test_pack_runs_cases_and_summarises(fake_openai, capsys):
    module = _load_eval_module()
    exit_code = module.main(["--limit", "2"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "D1:" in out and "D2:" in out
    assert "resolution=agree" in out  # fake LLM classifies delay, rules say delay
    assert "judge score 0.95" in out
    assert "tokens=" in out
    assert "Cross-check" in out
    assert "Judged drafts        : 2, clean 2" in out
    assert "Estimated cost" in out
    assert out.rstrip().endswith("PASS")


def test_pack_fails_on_ungrounded_draft(fake_openai, capsys):
    FakeEvalOpenAI.judge_text = JUDGE_DIRTY
    module = _load_eval_module()
    exit_code = module.main(["--limit", "1"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "invented_eta=True" in out
    assert out.rstrip().endswith("FAIL")


def test_pack_fails_on_judge_error(fake_openai, capsys):
    FakeEvalOpenAI.judge_text = "the judge rambles without json"
    module = _load_eval_module()
    exit_code = module.main(["--limit", "1"])
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "JUDGE ERROR" in out


def test_estimate_cost_table(fake_openai):
    module = _load_eval_module()
    usage = {"input_tokens": 1_000_000, "output_tokens": 500_000, "calls": 3}
    assert module.estimate_cost("gpt-4o-mini", usage) == pytest.approx(0.15 + 0.30)
    assert module.estimate_cost("some-unknown-model", usage) is None
    assert module.estimate_cost("gpt-4o-mini", None) is None
