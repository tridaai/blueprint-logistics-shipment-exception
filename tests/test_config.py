""".env loading: parsing, precedence, and backend selection driven by .env."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from shipment_agent.config import load_dotenv
from shipment_agent.model_backends import OpenAIBackend, get_backend

ENV_VARS = [
    "MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "LLM_TIMEOUT_SECONDS", "RETRIEVER",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def restore_dotenv_keys():
    """load_dotenv writes straight into os.environ — snapshot and restore
    the keys these tests load so nothing leaks into other test files."""
    import os

    keys = ["MODEL_BACKEND", "OPENAI_API_KEY", "OPENAI_MODEL", "LLM_TIMEOUT_SECONDS"]
    saved = {k: os.environ.get(k) for k in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_load_dotenv_reads_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment line\n"
        "\n"
        "MODEL_BACKEND=anthropic\n"
        "export OPENAI_API_KEY=sk-from-file\n"
        "OPENAI_MODEL=\"gpt-4o\"\n"
        "LLM_TIMEOUT_SECONDS='30'\n"
        "NOT_A_PAIR\n",
        encoding="utf-8",
    )
    loaded = load_dotenv(env_file)
    assert loaded == env_file
    import os

    assert os.environ["MODEL_BACKEND"] == "anthropic"
    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"  # export prefix ok
    assert os.environ["OPENAI_MODEL"] == "gpt-4o"  # quotes stripped
    assert os.environ["LLM_TIMEOUT_SECONDS"] == "30"


def test_load_dotenv_never_overrides_real_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-environment")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-from-file\n", encoding="utf-8")
    load_dotenv(env_file)
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-real-environment"


def test_load_dotenv_missing_file_returns_none(tmp_path):
    assert load_dotenv(tmp_path / "does-not-exist.env") is None


def test_dotenv_file_drives_backend_selection(tmp_path, monkeypatch):
    """End to end: a .env in the working directory selects the backend and
    supplies the key — no process environment involved."""
    (tmp_path / ".env").write_text(
        "MODEL_BACKEND=openai\nOPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    backend = get_backend()
    assert isinstance(backend, OpenAIBackend)
    assert captured["api_key"] == "sk-from-dotenv"
