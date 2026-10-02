"""Tests for Groq LLM provider integration.

All tests are fully mocked — no real Groq API calls are made.
Tests verify:
- Provider selection from LLM_PROVIDER env var
- Missing API key → LLM_CONFIGURATION_ERROR
- Groq timeout / rate-limit / 503 / auth / model errors → correct error codes
- Full mocked Groq flow returning 8 employee rows
- FastAPI /query endpoint with mocked Groq provider
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.agent import ScopeResult, SQLResult, run_agent
from backend.errors import (
    GROQ_AUTH_ERROR,
    GROQ_MODEL_ERROR,
    GROQ_RATE_LIMITED,
    GROQ_TIMEOUT,
    GROQ_UNAVAILABLE,
    INTERNAL_ERROR,
    LLM_CONFIGURATION_ERROR,
    LLMProviderError,
    classify_llm_error,
)
from backend.llm import create_llm, get_active_provider, get_fallback_provider
from backend.main import app


# ---------------------------------------------------------------------------
# Fake LLM helpers
# ---------------------------------------------------------------------------

class FakeGroqErrorLLM:
    """Raises a given exception on every invoke (and on with_structured_output)."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def with_structured_output(self, _model):
        return self

    def invoke(self, _prompt):
        raise self._exc


class FakeGroqSuccessLLM:
    """Returns valid structured / text responses, simulating a working Groq provider."""

    def __init__(self):
        self._target_model = None

    def with_structured_output(self, model):
        self._target_model = model
        return self

    def invoke(self, prompt):
        if self._target_model is ScopeResult:
            return ScopeResult(in_scope=True, reason="Valid database query")
        if self._target_model is SQLResult:
            return SQLResult(
                sql="SELECT employee_id, name, department_id, job_title, salary, hire_date FROM employees",
                ambiguity="",
            )

        class TextMsg:
            text = "Mocked Groq explanation or optimization text."
            content = "Mocked Groq explanation or optimization text."

        return TextMsg()


# ---------------------------------------------------------------------------
# Fake Groq exception types (mirror groq SDK hierarchy without importing it)
# ---------------------------------------------------------------------------

class _FakeGroqAPIError(Exception):
    """Minimal stand-in for groq.APIError with a numeric status_code attribute."""

    def __init__(self, status_code: int, message: str = "groq api error"):
        super().__init__(message)
        self.status_code = status_code
        self.__module__ = "groq"


class _FakeGroqAPITimeoutError(_FakeGroqAPIError):
    """Stand-in for groq.APITimeoutError."""

    def __init__(self, message: str = "request timed out"):
        super().__init__(408, message)
        self.__class__.__name__ = "APITimeoutError"
        self.__module__ = "groq"


class _FakeGroqRateLimitError(_FakeGroqAPIError):
    """Stand-in for groq.RateLimitError (HTTP 429)."""

    def __init__(self, message: str = "rate limit exceeded"):
        super().__init__(429, message)
        self.__module__ = "groq"


class _FakeGroqAuthError(_FakeGroqAPIError):
    """Stand-in for groq.AuthenticationError (HTTP 401)."""

    def __init__(self, message: str = "invalid api key"):
        super().__init__(401, message)
        self.__module__ = "groq"


class _FakeGroqNotFoundError(_FakeGroqAPIError):
    """Stand-in for groq.NotFoundError / model not found (HTTP 404)."""

    def __init__(self, message: str = "model not found"):
        super().__init__(404, message)
        self.__module__ = "groq"


class _FakeGroqServerError(_FakeGroqAPIError):
    """Stand-in for groq.InternalServerError / service unavailable (HTTP 503)."""

    def __init__(self, message: str = "service unavailable"):
        super().__init__(503, message)
        self.__module__ = "groq"


# ===========================================================================
# 1. Provider selection tests (llm.py)
# ===========================================================================

def test_get_active_provider_defaults_to_gemini(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_active_provider() == "gemini"


def test_get_active_provider_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert get_active_provider() == "gemini"


def test_get_active_provider_groq(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert get_active_provider() == "groq"


def test_get_active_provider_uppercase_normalized(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "GROQ")
    assert get_active_provider() == "groq"


def test_get_active_provider_whitespace_normalized(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  groq  ")
    assert get_active_provider() == "groq"


def test_get_fallback_provider_not_set(monkeypatch):
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    assert get_fallback_provider() is None


def test_get_fallback_provider_same_as_active_ignored(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "gemini")
    assert get_fallback_provider() is None


def test_get_fallback_provider_different(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "groq")
    assert get_fallback_provider() == "groq"


# ===========================================================================
# 2. create_llm() missing API key tests
# ===========================================================================

def test_create_llm_groq_missing_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMProviderError) as exc_info:
        create_llm("groq")
    assert exc_info.value.code == LLM_CONFIGURATION_ERROR


def test_create_llm_gemini_missing_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMProviderError) as exc_info:
        create_llm("gemini")
    assert exc_info.value.code == LLM_CONFIGURATION_ERROR


def test_create_llm_unsupported_provider_raises(monkeypatch):
    with pytest.raises(LLMProviderError) as exc_info:
        create_llm("openai")
    assert exc_info.value.code == LLM_CONFIGURATION_ERROR


# ===========================================================================
# 3. classify_llm_error() for Groq exceptions
# ===========================================================================

def test_classify_groq_timeout():
    exc = _FakeGroqAPITimeoutError("Groq timed out after 15s")
    error = classify_llm_error(exc, provider="groq")
    assert error.code == GROQ_TIMEOUT
    assert error.http_status == 504


def test_classify_groq_rate_limited():
    exc = _FakeGroqRateLimitError("rate limit exceeded")
    error = classify_llm_error(exc, provider="groq")
    assert error.code == GROQ_RATE_LIMITED
    assert error.http_status == 429


def test_classify_groq_auth_error():
    exc = _FakeGroqAuthError("invalid api key")
    error = classify_llm_error(exc, provider="groq")
    assert error.code == GROQ_AUTH_ERROR
    assert error.http_status == 401


def test_classify_groq_model_error():
    exc = _FakeGroqNotFoundError("model not found")
    error = classify_llm_error(exc, provider="groq")
    assert error.code == GROQ_MODEL_ERROR
    assert error.http_status == 400


def test_classify_groq_unavailable():
    exc = _FakeGroqServerError("service unavailable")
    error = classify_llm_error(exc, provider="groq")
    assert error.code == GROQ_UNAVAILABLE
    assert error.http_status == 503


def test_classify_groq_rate_limited_via_status_code():
    """Verify status_code attribute on exception drives classification."""
    exc = _FakeGroqAPIError(429, "too many requests")
    error = classify_llm_error(exc, provider="groq")
    assert error.code == GROQ_RATE_LIMITED


def test_classify_groq_timeout_via_lowered_text():
    """'timed out' text in message drives GROQ_TIMEOUT even with generic exception."""
    exc = RuntimeError("groq request timed out after 15 seconds")
    # Provide provider hint so it's treated as groq
    error = classify_llm_error(exc, provider="groq")
    assert error.code == GROQ_TIMEOUT


def test_classify_groq_unexpected():
    exc = RuntimeError("weird groq crash")
    error = classify_llm_error(exc, provider="groq")
    assert error.code == INTERNAL_ERROR
    # Message must NOT expose raw exception text
    assert "weird groq crash" not in error.message


# ===========================================================================
# 4. Agent-level Groq error propagation (using monkeypatched get_llm)
# ===========================================================================

def test_agent_groq_timeout_controlled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(_FakeGroqAPITimeoutError("groq timed out")),
    )
    result = run_agent("show all employees")
    assert result["error_code"] == GROQ_TIMEOUT
    assert result["rows"] == []
    assert "timed out" in result["message"].lower() or "groq" in result["message"].lower()


def test_agent_groq_rate_limited_controlled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(_FakeGroqRateLimitError("rate limit exceeded")),
    )
    result = run_agent("show all employees")
    assert result["error_code"] == GROQ_RATE_LIMITED


def test_agent_groq_unavailable_controlled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(_FakeGroqServerError("service unavailable")),
    )
    result = run_agent("show all employees")
    assert result["error_code"] == GROQ_UNAVAILABLE


def test_agent_groq_auth_error_controlled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(_FakeGroqAuthError("invalid api key")),
    )
    result = run_agent("show all employees")
    assert result["error_code"] == GROQ_AUTH_ERROR


def test_agent_groq_model_error_controlled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(_FakeGroqNotFoundError("model not found")),
    )
    result = run_agent("show all employees")
    assert result["error_code"] == GROQ_MODEL_ERROR


def test_agent_groq_unexpected_controlled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(RuntimeError("unexpected groq crash")),
    )
    result = run_agent("show all employees")
    assert result["error_code"] == INTERNAL_ERROR
    # Raw crash text must NOT leak into user-facing message
    assert "unexpected groq crash" not in result["message"]


def test_agent_groq_config_error_propagated(monkeypatch):
    """Simulate create_llm() raising LLM_CONFIGURATION_ERROR when key is absent."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def _fail_create(*_args, **_kwargs):
        raise LLMProviderError(
            LLM_CONFIGURATION_ERROR,
            "GROQ_API_KEY is missing. Add it to the .env file before using the Groq provider.",
        )

    monkeypatch.setattr("backend.agent.get_llm", _fail_create)
    result = run_agent("show all employees")
    assert result["error_code"] == LLM_CONFIGURATION_ERROR
    assert result["rows"] == []


# ===========================================================================
# 5. Full mocked Groq success flow
# ===========================================================================

def test_agent_groq_success_flow(monkeypatch):
    """End-to-end mock: full LangGraph execution with Groq returning 8 employee rows."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr("backend.agent.get_llm", lambda: FakeGroqSuccessLLM())
    result = run_agent("show all employees")
    assert result.get("error_code") is None
    assert result["in_scope"] is True
    assert "SELECT" in result["sql"]
    assert len(result["rows"]) == 8
    assert "employee_id" in result["columns"]
    assert result["explanation"] != ""
    assert result["optimization"] != ""


# ===========================================================================
# 6. FastAPI /query endpoint with Groq provider
# ===========================================================================

def test_fastapi_groq_timeout_returns_504(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(_FakeGroqAPITimeoutError("groq timed out")),
    )
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employees"})
    assert resp.status_code == 504
    data = resp.json()
    assert data["error_code"] == GROQ_TIMEOUT


def test_fastapi_groq_rate_limited_returns_429(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(_FakeGroqRateLimitError("rate limit exceeded")),
    )
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employees"})
    assert resp.status_code == 429
    data = resp.json()
    assert data["error_code"] == GROQ_RATE_LIMITED


def test_fastapi_groq_unavailable_returns_503(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeGroqErrorLLM(_FakeGroqServerError("service unavailable")),
    )
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employees"})
    assert resp.status_code == 503
    data = resp.json()
    assert data["error_code"] == GROQ_UNAVAILABLE


def test_fastapi_groq_missing_key_returns_500(monkeypatch):
    """When GROQ_API_KEY is absent, /query must return 500 with LLM_CONFIGURATION_ERROR."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def _raise_config(*_args, **_kwargs):
        raise LLMProviderError(
            LLM_CONFIGURATION_ERROR,
            "GROQ_API_KEY is missing. Add it to the .env file before using the Groq provider.",
        )

    monkeypatch.setattr("backend.agent.get_llm", _raise_config)
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employees"})
    assert resp.status_code == 500
    data = resp.json()
    assert data["error_code"] == LLM_CONFIGURATION_ERROR


def test_fastapi_groq_success_flow_returns_200(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr("backend.agent.get_llm", lambda: FakeGroqSuccessLLM())
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employees"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["in_scope"] is True
    assert "SELECT" in data["sql"]
    assert len(data["rows"]) == 8
    assert data.get("error_code") is None
