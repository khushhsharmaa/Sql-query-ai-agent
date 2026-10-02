import pytest
from google.genai.errors import APIError, ClientError
from fastapi.testclient import TestClient

from backend.agent import ScopeResult, SQLResult, run_agent
from backend.errors import (
    GEMINI_AUTH_ERROR,
    GEMINI_CONNECTION_ERROR,
    GEMINI_MODEL_ERROR,
    GEMINI_RATE_LIMITED,
    GEMINI_TIMEOUT,
    GEMINI_UNAVAILABLE,
    INTERNAL_ERROR,
    classify_gemini_error,
)
from backend.main import app


@pytest.fixture(autouse=True)
def use_gemini_provider_for_tests(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")


class FakeErrorLLM:
    def __init__(self, exc):
        self._exc = exc

    def with_structured_output(self, _model):
        return self

    def invoke(self, _prompt):
        raise self._exc


class FakeSuccessLLM:
    """Simulates a fully functioning Gemini provider returning structured and text outputs."""
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
        # For optimization and explanation prompts
        class TextMsg:
            text = "Mocked explanation or optimization text."
            content = "Mocked explanation or optimization text."
        return TextMsg()


def test_classify_timeout():
    error = classify_gemini_error(TimeoutError("gemini request timed out"))
    assert error.code == GEMINI_TIMEOUT
    assert error.http_status == 504


def test_classify_429():
    error = classify_gemini_error(
        APIError(429, {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"})
    )
    assert error.code == GEMINI_RATE_LIMITED
    assert error.http_status == 429


def test_classify_503():
    error = classify_gemini_error(
        APIError(503, {"status": "UNAVAILABLE", "message": "service unavailable"})
    )
    assert error.code == GEMINI_UNAVAILABLE
    assert error.http_status == 503


def test_classify_auth_error():
    error = classify_gemini_error(
        ClientError(401, {"status": "UNAUTHENTICATED", "message": "API key not valid"})
    )
    assert error.code == GEMINI_AUTH_ERROR
    assert error.http_status == 401


def test_classify_model_error():
    error = classify_gemini_error(
        ClientError(404, {"status": "NOT_FOUND", "message": "models/unknown is not supported"})
    )
    assert error.code == GEMINI_MODEL_ERROR
    assert error.http_status == 400


def test_classify_connection_error():
    error = classify_gemini_error(ConnectionError("Failed to connect to host"))
    assert error.code == GEMINI_CONNECTION_ERROR
    assert error.http_status == 503


def test_classify_unexpected():
    error = classify_gemini_error(RuntimeError("unexpected boom"))
    assert error.code == INTERNAL_ERROR
    assert error.http_status == 500
    assert "boom" in error.log_detail


def test_agent_timeout_is_controlled(monkeypatch):
    monkeypatch.setattr("backend.agent.get_llm", lambda: FakeErrorLLM(TimeoutError("timed out")))
    result = run_agent("show all employee")
    assert result["error_code"] == GEMINI_TIMEOUT
    assert result["rows"] == []
    assert "timed out" in result["message"].lower() or "timeout" in result["message"].lower()


def test_agent_429_is_controlled(monkeypatch):
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeErrorLLM(APIError(429, {"status": "RESOURCE_EXHAUSTED", "message": "quota"})),
    )
    result = run_agent("show all employee")
    assert result["error_code"] == GEMINI_RATE_LIMITED


def test_agent_503_is_controlled(monkeypatch):
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeErrorLLM(APIError(503, {"status": "UNAVAILABLE", "message": "down"})),
    )
    result = run_agent("show all employee")
    assert result["error_code"] == GEMINI_UNAVAILABLE


def test_agent_auth_is_controlled(monkeypatch):
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeErrorLLM(ClientError(401, {"status": "UNAUTHENTICATED", "message": "API key not valid"})),
    )
    result = run_agent("show all employee")
    assert result["error_code"] == GEMINI_AUTH_ERROR


def test_agent_model_error_is_controlled(monkeypatch):
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeErrorLLM(ClientError(404, {"status": "NOT_FOUND", "message": "model not found"})),
    )
    result = run_agent("show all employee")
    assert result["error_code"] == GEMINI_MODEL_ERROR


def test_agent_unexpected_is_controlled(monkeypatch):
    monkeypatch.setattr("backend.agent.get_llm", lambda: FakeErrorLLM(RuntimeError("weird provider crash")))
    result = run_agent("show all employee")
    assert result["error_code"] == INTERNAL_ERROR
    assert "weird provider crash" not in result["message"]


def test_mocked_success_flow(monkeypatch):
    """End-to-end mock test verifying full graph execution and SQLite querying."""
    monkeypatch.setattr("backend.agent.get_llm", lambda: FakeSuccessLLM())
    result = run_agent("show all employee")
    assert result.get("error_code") is None
    assert result["in_scope"] is True
    assert "SELECT" in result["sql"]
    assert len(result["rows"]) == 8
    assert "employee_id" in result["columns"]
    assert result["explanation"] != ""
    assert result["optimization"] != ""


def test_fastapi_health():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_fastapi_query_empty_validation():
    client = TestClient(app)
    resp = client.post("/query", json={"question": "   "})
    assert resp.status_code == 422


def test_fastapi_query_provider_timeout_handled(monkeypatch):
    monkeypatch.setattr("backend.agent.get_llm", lambda: FakeErrorLLM(TimeoutError("timed out")))
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employee"})
    assert resp.status_code == 504
    data = resp.json()
    assert data["error_code"] == GEMINI_TIMEOUT
    assert "timed out" in data["message"].lower()


def test_fastapi_query_provider_429_handled(monkeypatch):
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeErrorLLM(APIError(429, {"status": "RESOURCE_EXHAUSTED", "message": "quota exceeded"})),
    )
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employee"})
    assert resp.status_code == 429
    data = resp.json()
    assert data["error_code"] == GEMINI_RATE_LIMITED


def test_fastapi_query_provider_503_handled(monkeypatch):
    monkeypatch.setattr(
        "backend.agent.get_llm",
        lambda: FakeErrorLLM(APIError(503, {"status": "UNAVAILABLE", "message": "service unavailable"})),
    )
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employee"})
    assert resp.status_code == 503
    data = resp.json()
    assert data["error_code"] == GEMINI_UNAVAILABLE


def test_fastapi_query_success_flow(monkeypatch):
    monkeypatch.setattr("backend.agent.get_llm", lambda: FakeSuccessLLM())
    client = TestClient(app)
    resp = client.post("/query", json={"question": "show all employee"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["in_scope"] is True
    assert "SELECT" in data["sql"]
    assert len(data["rows"]) == 8
    assert data.get("error_code") is None
