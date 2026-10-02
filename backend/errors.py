"""Categorized LLM provider (Gemini / Groq) errors with safe, non-secret messages."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Gemini error codes
GEMINI_TIMEOUT = "GEMINI_TIMEOUT"
GEMINI_RATE_LIMITED = "GEMINI_RATE_LIMITED"
GEMINI_UNAVAILABLE = "GEMINI_UNAVAILABLE"
GEMINI_AUTH_ERROR = "GEMINI_AUTH_ERROR"
GEMINI_MODEL_ERROR = "GEMINI_MODEL_ERROR"
GEMINI_CONNECTION_ERROR = "GEMINI_CONNECTION_ERROR"

# Groq error codes
GROQ_TIMEOUT = "GROQ_TIMEOUT"
GROQ_RATE_LIMITED = "GROQ_RATE_LIMITED"
GROQ_UNAVAILABLE = "GROQ_UNAVAILABLE"
GROQ_AUTH_ERROR = "GROQ_AUTH_ERROR"
GROQ_MODEL_ERROR = "GROQ_MODEL_ERROR"

# Generic / infrastructure error codes
LLM_CONFIGURATION_ERROR = "LLM_CONFIGURATION_ERROR"
LLM_CONNECTION_ERROR = "LLM_CONNECTION_ERROR"
INTERNAL_ERROR = "INTERNAL_ERROR"

HTTP_STATUS_BY_CODE = {
    GEMINI_TIMEOUT: 504,
    GEMINI_RATE_LIMITED: 429,
    GEMINI_UNAVAILABLE: 503,
    GEMINI_AUTH_ERROR: 401,
    GEMINI_MODEL_ERROR: 400,
    GEMINI_CONNECTION_ERROR: 503,
    GROQ_TIMEOUT: 504,
    GROQ_RATE_LIMITED: 429,
    GROQ_UNAVAILABLE: 503,
    GROQ_AUTH_ERROR: 401,
    GROQ_MODEL_ERROR: 400,
    LLM_CONFIGURATION_ERROR: 500,
    LLM_CONNECTION_ERROR: 503,
    INTERNAL_ERROR: 500,
}

USER_MESSAGES = {
    GEMINI_TIMEOUT: "The Gemini provider timed out. Please try again.",
    GEMINI_RATE_LIMITED: "The Gemini provider is rate-limited (HTTP 429). Please wait and try again.",
    GEMINI_UNAVAILABLE: "The Gemini provider is temporarily unavailable. Please try again later.",
    GEMINI_AUTH_ERROR: "Gemini authentication failed. Check that GOOGLE_API_KEY is valid.",
    GEMINI_MODEL_ERROR: "The configured Gemini model is invalid or rejected by the provider.",
    GEMINI_CONNECTION_ERROR: "Could not reach the Gemini provider. Check network connectivity.",
    GROQ_TIMEOUT: "The Groq provider timed out. Please try again.",
    GROQ_RATE_LIMITED: "The Groq provider is rate-limited (HTTP 429). Please wait and try again.",
    GROQ_UNAVAILABLE: "The Groq provider is temporarily unavailable. Please try again later.",
    GROQ_AUTH_ERROR: "Groq authentication failed. Check that GROQ_API_KEY is valid.",
    GROQ_MODEL_ERROR: "The configured Groq model is invalid or rejected by the provider.",
    LLM_CONFIGURATION_ERROR: "LLM configuration error. Check your provider and API key settings.",
    LLM_CONNECTION_ERROR: "Could not reach the LLM provider. Check network connectivity.",
    INTERNAL_ERROR: "The query agent hit an unexpected internal error.",
}

_SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"gsk_[0-9A-Za-z_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+"),
)


def redact(text: str) -> str:
    """Remove API keys and credential-like values from diagnostic text."""
    if not text:
        return ""
    redacted = text
    for env_name in ("GOOGLE_API_KEY", "GROQ_API_KEY"):
        env_val = os.getenv(env_name)
        if env_val:
            redacted = redacted.replace(env_val, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted[:2000]


class LLMProviderError(Exception):
    def __init__(self, code: str, message: str | None = None, log_detail: str = ""):
        self.code = code if code in HTTP_STATUS_BY_CODE else INTERNAL_ERROR
        self.http_status = HTTP_STATUS_BY_CODE[self.code]
        self.message = message or USER_MESSAGES.get(self.code, USER_MESSAGES[INTERNAL_ERROR])
        self.log_detail = redact(log_detail)
        super().__init__(self.message)

    def as_state(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "http_status": self.http_status,
            "log_detail": self.log_detail,
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> "LLMProviderError":
        return cls(
            payload.get("code") or INTERNAL_ERROR,
            payload.get("message"),
            payload.get("log_detail") or "",
        )


# Backward compatibility aliases
GeminiProviderError = LLMProviderError
GroqProviderError = LLMProviderError


def _walk_exceptions(exc: BaseException):
    seen: set[int] = set()
    stack: list[BaseException | None] = [exc]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        stack.append(current.__cause__)
        stack.append(current.__context__)


def _status_from_exception(exc: BaseException) -> int | None:
    for attr in ("code", "status_code", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 400 <= value <= 599:
            return value
    return None


def classify_llm_error(exc: BaseException, provider: str | None = None) -> LLMProviderError:
    """Map provider/network exceptions to a categorized LLMProviderError."""
    if isinstance(exc, LLMProviderError):
        return exc

    joined = " ".join(
        redact(f"{type(item).__name__}: {item}") for item in _walk_exceptions(exc)
    )
    lowered = joined.lower()

    status = None
    for item in _walk_exceptions(exc):
        status = _status_from_exception(item)
        if status is not None:
            break

    names = {type(item).__name__.lower() for item in _walk_exceptions(exc)}
    modules = " ".join(type(item).__module__.lower() for item in _walk_exceptions(exc))

    # Detect active provider if not explicitly given
    if not provider:
        if "groq" in modules or any("groq" in n for n in names) or "groq" in lowered:
            provider = "groq"
        elif "google" in modules or "genai" in modules or "gemini" in lowered:
            provider = "gemini"
        else:
            provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

    # 1. Missing API Key / Configuration error
    if "groq_api_key is missing" in lowered:
        code = LLM_CONFIGURATION_ERROR
    elif "google_api_key is missing" in lowered or "api_key is missing" in lowered:
        code = LLM_CONFIGURATION_ERROR if provider == "groq" else GEMINI_AUTH_ERROR
    elif "unsupported llm_provider" in lowered:
        code = LLM_CONFIGURATION_ERROR

    # 2. Groq-specific classification
    elif provider == "groq":
        if status == 429 or "ratelimiterror" in names or "rate limit" in lowered or "resource_exhausted" in lowered:
            code = GROQ_RATE_LIMITED
        elif status in {408, 504} or "apitimeouterror" in names or "timeoutexception" in names or "timeout" in lowered or "timed out" in lowered:
            code = GROQ_TIMEOUT
        elif status in {401, 403} or "authenticationerror" in names or "permissiondeniederror" in names or ("invalid" in lowered and "api key" in lowered):
            code = GROQ_AUTH_ERROR
        elif status == 404 or "notfounderror" in names or "model not found" in lowered or "invalid model" in lowered or "model_not_found" in lowered:
            code = GROQ_MODEL_ERROR
        elif status in {500, 502, 503} or "internalservererror" in names or "unavailable" in lowered or "overloaded" in lowered:
            code = GROQ_UNAVAILABLE
        elif "apiconnectionerror" in names or "connectionerror" in names or "connecterror" in names or "networkerror" in names or "connection" in lowered:
            code = LLM_CONNECTION_ERROR
        else:
            code = INTERNAL_ERROR

    # 3. Gemini-specific classification
    else:
        if status == 429 or "429" in lowered or "resource_exhausted" in lowered or "rate limit" in lowered:
            code = GEMINI_RATE_LIMITED
        elif status in {408, 504} or "timeout" in lowered or "timed out" in lowered or "timeoutexception" in names:
            code = GEMINI_TIMEOUT
        elif status in {401, 403} or "unauthenticated" in lowered or "permission_denied" in lowered or ("api key" in lowered and "invalid" in lowered):
            code = GEMINI_AUTH_ERROR
        elif (
            status == 404
            or "not found" in lowered
            or "invalid model" in lowered
            or "is not supported" in lowered
            or "unknown model" in lowered
        ):
            code = GEMINI_MODEL_ERROR
        elif status in {500, 502, 503} or "unavailable" in lowered or "overloaded" in lowered:
            code = GEMINI_UNAVAILABLE
        elif (
            "connecterror" in names
            or "connecttimeout" in names
            or "networkerror" in names
            or "connectionerror" in names
            or "connection" in lowered
            or "network" in lowered
        ):
            code = GEMINI_CONNECTION_ERROR
        elif "google" in modules or "genai" in modules or "gemini" in lowered:
            code = GEMINI_UNAVAILABLE
        else:
            code = INTERNAL_ERROR

    error = LLMProviderError(code, USER_MESSAGES.get(code, USER_MESSAGES[INTERNAL_ERROR]), joined)
    logger.error("LLM error classified as %s (%s): %s", error.code, provider, error.log_detail)
    return error


# Backward compatibility alias
classify_gemini_error = classify_llm_error
