"""LLM Provider Factory supporting Google Gemini and Groq with bounded timeout and retries."""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from .errors import (
    LLM_CONFIGURATION_ERROR,
    LLMProviderError,
    classify_llm_error,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: float = 15.0
DEFAULT_MAX_RETRIES: int = 0
DEFAULT_TEMPERATURE: float = 0.0

DEFAULT_GEMINI_MODEL: str = "gemini-3.7-flash"
DEFAULT_GROQ_MODEL: str = "llama-3.3-70b-versatile"


def get_active_provider() -> str:
    """Return the configured active provider ('gemini' or 'groq'). Defaults to 'gemini'."""
    val = os.getenv("LLM_PROVIDER")
    if not val or not val.strip():
        return "gemini"
    return val.strip().lower()


def get_fallback_provider() -> str | None:
    """Return the optional fallback provider if configured and different from active."""
    val = os.getenv("LLM_FALLBACK_PROVIDER")
    if not val or not val.strip():
        return None
    fallback = val.strip().lower()
    if fallback == get_active_provider():
        return None
    return fallback


def create_llm(provider: str | None = None) -> Any:
    """Instantiate and return the requested or active Chat model client."""
    chosen_provider = (provider or get_active_provider()).strip().lower()

    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))

    if chosen_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or not api_key.strip():
            raise LLMProviderError(
                LLM_CONFIGURATION_ERROR,
                "GOOGLE_API_KEY is missing. Add it to the .env file before using the Gemini provider.",
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        logger.info("Initializing Gemini LLM client (model=%s, timeout=%ss, max_retries=%s)", model, timeout, max_retries)
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            google_api_key=api_key.strip(),
            timeout=timeout,
            max_retries=max_retries,
        )

    elif chosen_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or not api_key.strip():
            raise LLMProviderError(
                LLM_CONFIGURATION_ERROR,
                "GROQ_API_KEY is missing. Add it to the .env file before using the Groq provider.",
            )

        from langchain_groq import ChatGroq

        model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL).strip()
        logger.info("Initializing Groq LLM client (model=%s, timeout=%ss, max_retries=%s)", model, timeout, max_retries)
        return ChatGroq(
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            groq_api_key=api_key.strip(),
            timeout=timeout,
            max_retries=max_retries,
        )

    else:
        raise LLMProviderError(
            LLM_CONFIGURATION_ERROR,
            f"Unsupported LLM_PROVIDER '{chosen_provider}'. Supported values are 'gemini' and 'groq'.",
        )


def get_llm(provider: str | None = None) -> Any:
    """Convenience alias for creating an LLM client for the active or requested provider."""
    return create_llm(provider)
