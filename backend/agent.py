import logging
import os
import re
from typing import TypedDict, Any

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from .db import get_schema_text, explain_query, execute_select
from .errors import LLMProviderError, GeminiProviderError, classify_llm_error
from .llm import get_llm, create_llm, get_fallback_provider
from .prompts import SCOPE_PROMPT, SQL_PROMPT, OPTIMIZE_PROMPT, EXPLAIN_PROMPT

logger = logging.getLogger(__name__)


class ScopeResult(BaseModel):
    in_scope: bool
    reason: str = Field(description="Short reason for the classification.")


class SQLResult(BaseModel):
    sql: str
    ambiguity: str = ""


class AgentState(TypedDict, total=False):
    question: str
    schema: str
    in_scope: bool
    scope_reason: str
    sql: str
    ambiguity: str
    validation_error: str
    optimization: str
    explanation: str
    columns: list[str]
    rows: list[dict[str, Any]]
    error: str
    final_message: str
    provider_error: dict[str, Any]


def _llm_invoke(llm: Any, prompt: Any, structured_model: type[BaseModel] | None = None) -> Any:
    try:
        return llm.invoke(prompt)
    except LLMProviderError:
        raise
    except Exception as primary_exc:
        fallback_name = get_fallback_provider()
        if fallback_name:
            try:
                logger.warning("Primary LLM invocation failed; attempting fallback to '%s'", fallback_name)
                fallback_llm = create_llm(fallback_name)
                if structured_model is not None and hasattr(fallback_llm, "with_structured_output"):
                    fallback_llm = fallback_llm.with_structured_output(structured_model)
                return fallback_llm.invoke(prompt)
            except Exception as fb_exc:
                logger.error("Fallback LLM '%s' also failed: %s", fallback_name, fb_exc)
                raise classify_llm_error(primary_exc) from primary_exc
        raise classify_llm_error(primary_exc) from primary_exc


def _provider_failure(exc: Exception) -> dict[str, Any]:
    error = classify_llm_error(exc)
    logger.error("LLM node failed (%s): %s", error.code, error.log_detail)
    return {"provider_error": error.as_state()}


def _extract_text(result: Any) -> str:
    if hasattr(result, "text") and isinstance(result.text, str) and result.text:
        return result.text.strip()
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts).strip()
    return str(content).strip() if content is not None else ""


def _parse_structured_result(result: Any, model_class: type[BaseModel]) -> Any:
    if isinstance(result, model_class):
        return result
    if isinstance(result, dict):
        try:
            return model_class.model_validate(result)
        except Exception:
            return result
    text = _extract_text(result)
    if text and "{" in text and "}" in text:
        import json
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            return model_class.model_validate(data)
        except Exception:
            pass
    return result


def scope_node(state: AgentState):
    try:
        base_llm = get_llm()
        llm = (
            base_llm.with_structured_output(ScopeResult)
            if hasattr(base_llm, "with_structured_output")
            else base_llm
        )
        raw = _llm_invoke(llm, SCOPE_PROMPT.format(question=state["question"]), structured_model=ScopeResult)
        result = _parse_structured_result(raw, ScopeResult)
        if isinstance(result, ScopeResult):
            in_scope = result.in_scope
            reason = result.reason
        elif isinstance(result, dict):
            in_scope = bool(result.get("in_scope", True))
            reason = str(result.get("reason", ""))
        else:
            in_scope = bool(getattr(result, "in_scope", True))
            reason = str(getattr(result, "reason", ""))
        return {
            "in_scope": in_scope,
            "scope_reason": reason,
        }
    except Exception as exc:
        return _provider_failure(exc)


def schema_node(state: AgentState):
    return {"schema": get_schema_text()}


def generate_sql_node(state: AgentState):
    try:
        base_llm = get_llm()
        llm = (
            base_llm.with_structured_output(SQLResult)
            if hasattr(base_llm, "with_structured_output")
            else base_llm
        )
        raw = _llm_invoke(
            llm,
            SQL_PROMPT.format(
                schema=state["schema"],
                question=state["question"],
            ),
            structured_model=SQLResult,
        )
        result = _parse_structured_result(raw, SQLResult)
        if isinstance(result, SQLResult):
            sql = result.sql
            ambiguity = result.ambiguity
        elif isinstance(result, dict):
            sql = result.get("sql", "")
            ambiguity = result.get("ambiguity", "")
        else:
            sql = getattr(result, "sql", "")
            ambiguity = getattr(result, "ambiguity", "")
        return {"sql": str(sql or "").strip(), "ambiguity": str(ambiguity or "").strip()}
    except Exception as exc:
        return _provider_failure(exc)


FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|REPLACE|CREATE|ATTACH|DETACH|"
    r"VACUUM|REINDEX|PRAGMA|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def validate_sql_node(state: AgentState):
    sql = state.get("sql", "").strip().rstrip(";")
    if not sql:
        return {"validation_error": "The agent did not produce a SQL query."}

    if ";" in sql:
        return {"validation_error": "Only one SQL statement is allowed."}

    if FORBIDDEN.search(sql):
        return {"validation_error": "Only read-only SELECT queries are allowed."}

    if not re.match(r"^(SELECT|WITH)\b", sql, re.IGNORECASE):
        return {"validation_error": "Only SELECT/WITH read-only queries are allowed."}

    try:
        explain_query(sql)
    except Exception as exc:
        return {"validation_error": f"SQL validation failed: {exc}"}

    return {"sql": sql, "validation_error": ""}


def optimization_node(state: AgentState):
    try:
        llm = get_llm()
        result = _llm_invoke(
            llm,
            OPTIMIZE_PROMPT.format(sql=state["sql"], schema=state["schema"]),
        )
        return {"optimization": _extract_text(result)}
    except Exception as exc:
        return _provider_failure(exc)


def execute_node(state: AgentState):
    try:
        columns, rows = execute_select(state["sql"])
        return {"columns": columns, "rows": rows, "error": ""}
    except Exception as exc:
        logger.exception("SQLite execution failed")
        return {"error": str(exc)}


def explain_node(state: AgentState):
    try:
        llm = get_llm()
        result = _llm_invoke(llm, EXPLAIN_PROMPT.format(sql=state["sql"]))
        return {"explanation": _extract_text(result)}
    except Exception as exc:
        return _provider_failure(exc)


def _has_provider_error(state: AgentState) -> bool:
    return bool(state.get("provider_error"))


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("scope", scope_node)
    graph.add_node("schema", schema_node)
    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("validate", validate_sql_node)
    graph.add_node("optimize", optimization_node)
    graph.add_node("execute", execute_node)
    graph.add_node("explain", explain_node)

    graph.set_entry_point("scope")

    def after_scope(state):
        if _has_provider_error(state):
            return END
        return "schema" if state.get("in_scope") else END

    graph.add_conditional_edges("scope", after_scope, {"schema": "schema", END: END})
    graph.add_edge("schema", "generate_sql")

    def after_generate(state):
        if _has_provider_error(state):
            return END
        return "validate"

    graph.add_conditional_edges(
        "generate_sql",
        after_generate,
        {"validate": "validate", END: END},
    )

    def after_validation(state):
        if _has_provider_error(state):
            return END
        return "optimize" if not state.get("validation_error") else END

    graph.add_conditional_edges(
        "validate",
        after_validation,
        {"optimize": "optimize", END: END},
    )

    def after_optimize(state):
        if _has_provider_error(state):
            return END
        return "execute"

    graph.add_conditional_edges(
        "optimize",
        after_optimize,
        {"execute": "execute", END: END},
    )

    def after_execute(state):
        if _has_provider_error(state) or state.get("error"):
            return END
        return "explain"

    graph.add_conditional_edges(
        "execute",
        after_execute,
        {"explain": "explain", END: END},
    )
    graph.add_edge("explain", END)
    return graph.compile()


GRAPH = build_graph()


def _empty_payload(**overrides):
    payload = {
        "in_scope": True,
        "message": "",
        "sql": "",
        "explanation": "",
        "optimization": "",
        "columns": [],
        "rows": [],
    }
    payload.update(overrides)
    return payload


def run_agent(question: str):
    try:
        state = GRAPH.invoke({"question": question})
    except Exception as exc:
        error = classify_llm_error(exc)
        logger.error("LangGraph invocation failed (%s): %s", error.code, error.log_detail)
        return _empty_payload(
            message=error.message,
            error_code=error.code,
        )

    if state.get("provider_error"):
        error = LLMProviderError.from_state(state["provider_error"])
        return _empty_payload(
            in_scope=state.get("in_scope", True),
            message=error.message,
            sql=state.get("sql", ""),
            explanation=state.get("explanation", ""),
            optimization=state.get("optimization", ""),
            columns=state.get("columns") or [],
            rows=state.get("rows") or [],
            error_code=error.code,
        )

    if not state.get("in_scope"):
        return _empty_payload(
            in_scope=False,
            message="I'm designed to assist only with SQL and database-related tasks.",
            explanation=state.get("scope_reason", ""),
        )

    if state.get("validation_error"):
        return _empty_payload(
            message=state["validation_error"],
            sql=state.get("sql", ""),
        )

    if state.get("error"):
        return _empty_payload(
            message=f"The query could not be executed: {state['error']}",
            sql=state.get("sql", ""),
            optimization=state.get("optimization", ""),
        )

    return _empty_payload(
        message="Query executed successfully.",
        sql=state["sql"],
        explanation=state.get("explanation", ""),
        optimization=state.get("optimization", ""),
        columns=state.get("columns", []),
        rows=state.get("rows", []),
    )
