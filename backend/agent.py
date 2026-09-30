import os
import re
from typing import TypedDict, Any
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .db import get_schema_text, explain_query, execute_select
from .prompts import SCOPE_PROMPT, SQL_PROMPT, OPTIMIZE_PROMPT, EXPLAIN_PROMPT


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


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it to the .env file before using the AI agent."
        )

    return ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=0,
    google_api_key=api_key,
)

def scope_node(state: AgentState):
    llm = get_llm().with_structured_output(ScopeResult)
    result = llm.invoke(SCOPE_PROMPT.format(question=state["question"]))
    return {
        "in_scope": result.in_scope,
        "scope_reason": result.reason,
    }


def schema_node(state: AgentState):
    return {"schema": get_schema_text()}


def generate_sql_node(state: AgentState):
    llm = get_llm().with_structured_output(SQLResult)
    result = llm.invoke(SQL_PROMPT.format(
        schema=state["schema"],
        question=state["question"],
    ))
    return {"sql": result.sql.strip(), "ambiguity": result.ambiguity.strip()}


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
    llm = get_llm()
    result = llm.invoke(OPTIMIZE_PROMPT.format(
        sql=state["sql"], schema=state["schema"]
    ))
    return {"optimization": result.text.strip()}


def execute_node(state: AgentState):
    try:
        columns, rows = execute_select(state["sql"])
        return {"columns": columns, "rows": rows, "error": ""}
    except Exception as exc:
        return {"error": str(exc)}


def explain_node(state: AgentState):
    llm = get_llm()
    result = llm.invoke(EXPLAIN_PROMPT.format(sql=state["sql"]))
    return {"explanation": result.text.strip()}


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
        return "schema" if state.get("in_scope") else END

    graph.add_conditional_edges("scope", after_scope, {"schema": "schema", END: END})
    graph.add_edge("schema", "generate_sql")

    def after_validation(state):
        return "optimize" if not state.get("validation_error") else END

    graph.add_edge("generate_sql", "validate")
    graph.add_conditional_edges(
        "validate",
        after_validation,
        {"optimize": "optimize", END: END},
    )
    graph.add_edge("optimize", "execute")
    graph.add_edge("execute", "explain")
    graph.add_edge("explain", END)
    return graph.compile()


GRAPH = build_graph()


def run_agent(question: str):
    state = GRAPH.invoke({"question": question})
    if not state.get("in_scope"):
        return {
            "in_scope": False,
            "message": "I'm designed to assist only with SQL and database-related tasks.",
            "sql": "",
            "explanation": state.get("scope_reason", ""),
            "optimization": "",
            "columns": [],
            "rows": [],
        }

    if state.get("validation_error"):
        return {
            "in_scope": True,
            "message": state["validation_error"],
            "sql": state.get("sql", ""),
            "explanation": "",
            "optimization": "",
            "columns": [],
            "rows": [],
        }

    if state.get("error"):
        return {
            "in_scope": True,
            "message": f"The query could not be executed: {state['error']}",
            "sql": state.get("sql", ""),
            "explanation": "",
            "optimization": state.get("optimization", ""),
            "columns": [],
            "rows": [],
        }

    return {
        "in_scope": True,
        "message": "Query executed successfully.",
        "sql": state["sql"],
        "explanation": state.get("explanation", ""),
        "optimization": state.get("optimization", ""),
        "columns": state.get("columns", []),
        "rows": state.get("rows", []),
    }
