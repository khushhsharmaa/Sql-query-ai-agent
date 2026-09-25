# Architecture

The application is intentionally split into a UI, API, orchestration layer, and database.

1. Streamlit collects the user's natural-language request.
2. FastAPI receives the request.
3. LangGraph runs a deterministic sequence of task-specific nodes.
4. Scope checking rejects unrelated requests.
5. Schema retrieval provides the current database structure to the LLM.
6. SQL generation produces a single SQLite SELECT/WITH query.
7. Validation rejects multi-statement or mutating SQL and uses SQLite EXPLAIN QUERY PLAN to catch schema/syntax errors.
8. Optimization produces practical performance suggestions.
9. The query executes against SQLite.
10. The explanation node turns the SQL into plain English.
11. Streamlit renders SQL, explanation, optimization suggestions, and tabular results.
