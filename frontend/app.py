import io
import json
import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="SQL Query AI Agent",
    page_icon="🗄️",
    layout="wide",
)

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None

st.title("🗄️ SQL Query AI Agent")
st.caption("Ask questions about the sample employee database. The agent generates and executes read-only SQL.")

with st.sidebar:
    st.header("History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            if st.button(item["question"], key=f"history_{i}", use_container_width=True):
                st.session_state.result = item["result"]
                st.rerun()
    else:
        st.info("Your recent queries will appear here.")

    st.divider()
    st.write("**Scope:** SQL and database questions only.")
    st.write("**Safety:** SELECT/WITH queries only.")

question = st.text_area(
    "Ask your database question",
    placeholder="Example: Show all employees hired after January 2024.",
    height=100,
)

col1, col2 = st.columns([1, 5])
with col1:
    ask = st.button("Run query", type="primary", use_container_width=True)
with col2:
    if st.button("Clear"):
        st.session_state.result = None
        st.rerun()

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking, validating and executing..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question.strip()},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
                st.session_state.result = result
                st.session_state.history.append({
                    "question": question.strip(),
                    "result": result,
                })
            except requests.RequestException as exc:
                st.error(f"Could not connect to the backend: {exc}")

result = st.session_state.result

if result:
    if not result.get("in_scope", True):
        st.warning(result.get("message", "This request is outside the supported scope."))
    else:
        st.success(result.get("message", "Done"))

        st.subheader("Generated SQL")
        st.code(result.get("sql", ""), language="sql")

        if result.get("explanation"):
            st.subheader("Explanation")
            st.write(result["explanation"])

        if result.get("optimization"):
            st.subheader("Optimization suggestions")
            st.write(result["optimization"])

        rows = result.get("rows", [])
        columns = result.get("columns", [])
        if rows:
            st.subheader("Results")
            df = pd.DataFrame(rows, columns=columns or None)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv",
            )
        else:
            st.info("The query returned no rows.")
