import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

load_dotenv()


def _get_setting(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        value = None
    return str(value or os.getenv(name, default))


st.set_page_config(
    page_title="SQL Query AI Agent",
    page_icon="🗄️",
    layout="wide",
)

API_URL = _get_setting("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
# Worst case: several sequential Gemini calls at 15s each. Keep the UI bounded.
REQUEST_TIMEOUT_SECONDS = 70

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
    st.write(f"**Backend:** `{API_URL}`")
    st.write(f"**Provider:** `{os.getenv('LLM_PROVIDER', 'gemini').lower()}`")

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


def _user_error_from_http(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if isinstance(payload, dict):
        code = payload.get("error_code")
        message = payload.get("message") or payload.get("detail")
        if code and message:
            return f"{code}: {message}"
        if message:
            return str(message)
        if isinstance(payload.get("detail"), dict):
            detail = payload["detail"]
            code = detail.get("error_code")
            message = detail.get("message")
            if code and message:
                return f"{code}: {message}"
    return f"Backend returned HTTP {response.status_code}."


if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking, validating and executing..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question.strip()},
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                try:
                    result = response.json()
                except ValueError:
                    result = None

                if response.ok and isinstance(result, dict):
                    st.session_state.result = result
                    st.session_state.history.append({
                        "question": question.strip(),
                        "result": result,
                    })
                elif isinstance(result, dict):
                    st.session_state.result = result
                    if not result.get("error_code"):
                        st.error(_user_error_from_http(response))
                else:
                    st.session_state.result = None
                    st.error(_user_error_from_http(response))
            except requests.Timeout:
                st.session_state.result = None
                st.error("GEMINI_TIMEOUT: The backend did not respond in time. The Gemini provider may be slow or unavailable.")
            except requests.ConnectionError:
                st.session_state.result = None
                st.error(f"Could not connect to the backend at {API_URL}. Start FastAPI and check BACKEND_URL.")
            except requests.RequestException as exc:
                st.session_state.result = None
                st.error(f"Backend request failed: {exc}")

result = st.session_state.result

if result:
    error_code = result.get("error_code")
    if error_code:
        st.error(f"{error_code}: {result.get('message', 'The Gemini provider failed.')}")
    elif not result.get("in_scope", True):
        st.warning(result.get("message", "This request is outside the supported scope."))
    elif "could not be executed" in result.get("message", "").lower() or "validation failed" in result.get("message", "").lower():
        st.error(result.get("message"))
    else:
        st.success(result.get("message", "Done"))

    if result.get("sql"):
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
    elif result.get("in_scope", True) and result.get("sql") and not error_code:
        st.info("The query returned no rows.")
