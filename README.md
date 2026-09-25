# SQL Query AI Agent

A task-oriented AI agent that converts natural-language database questions into safe, read-only SQLite queries.

## Features

- LangGraph workflow: scope → schema → SQL generation → validation → optimization → execution → explanation
- Read-only SQL guardrails
- SQLite schema retrieval
- SQL execution and results table
- Query history
- CSV download
- FastAPI backend
- Streamlit frontend
- Basic automated tests

## Setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure the API key

Copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=your_key_here
```

Never commit `.env`.

### 4. Start the backend

From the project root:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

### 5. Start the frontend

Open a second terminal:

```powershell
streamlit run frontend/app.py
```

Open the Streamlit URL shown in the terminal, normally `http://localhost:8501`.

## Example prompts

- Show all employees hired after January 2024.
- What is the average salary by department?
- List software engineers and their salaries.
- Which department has the highest average salary?

Out-of-scope example:

- Who won the FIFA World Cup?

Safety example:

- DELETE all employees.

The application should reject non-read-only SQL.

## Architecture

```text
User
  ↓
Streamlit UI
  ↓
FastAPI /query
  ↓
LangGraph
  ├─ Scope check
  ├─ Schema retrieval
  ├─ SQL generation
  ├─ SQL validation
  ├─ Optimization
  ├─ Execution
  └─ Explanation
  ↓
SQLite
```

## Attribution

If this project is submitted for a hiring assignment, follow the assignment's rules about external/open-source code. Do not represent another developer's code as original work. This implementation is intended as a clean reference implementation and should be customized, tested, and documented according to the employer's requirements.
