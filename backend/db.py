from pathlib import Path
import sqlite3

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "company.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS departments (
    department_id INTEGER PRIMARY KEY,
    department_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    job_title TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);
"""

SEED_SQL = """
INSERT OR IGNORE INTO departments(department_id, department_name) VALUES
(1, 'Engineering'), (2, 'Sales'), (3, 'HR'), (4, 'Finance');

INSERT OR IGNORE INTO employees
(employee_id, name, department_id, job_title, salary, hire_date) VALUES
(101, 'Aarav Mehta', 1, 'Software Engineer', 85000, '2023-05-15'),
(102, 'Diya Shah', 1, 'Senior Software Engineer', 115000, '2024-02-10'),
(103, 'Kabir Singh', 2, 'Sales Executive', 65000, '2024-01-22'),
(104, 'Meera Iyer', 3, 'HR Executive', 60000, '2022-11-08'),
(105, 'Rohan Gupta', 4, 'Financial Analyst', 78000, '2024-06-17'),
(106, 'Ananya Rao', 1, 'Data Engineer', 92000, '2024-08-03'),
(107, 'Vivaan Kapoor', 2, 'Account Manager', 74000, '2023-09-19'),
(108, 'Sara Khan', 3, 'Recruiter', 62000, '2024-03-11');
"""

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SEED_SQL)

def get_schema_text():
    with get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        chunks = []
        for row in tables:
            table = row["name"]
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_text = ", ".join(
                f"{c['name']} ({c['type']})" for c in cols
            )
            chunks.append(f"TABLE {table}: {col_text}")
        return "\n".join(chunks)

def explain_query(sql: str):
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()]

def execute_select(sql: str):
    with get_connection() as conn:
        cursor = conn.execute(sql)
        rows = [dict(r) for r in cursor.fetchmany(200)]
        columns = [d[0] for d in cursor.description] if cursor.description else []
        return columns, rows
