import logging
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .db import init_db
from .agent import run_agent
from .errors import (
    HTTP_STATUS_BY_CODE,
    INTERNAL_ERROR,
    LLM_CONFIGURATION_ERROR,
    LLMProviderError,
    USER_MESSAGES,
    redact,
)

load_dotenv()
init_db()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SQL Query AI Agent",
    description="Task-oriented AI agent that converts natural language into safe, read-only SQL.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty or whitespace.")

    try:
        result = run_agent(question)
    except (RuntimeError, LLMProviderError) as exc:
        redacted_msg = redact(str(exc))
        logger.error("Configuration error: %s", redacted_msg)
        error_code = getattr(exc, "code", LLM_CONFIGURATION_ERROR)
        status = HTTP_STATUS_BY_CODE.get(error_code, 500)
        return JSONResponse(
            status_code=status,
            content={
                "in_scope": True,
                "message": redacted_msg,
                "sql": "",
                "explanation": "",
                "optimization": "",
                "columns": [],
                "rows": [],
                "error_code": error_code,
            },
        )
    except Exception:
        logger.error("Unexpected /query failure:\n%s", redact(traceback.format_exc()))
        return JSONResponse(
            status_code=500,
            content={
                "in_scope": True,
                "message": USER_MESSAGES[INTERNAL_ERROR],
                "sql": "",
                "explanation": "",
                "optimization": "",
                "columns": [],
                "rows": [],
                "error_code": INTERNAL_ERROR,
            },
        )

    error_code = result.get("error_code")
    if error_code:
        status = HTTP_STATUS_BY_CODE.get(error_code, 500)
        logger.error("Query failed with %s (HTTP %s)", error_code, status)
        return JSONResponse(status_code=status, content=result)

    return result
