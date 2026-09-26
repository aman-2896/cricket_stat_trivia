"""
FastAPI entrypoint for the Cricket Assistant.

Exposes a single /chat endpoint that forwards the user's question to the
LangGraph agent (services.agent.run_agent) and returns its answer plus the
list of tools the agent decided to call.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.agent import run_agent

# Simple complexity (creating a FastAPI app instance) — this just builds the
# app object that uvicorn/fastapi[standard] will serve; no request handling
# happens here yet.
app = FastAPI(title="Cricket Assistant")


class ChatRequest(BaseModel):
    # Pydantic model describing the expected JSON body of POST /chat.
    question: str


class ChatResponse(BaseModel):
    # Pydantic model describing the JSON response of POST /chat.
    answer: str
    tools_used: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Answer a cricket question.

    Delegates all the actual reasoning (tool selection, SQL generation,
    rules retrieval, etc.) to run_agent(); this endpoint is just a thin
    HTTP wrapper around it.
    """
    try:
        result = run_agent(req.question)
        return ChatResponse(answer=result["answer"], tools_used=result["called_tools"])
    except Exception as e:
        # Catch-all so any failure inside the agent (LLM error, DB error,
        # etc.) surfaces as a 500 instead of crashing the server process.
        raise HTTPException(status_code=500, detail=f"Something went wrong processing your request \n {e}")
