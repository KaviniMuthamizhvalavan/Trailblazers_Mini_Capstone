"""
FastAPI application for the Corporate Learning Path Recommender.
Provides chat endpoint, path retrieval, and learner management.
"""

import os
import sys
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from langsmith import traceable

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from langchain_core.messages import HumanMessage
from agent.graph import run_agent
from agent.lms_tool_node import get_all_learners

app = FastAPI(
    title="Corporate Learning Path Recommender",
    description="AI-powered learning path recommendation using LangGraph + RAG",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# ── Session state store (in-memory) ──────────────────────────────────────────

sessions: dict[str, dict] = {}


def get_or_create_session(session_id: str, learner_id: str = "") -> dict:
    """Get existing session state or create a new one."""
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "goals": "",
            "background": "",
            "time_availability": "",
            "learner_id": learner_id,
            "constraints": [],
            "rag_context": [],
            "rag_metadata": [],
            "lms_completed": [],
            "lms_enrolled": [],
            "current_path": [],
            "is_refinement": False,
            "turn_number": 0,
            "error": None,
        }
    # Update learner_id if provided
    if learner_id:
        sessions[session_id]["learner_id"] = learner_id
    return sessions[session_id]


# ── Request/Response models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    learner_id: Optional[str] = ""


class ChatResponse(BaseModel):
    response: str
    session_id: str
    learning_path: list[dict]
    turn_number: int
    constraints: list[dict]


class PathResponse(BaseModel):
    learning_path: list[dict]
    session_id: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Corporate Learning Path Recommender API", "docs": "/docs"}


@app.post("/chat", response_model=ChatResponse)
@traceable(name="chat_endpoint", run_type="chain")
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Sends a message through the LangGraph agent
    and returns the response with the updated learning path.
    """
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())
    state = get_or_create_session(session_id, request.learner_id or "")

    # Add the user message to state
    state["messages"] = list(state["messages"]) + [HumanMessage(content=request.message)]

    try:
        # Run the agent
        result = run_agent(state)

        # Update session state with agent results
        sessions[session_id] = {
            "messages": list(result.get("messages", [])),
            "goals": result.get("goals", state.get("goals", "")),
            "background": result.get("background", state.get("background", "")),
            "time_availability": result.get("time_availability", state.get("time_availability", "")),
            "learner_id": result.get("learner_id", state.get("learner_id", "")),
            "constraints": result.get("constraints", state.get("constraints", [])),
            "rag_context": result.get("rag_context", state.get("rag_context", [])),
            "rag_metadata": result.get("rag_metadata", state.get("rag_metadata", [])),
            "lms_completed": result.get("lms_completed", state.get("lms_completed", [])),
            "lms_enrolled": result.get("lms_enrolled", state.get("lms_enrolled", [])),
            "current_path": result.get("current_path", []),
            "is_refinement": result.get("is_refinement", False),
            "turn_number": result.get("turn_number", state.get("turn_number", 0)),
            "error": result.get("error"),
        }

        # Get the AI response (last AI message)
        ai_response = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "ai":
                ai_response = msg.content
                break
            elif hasattr(msg, "content") and not isinstance(msg, HumanMessage):
                ai_response = msg.content
                break

        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            learning_path=result.get("current_path", []),
            turn_number=sessions[session_id]["turn_number"],
            constraints=sessions[session_id]["constraints"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/path/{session_id}", response_model=PathResponse)
async def get_path(session_id: str):
    """Get the current learning path for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return PathResponse(
        learning_path=sessions[session_id].get("current_path", []),
        session_id=session_id,
    )


@app.get("/learners")
async def list_learners():
    """List all available learner profiles."""
    learners = get_all_learners()
    return {"learners": learners}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
