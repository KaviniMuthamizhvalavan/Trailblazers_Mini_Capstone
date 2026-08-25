"""
Shared state schema for the LangGraph agent.
Carries forward all context across nodes and across conversation turns.
"""

from typing import TypedDict, Annotated, Sequence, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class LearningPathItem(TypedDict):
    """A single course in the recommended learning path."""
    course_id: str
    title: str
    track: str
    estimated_hours: float
    difficulty_level: str
    week_start: int
    week_end: int
    prerequisite_ids: list[str]
    reason: str


class AgentState(TypedDict):
    """
    Shared state for the Learning Path Recommender agent.

    CRITICAL: This state persists across turns. The constraints list and
    rag_context/lms_results are accumulated, not replaced, so that
    refinement turns can build on prior context without re-fetching.
    """
    # Conversation messages (auto-accumulated via add_messages)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Extracted learner profile info (updated each turn)
    goals: str
    background: str
    time_availability: str  # e.g., "5 hours per week"
    learner_id: str

    # Running list of ALL constraints across all turns.
    # Each constraint is a dict: {"turn": int, "constraint": str, "superseded": bool}
    constraints: list[dict]

    # RAG retrieval results (persisted across turns)
    rag_context: list[str]  # Retrieved document chunks
    rag_metadata: list[dict]  # Metadata for retrieved chunks

    # LMS tool results (persisted across turns)
    lms_completed: list[str]  # Completed course IDs
    lms_enrolled: list[str]  # Currently enrolled course IDs

    # Current recommended path
    current_path: list[dict]  # List of LearningPathItem dicts

    # Control flags
    is_refinement: bool  # True if this is a follow-up turn
    turn_number: int
    error: Optional[str]
