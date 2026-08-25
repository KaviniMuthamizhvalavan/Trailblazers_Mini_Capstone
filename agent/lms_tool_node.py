"""
LMS Tool Node: Queries the SQLite database for a learner's completed
and enrolled courses. Results are used by the composition node to filter
out courses that shouldn't be re-recommended.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import get_session, Learner
from langsmith import traceable


@traceable(name="fetch_learner_history", run_type="tool")
def fetch_learner_history(learner_id: str) -> dict:
    """
    Fetch completed and enrolled course IDs for a learner from the database.
    Returns {"completed": [...], "enrolled": [...], "learner_name": str}
    """
    session = get_session()
    try:
        learner = session.query(Learner).filter_by(learner_id=learner_id).first()
        if not learner:
            return {
                "completed": [],
                "enrolled": [],
                "learner_name": "Unknown",
                "error": f"Learner '{learner_id}' not found in LMS database.",
            }

        return {
            "completed": [c.id for c in learner.completed_courses],
            "enrolled": [c.id for c in learner.enrolled_courses],
            "learner_name": learner.name,
        }
    finally:
        session.close()


def get_all_learners() -> list[dict]:
    """Return all learner profiles for the frontend selector."""
    session = get_session()
    try:
        learners = session.query(Learner).all()
        return [learner.to_dict() for learner in learners]
    finally:
        session.close()


@traceable(name="lms_tool_node", run_type="tool")
def lms_tool_node(state: dict) -> dict:
    """
    LMS tool node. Fetches the learner's course history from the database.
    Stores completed and enrolled course IDs in state for the composition node.
    """
    learner_id = state.get("learner_id", "")

    if not learner_id:
        # No learner identified — return empty results
        return {
            "lms_completed": [],
            "lms_enrolled": [],
        }

    history = fetch_learner_history(learner_id)

    result = {
        "lms_completed": history.get("completed", []),
        "lms_enrolled": history.get("enrolled", []),
    }

    if "error" in history:
        result["error"] = history["error"]

    return result
