from langchain_core.tools import tool
from db.models import get_session, Learner

@tool
def fetch_learner_history(learner_id: str) -> dict:
    """
    Look up a learner's completed and currently-enrolled course IDs from the
    LMS database. Call this when you need to know what courses a specific
    learner has already finished or is currently taking, so they aren't
    recommended again.
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
