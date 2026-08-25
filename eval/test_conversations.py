"""
Scripted multi-turn test conversations for Context-Aware Refinement (35%).
Tests the 4-turn sequence from planner:
  Turn 1: state goal
  Turn 2: add constraint
  Turn 3: add second unrelated constraint (must prove Turn 2 constraint survives)
  Turn 4: contradiction test (must prove only contradicted constraint is superseded)

Captures real transcripts with goals/background/time_availability per turn.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agent.graph import run_agent


def run_conversation(conversation_name: str, turns: list[dict], learner_id: str = ""):
    """
    Run a scripted multi-turn conversation through the agent.
    Returns the full transcript.
    """
    print(f"\n{'='*70}")
    print(f"CONVERSATION: {conversation_name}")
    print(f"{'='*70}")

    state = {
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

    transcript = []

    for i, turn in enumerate(turns):
        print(f"\n--- Turn {i+1}: {turn['description']} ---")
        print(f"USER: {turn['message']}")

        # Add user message
        state["messages"] = list(state.get("messages", [])) + [HumanMessage(content=turn["message"])]

        # Run agent
        try:
            result = run_agent(state)

            # Extract AI response
            ai_response = ""
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "type") and msg.type == "ai":
                    ai_response = msg.content
                    break

            print(f"AI: {ai_response[:300]}...")

            # Record path
            current_path = result.get("current_path", [])
            path_ids = [c.get("course_id", "?") for c in current_path]
            print(f"PATH ({len(current_path)} courses): {path_ids}")

            # Record state fields
            goals = result.get("goals", "")
            background = result.get("background", "")
            time_availability = result.get("time_availability", "")
            print(f"GOALS: {goals}")
            print(f"BACKGROUND: {background}")
            print(f"TIME_AVAILABILITY: {time_availability}")

            # Record constraints
            constraints = result.get("constraints", [])
            active = [c for c in constraints if not c.get("superseded", False)]
            superseded = [c for c in constraints if c.get("superseded", False)]
            print(f"ACTIVE CONSTRAINTS: {[c['constraint'] for c in active]}")
            if superseded:
                print(f"SUPERSEDED: {[c['constraint'] for c in superseded]}")

            # Validation checks
            validations = turn.get("validate", {})
            validation_results = {}

            if "path_should_contain" in validations:
                for cid in validations["path_should_contain"]:
                    found = cid in path_ids
                    validation_results[f"path contains {cid}"] = found
                    status = "[PASS]" if found else "[FAIL]"
                    print(f"  {status} Path should contain {cid}: {found}")

            if "path_should_not_contain" in validations:
                for cid in validations["path_should_not_contain"]:
                    not_found = cid not in path_ids
                    validation_results[f"path excludes {cid}"] = not_found
                    status = "[PASS]" if not_found else "[FAIL]"
                    print(f"  {status} Path should NOT contain {cid}: {not_found}")

            if "constraint_should_be_active" in validations:
                for constraint_text in validations["constraint_should_be_active"]:
                    # Fuzzy match: check if any active constraint contains the key phrase
                    is_active = any(
                        constraint_text.lower() in c["constraint"].lower()
                        for c in active
                    )
                    validation_results[f"active: {constraint_text}"] = is_active
                    status = "[PASS]" if is_active else "[FAIL]"
                    print(f"  {status} Constraint active (contains '{constraint_text}'): {is_active}")

            if "constraint_should_be_superseded" in validations:
                for constraint_text in validations["constraint_should_be_superseded"]:
                    # Fuzzy match: check if any superseded constraint contains the key phrase
                    is_superseded = any(
                        constraint_text.lower() in c["constraint"].lower()
                        for c in superseded
                    )
                    validation_results[f"superseded: {constraint_text}"] = is_superseded
                    status = "[PASS]" if is_superseded else "[FAIL]"
                    print(f"  {status} Constraint superseded (contains '{constraint_text}'): {is_superseded}")

            if "time_availability_contains" in validations:
                check_str = validations["time_availability_contains"]
                has_time = check_str.lower() in time_availability.lower() if time_availability else False
                validation_results[f"time_availability contains '{check_str}'"] = has_time
                status = "[PASS]" if has_time else "[FAIL]"
                print(f"  {status} time_availability contains '{check_str}': {has_time} (actual: '{time_availability}')")

            if "goals_contains" in validations:
                check_str = validations["goals_contains"]
                has_goal = check_str.lower() in goals.lower() if goals else False
                validation_results[f"goals contains '{check_str}'"] = has_goal
                status = "[PASS]" if has_goal else "[FAIL]"
                print(f"  {status} goals contains '{check_str}': {has_goal} (actual: '{goals}')")

            transcript.append({
                "turn": i + 1,
                "description": turn["description"],
                "user_message": turn["message"],
                "ai_response": ai_response,
                "path": current_path,
                "path_ids": path_ids,
                "goals": goals,
                "background": background,
                "time_availability": time_availability,
                "active_constraints": [c["constraint"] for c in active],
                "superseded_constraints": [c["constraint"] for c in superseded],
                "all_constraints": constraints,
                "validation_results": validation_results,
                "is_refinement": result.get("is_refinement", False),
            })

            # Update state for next turn
            state = {
                "messages": list(result.get("messages", [])),
                "goals": result.get("goals", ""),
                "background": result.get("background", ""),
                "time_availability": result.get("time_availability", ""),
                "learner_id": result.get("learner_id", learner_id),
                "constraints": result.get("constraints", []),
                "rag_context": result.get("rag_context", []),
                "rag_metadata": result.get("rag_metadata", []),
                "lms_completed": result.get("lms_completed", []),
                "lms_enrolled": result.get("lms_enrolled", []),
                "current_path": result.get("current_path", []),
                "is_refinement": result.get("is_refinement", False),
                "turn_number": result.get("turn_number", 0),
                "error": None,
            }

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            transcript.append({
                "turn": i + 1,
                "description": turn["description"],
                "user_message": turn["message"],
                "error": str(e),
            })

        # Small delay to avoid rate limits
        time.sleep(1)

    return transcript


# ── Test Conversation 1: Salesforce Admin (4-Turn Refinement Test) ────
SALESFORCE_CONVERSATION = {
    "name": "Salesforce Administrator - 4-Turn Refinement Test",
    "learner_id": "L003",  # Maria Santos, complete beginner
    "turns": [
        {
            "description": "State goal",
            "message": "I want to become a Salesforce administrator.",
            "validate": {
                "goals_contains": "salesforce",
            }
        },
        {
            "description": "Add constraint: no prior experience",
            "message": "I have no prior Salesforce experience at all.",
            "validate": {
                "path_should_contain": ["SF-101"],
                "constraint_should_be_active": ["no prior"],
            }
        },
        {
            "description": "Add second unrelated constraint: time limit",
            "message": "I only have 5 hours a week to study.",
            "validate": {
                "path_should_contain": ["SF-101"],
                # KEY TEST: Turn 2 constraint must still be active
                "constraint_should_be_active": ["no prior"],
                # KEY TEST: time_availability must reflect "5 hours"
                "time_availability_contains": "5",
            }
        },
        {
            "description": "Contradiction: revise experience claim",
            "message": "Actually, I do have some Salesforce basics. I've used it at a previous job for basic data entry and reporting.",
            "validate": {
                # KEY TEST: the experience constraint should now be superseded
                "constraint_should_be_superseded": ["no prior"],
                # KEY TEST: time_availability must survive the contradiction
                "time_availability_contains": "5",
            }
        },
    ]
}

# ── Test Conversation 2: SAP with LMS history ────
SAP_CONVERSATION = {
    "name": "SAP Consultant Path - With LMS History",
    "learner_id": "L001",  # Priya Sharma, has SAP-101/102/103 done, enrolled in SAP-104
    "turns": [
        {
            "description": "State goal with existing history",
            "message": "I'm learner L001. I want to become a certified SAP consultant. I already have some SAP training completed.",
            "validate": {
                "path_should_not_contain": ["SAP-101", "SAP-102", "SAP-103"],
            }
        },
        {
            "description": "Add time constraint",
            "message": "I can dedicate 8 hours per week to studying.",
            "validate": {
                "path_should_not_contain": ["SAP-101", "SAP-102", "SAP-103"],
                "time_availability_contains": "8",
            }
        },
    ]
}

# ── Test Conversation 3: Cloud with course change ────
CLOUD_CONVERSATION = {
    "name": "Cloud Architecture - Multi-Turn with Redirection",
    "learner_id": "",
    "turns": [
        {
            "description": "Initial AWS goal",
            "message": "I want to learn cloud computing and get AWS certified. I'm a complete beginner with no cloud experience.",
            "validate": {
                "path_should_contain": ["CLD-101"],
            }
        },
        {
            "description": "Add preference for Azure",
            "message": "Actually, my company uses Azure, so I'd prefer to focus on Azure track instead of AWS.",
            "validate": {
                "path_should_contain": ["CLD-105"],
            }
        },
    ]
}


def main():
    """Run all test conversations and save results."""
    all_transcripts = {}

    # Run conversations
    for conv in [SALESFORCE_CONVERSATION, SAP_CONVERSATION, CLOUD_CONVERSATION]:
        transcript = run_conversation(
            conv["name"],
            conv["turns"],
            conv.get("learner_id", ""),
        )
        all_transcripts[conv["name"]] = transcript

    # Save all transcripts
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "conversation_transcripts.json")
    with open(output_path, "w") as f:
        json.dump(all_transcripts, f, indent=2, default=str)

    print(f"\n\n{'='*70}")
    print(f"All conversation transcripts saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
