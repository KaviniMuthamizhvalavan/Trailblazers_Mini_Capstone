"""
Intake Node: Extracts/updates learner goals, background, time availability,
and constraints from the user message. Appends new constraints to the
running list (never replaces). Detects contradictions and marks superseded
constraints.
"""

import os
import json
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langsmith import traceable
from agent.llm import get_llm

load_dotenv()


INTAKE_SYSTEM_PROMPT = """You are an intake analyzer for a corporate learning path recommender.
Your job is to extract structured information from the user's message.

You MUST respond with valid JSON only, no markdown, no explanation.

IMPORTANT RULES:
- "goals", "background", "time_availability" are SEPARATE fields from "constraints".
- A contradiction to one constraint must NEVER affect goals, background, time_availability, or other unrelated constraints.
- Only mark a prior constraint as contradicted if the user EXPLICITLY revises that specific claim.
- If a field was not mentioned in this message, return null for it (the system will keep the prior value).

Current state from prior turns (for context — do NOT clear these unless the user explicitly changes them):
- Current goals: {current_goals}
- Current background: {current_background}
- Current time_availability: {current_time_availability}

Extract the following from THIS message only:
1. "goals": The learner's career/learning goals (string). Return null if not mentioned in THIS message.
2. "background": Their current skill level, prior experience, existing knowledge (string). Return null if not mentioned in THIS message.
3. "time_availability": How much time they can dedicate (string, e.g., "5 hours per week"). Return null if not mentioned in THIS message.
4. "new_constraints": A list of specific constraints or preferences stated in THIS message (list of strings). Examples: "no prior SAP experience", "prefer hands-on courses". Do NOT include goals or time_availability here — those go in their own fields. Return an empty list if none.
5. "contradictions": A list of constraints from the prior constraints list below that this message EXPLICITLY contradicts or revises (list of strings matching the exact prior constraint text). Only include a prior constraint here if this message directly reverses that specific claim. Do NOT include unrelated constraints. Return an empty list if no contradictions.
6. "learner_id": If the user identifies themselves by learner ID (e.g., "L001"), extract it (string). Otherwise null.

Prior constraints from earlier turns (only mark these as contradicted if THIS message explicitly reverses one):
{prior_constraints}

Respond ONLY with the JSON object."""


@traceable(name="intake_node", run_type="chain")
def intake_node(state: dict) -> dict:
    """
    Extract goals, background, time, and constraints from the latest user message.
    Accumulates constraints across turns, marks superseded ones.
    """
    llm = get_llm()

    # Get the latest user message
    messages = state.get("messages", [])
    latest_message = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            latest_message = msg.content
            break
        elif isinstance(msg, HumanMessage):
            latest_message = msg.content
            break

    if not latest_message:
        return state

    # Gather prior constraints for contradiction detection
    prior_constraints = state.get("constraints", [])
    active_constraints = [c["constraint"] for c in prior_constraints if not c.get("superseded", False)]
    prior_constraints_text = json.dumps(active_constraints) if active_constraints else "None yet."

    # Provide current state context so the LLM doesn't accidentally clear fields
    current_goals = state.get("goals", "") or "Not yet specified"
    current_background = state.get("background", "") or "Not yet specified"
    current_time = state.get("time_availability", "") or "Not yet specified"

    # Build the prompt
    system_prompt = INTAKE_SYSTEM_PROMPT.format(
        prior_constraints=prior_constraints_text,
        current_goals=current_goals,
        current_background=current_background,
        current_time_availability=current_time,
    )

    import time
    for attempt in range(5):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=latest_message),
            ])
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < 4:
                    print(f"Rate limit hit in intake_node, sleeping for 31 seconds... (Attempt {attempt+1})")
                    time.sleep(31)
                else:
                    raise e
            else:
                raise e

    # Parse the response
    content = response.content.strip()
    if "</think>" in content:
        content = content.split("</think>")[1].strip()
        
    try:
        extracted = json.loads(content)
    except json.JSONDecodeError:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        try:
            extracted = json.loads(content)
        except json.JSONDecodeError:
            extracted = {}

    # Determine turn number
    turn_number = state.get("turn_number", 0) + 1

    # Update goals, background, time_availability (only if newly provided)
    goals = extracted.get("goals") or state.get("goals", "")
    background = extracted.get("background") or state.get("background", "")
    time_availability = extracted.get("time_availability") or state.get("time_availability", "")
    learner_id = extracted.get("learner_id") or state.get("learner_id", "")

    # Handle contradictions — mark prior constraints as superseded
    constraints = list(state.get("constraints", []))
    contradictions = extracted.get("contradictions", [])
    if contradictions:
        for c in constraints:
            if c["constraint"] in contradictions:
                c["superseded"] = True

    # Append new constraints
    new_constraints = extracted.get("new_constraints", [])
    for nc in new_constraints:
        constraints.append({
            "turn": turn_number,
            "constraint": nc,
            "superseded": False,
        })

    # Determine if this is a refinement turn
    is_refinement = turn_number > 1 and bool(state.get("current_path"))

    return {
        "goals": goals,
        "background": background,
        "time_availability": time_availability,
        "learner_id": learner_id,
        "constraints": constraints,
        "is_refinement": is_refinement,
        "turn_number": turn_number,
    }
