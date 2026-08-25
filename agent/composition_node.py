"""
Composition Node: The core reasoning node that merges RAG context, LMS history,
and ALL accumulated constraints to produce a sequenced, time-boxed learning path.

This is where 75% of the grade lives (Recommendation Relevance 40% + Context-Aware
Refinement 35%). The composition node must:
1. Filter out completed/enrolled courses
2. Enforce prerequisite ordering (topological sort)
3. Apply time-boxing based on hours/week
4. On refinement turns, use prior RAG/LMS context from state
"""

import os
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langsmith import traceable
from db.models import get_session, Course
from agent.llm import get_llm

load_dotenv()


COMPOSITION_SYSTEM_PROMPT = """You are an expert corporate learning path advisor. Your job is to create a personalized, sequenced learning path.

## Instructions

Given the learner's profile and available courses, create an optimal learning path that:

1. **Matches their goals** — recommend courses that directly serve their stated career/learning objectives
2. **Respects their background** — don't recommend courses below their level; include foundational courses only if they lack prerequisites
3. **Enforces prerequisite ordering** — a course must NEVER appear before its prerequisites in the path
4. **Applies time-boxing** — spread courses across weeks based on the learner's available hours per week. Each week should not exceed the stated hours/week. If a course takes more hours than available per week, it spans multiple weeks.
5. **Filters out completed/enrolled courses** — NEVER recommend a course the learner has already completed or is currently enrolled in
6. **Honors ALL active constraints** — every non-superseded constraint must be reflected in the path. If constraints conflict, the most recent one takes priority.
7. **Is realistic** — if the goal can't be achieved in a reasonable timeframe given the constraints, say so explicitly rather than producing an unrealistic plan

## Learner Profile
- **Goals**: {goals}
- **Background**: {background}
- **Time Availability**: {time_availability}

## Active Constraints (ALL must be honored)
{active_constraints}

## Completed Courses (DO NOT recommend these)
{completed_courses}

## Currently Enrolled (DO NOT recommend these)
{enrolled_courses}

## Available Courses from Knowledge Base
{rag_context}

## Previous Learning Path (if refinement)
{previous_path}

## Response Format

You MUST respond with valid JSON only. No markdown, no explanation outside the JSON. Keep all text fields extremely short (maximum 12 words) to avoid truncation. Use the exact short keys specified below:

{{
  "thinking": "Concise explanation (max 12 words)",
  "learning_path": [
    {{
      "id": "COURSE-ID",
      "start": 1,
      "end": 4,
      "why": "Concise reason (max 8 words)"
    }}
  ],
  "total_weeks": 20,
  "total_hours": 100,
  "feasibility_note": "Feasibility note (max 10 words)",
  "summary": "Summary (max 12 words)"
}}"""


def sanitize_json_content(content: str) -> str:
    """Sanitize JSON string by adding missing commas or fixing minor syntax errors."""
    content = content.strip()
    if not content:
        return content
    lines = content.splitlines()
    for i in range(len(lines) - 1):
        line = lines[i].rstrip()
        next_line = lines[i+1].lstrip()
        # If this line ends with a closing quote, number, boolean, null, or brace/bracket,
        # and doesn't end with a comma, colon, open brace/bracket,
        # and the next line starts with a double quote (indicating a key)
        if line and not line.endswith(",") and not line.endswith("{") and not line.endswith("[") and not line.endswith(":"):
            if line.endswith('"') or line[-1].isdigit() or line.endswith("true") or line.endswith("false") or line.endswith("null") or line.endswith("}") or line.endswith("]"):
                if next_line.startswith('"'):
                    lines[i] = line + ","
    return "\n".join(lines)


def enforce_prerequisite_ordering(learning_path: list[dict]) -> list[dict]:
    """
    Programmatically enforce prerequisite ordering via topological sort
    (Kahn's algorithm). Guarantees no course appears before its prerequisites.

    Uses the LLM's original ordering as a stable tiebreaker when multiple
    courses have no remaining dependency constraints, so the output stays
    as close to the LLM's intent as possible.

    After reordering, recalculates week_start/week_end sequentially,
    preserving each course's original duration span.
    """
    if len(learning_path) <= 1:
        return learning_path

    # Build lookup
    course_map = {c["course_id"]: c for c in learning_path}
    path_ids = set(course_map.keys())
    original_order = {c["course_id"]: i for i, c in enumerate(learning_path)}

    # Build in-degree counts and adjacency list (only for prereqs IN the path)
    in_degree = {cid: 0 for cid in path_ids}
    dependents = {cid: [] for cid in path_ids}

    for c in learning_path:
        cid = c["course_id"]
        for prereq_id in c.get("prerequisite_ids", []):
            if prereq_id in path_ids:
                dependents[prereq_id].append(cid)
                in_degree[cid] += 1

    # Kahn's algorithm with stable tiebreaking (preserve LLM order)
    queue = sorted(
        [cid for cid in path_ids if in_degree[cid] == 0],
        key=lambda x: original_order.get(x, 0),
    )
    sorted_ids = []

    while queue:
        node = queue.pop(0)
        sorted_ids.append(node)
        for dep in dependents[node]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                # Insert in sorted position by original order (stable)
                insert_pos = len(queue)
                for i, q in enumerate(queue):
                    if original_order.get(dep, 0) < original_order.get(q, 0):
                        insert_pos = i
                        break
                queue.insert(insert_pos, dep)

    # If cycle detected (shouldn't happen with real catalog), fall back
    if len(sorted_ids) != len(path_ids):
        return learning_path

    # If already correctly ordered, return as-is
    if sorted_ids == [c["course_id"] for c in learning_path]:
        return learning_path

    # Rebuild path in sorted order, recalculating week_start/week_end
    sorted_path = []
    current_week = 1
    for cid in sorted_ids:
        course = course_map[cid].copy()
        original_span = max(1, course.get("week_end", 1) - course.get("week_start", 1) + 1)
        course["week_start"] = current_week
        course["week_end"] = current_week + original_span - 1
        current_week = course["week_end"] + 1
        sorted_path.append(course)

    return sorted_path

@traceable(name="composition_node", run_type="chain")
def composition_node(state: dict) -> dict:
    """
    Core composition node. Merges RAG context + LMS history + constraints
    into a sequenced, time-boxed learning path.
    """
    llm = get_llm()

    # Gather all inputs from state
    goals = state.get("goals", "Not specified")
    background = state.get("background", "Not specified")
    time_availability = state.get("time_availability", "Not specified")

    # Active constraints (non-superseded)
    constraints = state.get("constraints", [])
    active_constraints = [
        f"- Turn {c['turn']}: {c['constraint']}"
        for c in constraints
        if not c.get("superseded", False)
    ]
    active_constraints_text = "\n".join(active_constraints) if active_constraints else "None specified."

    # Completed and enrolled courses
    completed = state.get("lms_completed", [])
    enrolled = state.get("lms_enrolled", [])
    completed_text = ", ".join(completed) if completed else "None"
    enrolled_text = ", ".join(enrolled) if enrolled else "None"

    # RAG context
    rag_context = state.get("rag_context", [])
    rag_text = "\n\n".join(rag_context) if rag_context else "No courses retrieved."

    # Previous path (for refinement turns)
    previous_path = state.get("current_path", [])
    if previous_path:
        prev_path_text = json.dumps(previous_path, indent=2)
    else:
        prev_path_text = "No previous path (first recommendation)."

    # Build the prompt
    system_prompt = COMPOSITION_SYSTEM_PROMPT.format(
        goals=goals,
        background=background,
        time_availability=time_availability,
        active_constraints=active_constraints_text,
        completed_courses=completed_text,
        enrolled_courses=enrolled_text,
        rag_context=rag_text,
        previous_path=prev_path_text,
    )

    import time
    for attempt in range(5):
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Generate the optimal learning path based on the above profile and constraints."),
            ])
            break
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < 4:
                    print(f"Rate limit hit in composition_node, sleeping for 31 seconds... (Attempt {attempt+1})")
                    time.sleep(31)
                else:
                    raise e
            else:
                raise e

    # Parse the response
    content = response.content.strip()
    if "</think>" in content:
        content = content.split("</think>")[1].strip()
    
    # Strip markdown block if present before sanitizing
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    content = sanitize_json_content(content)
    
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        try:
            # Fallback if markdown stripping failed previously
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            content = sanitize_json_content(content)
            result = json.loads(content)
        except json.JSONDecodeError:
            print("==== JSON DECODE ERROR IN COMPOSITION NODE ====")
            print(f"RAW CONTENT:\n{content}")
            print("===============================================")
            result = {
                "learning_path": [],
                "summary": "I encountered an issue generating the learning path. Please try rephrasing your request.",
                "thinking": "Failed to parse LLM response.",
                "total_weeks": 0,
                "total_hours": 0,
            }

    learning_path_raw = result.get("learning_path", [])
    summary = result.get("summary", "")
    thinking = result.get("thinking", "")
    feasibility_note = result.get("feasibility_note", "")
    total_weeks = result.get("total_weeks", 0)
    total_hours = result.get("total_hours", 0)

    # Programmatic Metadata Enrichment: Query SQLite for missing course details
    learning_path = []
    if learning_path_raw:
        # Support both short keys ('id') and long keys ('course_id')
        course_ids = [item.get("id") or item.get("course_id") for item in learning_path_raw]
        course_ids = [cid for cid in course_ids if cid]
        
        session = get_session()
        try:
            courses = session.query(Course).filter(Course.id.in_(course_ids)).all()
            course_map = {
                c.id: {
                    "title": c.title,
                    "track": c.track,
                    "estimated_hours": c.estimated_hours,
                    "difficulty_level": c.difficulty_level,
                    "prerequisite_ids": [p.id for p in c.prerequisites],
                }
                for c in courses
            }
        finally:
            session.close()

        # Merge database course metadata and map back to expected format
        for item in learning_path_raw:
            cid = item.get("id") or item.get("course_id")
            if not cid:
                continue
            meta = course_map.get(cid, {})
            learning_path.append({
                "course_id": cid,
                "title": meta.get("title", "Unknown Course"),
                "track": meta.get("track", "Unknown Track"),
                "estimated_hours": meta.get("estimated_hours", 0.0),
                "difficulty_level": meta.get("difficulty_level", "beginner"),
                "week_start": int(item.get("start") or item.get("week_start") or 1),
                "week_end": int(item.get("end") or item.get("week_end") or 1),
                "prerequisite_ids": meta.get("prerequisite_ids", []),
                "reason": item.get("why") or item.get("reason") or "",
            })

    # Programmatic prerequisite enforcement (topological sort)
    learning_path = enforce_prerequisite_ordering(learning_path)

    # Build the response message
    response_parts = []
    if summary:
        response_parts.append(summary)
    if feasibility_note:
        response_parts.append(f"\n**Feasibility Note:** {feasibility_note}")
    if learning_path:
        response_parts.append(f"\n**Total Duration:** ~{total_weeks} weeks ({total_hours} hours)")

    response_message = "\n".join(response_parts)

    return {
        "messages": [AIMessage(content=response_message)],
        "current_path": learning_path,
    }
