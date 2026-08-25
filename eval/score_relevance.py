"""
Relevance Scoring Script (40% criterion).
Runs the real agent against all 5 expert reference personas from
reference_paths.json, computes overlap scores and prerequisite ordering checks.
Saves results to eval/relevance_results.json.
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


def load_reference_paths():
    """Load the 5 expert reference personas."""
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_paths.json")
    with open(ref_path, "r") as f:
        data = json.load(f)
    return data["reference_paths"]


def load_course_catalog():
    """Load the course catalog to get prerequisite info."""
    catalog_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "course_catalog.json"
    )
    with open(catalog_path, "r") as f:
        data = json.load(f)
    # Build a lookup: course_id -> prerequisite_course_ids
    prereq_map = {}
    for course in data["courses"]:
        prereq_map[course["id"]] = course.get("prerequisite_course_ids", [])
    return prereq_map


def build_persona_message(persona):
    """Build a natural language message from persona fields."""
    parts = []
    if persona.get("goals"):
        parts.append(f"My goal is: {persona['goals']}.")
    if persona.get("background"):
        parts.append(f"My background: {persona['background']}.")
    if persona.get("time_availability"):
        parts.append(f"I can study {persona['time_availability']}.")
    return " ".join(parts)


def compute_overlap(actual_ids, expected_ids):
    """
    Compute overlap: |actual ∩ expected| / |expected|.
    Simple, defensible metric.
    """
    if not expected_ids:
        return 1.0
    actual_set = set(actual_ids)
    expected_set = set(expected_ids)
    intersection = actual_set & expected_set
    return len(intersection) / len(expected_set)


def check_prerequisite_ordering(actual_ids, prereq_map):
    """
    For each course in the actual path, check that all of its prerequisite
    course IDs (that are also in the path) appear at earlier positions.
    Returns a list of violations (empty if ordering is correct).
    """
    violations = []
    id_positions = {cid: i for i, cid in enumerate(actual_ids)}

    for i, course_id in enumerate(actual_ids):
        prereqs = prereq_map.get(course_id, [])
        for prereq_id in prereqs:
            if prereq_id in id_positions:
                if id_positions[prereq_id] >= i:
                    violations.append({
                        "course": course_id,
                        "prerequisite": prereq_id,
                        "course_position": i,
                        "prereq_position": id_positions[prereq_id],
                        "issue": f"{prereq_id} (pos {id_positions[prereq_id]}) should appear before {course_id} (pos {i})"
                    })

    return violations


def run_persona(persona, prereq_map):
    """Run the agent for one persona and evaluate the result."""
    persona_id = persona["persona_id"]
    print(f"\n{'='*60}")
    print(f"PERSONA: {persona_id} — {persona['description']}")
    print(f"{'='*60}")

    # Build the user message
    message = build_persona_message(persona)
    print(f"MESSAGE: {message}")

    # Build initial state
    state = {
        "messages": [HumanMessage(content=message)],
        "goals": "",
        "background": "",
        "time_availability": "",
        "learner_id": persona.get("learner_id", ""),
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

    # Run agent
    try:
        result = run_agent(state)
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "persona_id": persona_id,
            "description": persona["description"],
            "error": str(e),
            "overlap_score": 0.0,
            "ordering_violations": [],
            "ordering_correct": False,
        }

    # Extract actual path
    current_path = result.get("current_path", [])
    actual_ids = [c.get("course_id", "?") for c in current_path]
    expected_ids = [c["course_id"] for c in persona["expected_path"]]

    print(f"ACTUAL PATH:   {actual_ids}")
    print(f"EXPECTED PATH: {expected_ids}")

    # Compute overlap
    overlap = compute_overlap(actual_ids, expected_ids)
    print(f"OVERLAP SCORE: {overlap:.3f} ({len(set(actual_ids) & set(expected_ids))}/{len(expected_ids)} expected courses found)")

    # Check prerequisite ordering
    violations = check_prerequisite_ordering(actual_ids, prereq_map)
    ordering_correct = len(violations) == 0
    print(f"ORDERING: {'CORRECT' if ordering_correct else f'{len(violations)} VIOLATIONS'}")
    for v in violations:
        print(f"  ⚠ {v['issue']}")

    return {
        "persona_id": persona_id,
        "description": persona["description"],
        "goals": persona["goals"],
        "background": persona["background"],
        "time_availability": persona["time_availability"],
        "expected_path_ids": expected_ids,
        "actual_path_ids": actual_ids,
        "actual_path": current_path,
        "overlap_score": round(overlap, 3),
        "overlap_detail": f"{len(set(actual_ids) & set(expected_ids))}/{len(expected_ids)} expected courses found",
        "ordering_violations": violations,
        "ordering_correct": ordering_correct,
    }


def main():
    """Run relevance scoring for all 5 reference personas."""
    print("=" * 60)
    print("RELEVANCE SCORING — Reference Path Evaluation")
    print("=" * 60)

    personas = load_reference_paths()
    prereq_map = load_course_catalog()

    results = []
    total_overlap = 0.0
    total_ordering_violations = 0

    for persona in personas:
        result = run_persona(persona, prereq_map)
        results.append(result)
        total_overlap += result.get("overlap_score", 0.0)
        total_ordering_violations += len(result.get("ordering_violations", []))
        # Delay between personas to avoid rate limits
        time.sleep(2)

    # Summary
    n = len(personas)
    avg_overlap = total_overlap / n if n > 0 else 0.0

    print(f"\n\n{'='*60}")
    print("RELEVANCE SCORING SUMMARY")
    print(f"{'='*60}")
    print(f"Personas evaluated:         {n}")
    print(f"Average overlap score:      {avg_overlap:.3f}")
    print(f"Total ordering violations:  {total_ordering_violations}")
    print(f"Personas with perfect ordering: {sum(1 for r in results if r.get('ordering_correct', False))}/{n}")
    print(f"{'='*60}")

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "relevance_results.json")
    output = {
        "summary": {
            "num_personas": n,
            "avg_overlap_score": round(avg_overlap, 3),
            "total_ordering_violations": total_ordering_violations,
            "personas_with_perfect_ordering": sum(1 for r in results if r.get("ordering_correct", False)),
        },
        "detailed_results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {output_path}")
    return avg_overlap, total_ordering_violations


if __name__ == "__main__":
    main()
