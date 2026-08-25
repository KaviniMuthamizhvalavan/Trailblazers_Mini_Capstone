"""
Tests for multi-turn refinement — verifying that intake_node correctly
accumulates constraints, preserves state fields, and handles contradictions.

These tests call the real intake_node() function with crafted state dicts,
not manual list simulations.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from agent.intake_node import intake_node
from agent.state import AgentState


class TestIntakeNodeRealCalls:
    """Tests that call the real intake_node with crafted state dicts."""

    def test_turn1_extracts_goals(self):
        """Turn 1: Feed a goal statement, verify goals are extracted."""
        state = {
            "messages": [HumanMessage(content="I want to become a Salesforce administrator.")],
            "goals": "",
            "background": "",
            "time_availability": "",
            "learner_id": "",
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

        result = intake_node(state)

        # Goals should be extracted
        assert result["goals"], "Goals should be non-empty after stating a goal"
        assert "salesforce" in result["goals"].lower(), f"Goals should mention salesforce, got: {result['goals']}"
        # Turn number should be 1
        assert result["turn_number"] == 1

    def test_turn2_adds_constraint_preserves_goals(self):
        """Turn 2: Add a constraint, verify it's appended and goals are preserved."""
        state = {
            "messages": [
                HumanMessage(content="I want to become a Salesforce administrator."),
                HumanMessage(content="I have no prior Salesforce experience at all."),
            ],
            "goals": "become a Salesforce administrator",
            "background": "",
            "time_availability": "",
            "learner_id": "",
            "constraints": [],
            "rag_context": [],
            "rag_metadata": [],
            "lms_completed": [],
            "lms_enrolled": [],
            "current_path": [],
            "is_refinement": False,
            "turn_number": 1,
            "error": None,
        }

        result = intake_node(state)

        # Goals should be preserved (not cleared)
        assert result["goals"], "Goals should be preserved from turn 1"
        # The 'no experience' info should be captured — either as a constraint or as background
        constraint_texts = [c["constraint"].lower() for c in result["constraints"]]
        background = (result.get("background") or "").lower()
        has_as_constraint = any("no prior" in ct or "no experience" in ct or "no salesforce" in ct
                                for ct in constraint_texts)
        has_as_background = "no" in background and ("experience" in background or "prior" in background or "salesforce" in background)
        assert has_as_constraint or has_as_background, \
            f"'No prior experience' should be captured as constraint or background. Got constraints: {constraint_texts}, background: '{background}'"

    def test_turn3_contradiction_preserves_time_availability(self):
        """
        Turn 3 (contradiction): User revises experience claim.
        The contradicted constraint should be superseded, but
        time_availability and other constraints must survive.
        """
        state = {
            "messages": [
                HumanMessage(content="I want to become a Salesforce administrator."),
                HumanMessage(content="I have no prior Salesforce experience at all."),
                HumanMessage(content="I only have 5 hours a week to study."),
                HumanMessage(content="Actually, I do have some Salesforce basics from my previous job."),
            ],
            "goals": "become a Salesforce administrator",
            "background": "",
            "time_availability": "5 hours per week",
            "learner_id": "",
            "constraints": [
                {"turn": 2, "constraint": "no prior Salesforce experience", "superseded": False},
            ],
            "rag_context": [],
            "rag_metadata": [],
            "lms_completed": [],
            "lms_enrolled": [],
            "current_path": [{"course_id": "SF-101"}],
            "is_refinement": False,
            "turn_number": 3,
            "error": None,
        }

        result = intake_node(state)

        # Time availability must survive the contradiction
        assert result["time_availability"], \
            f"time_availability should survive contradiction, got: '{result['time_availability']}'"
        assert "5" in result["time_availability"], \
            f"time_availability should still contain '5', got: '{result['time_availability']}'"

        # Goals should still be preserved
        assert result["goals"], "Goals should be preserved through contradiction"

        # The 'no prior' constraint should be superseded
        superseded = [c for c in result["constraints"] if c.get("superseded", False)]
        active = [c for c in result["constraints"] if not c.get("superseded", False)]

        assert any("no prior" in c["constraint"].lower() for c in superseded), \
            f"The 'no prior experience' constraint should be superseded, got superseded: {[c['constraint'] for c in superseded]}"


class TestStateSchema:
    """Test the AgentState schema structure."""

    def test_state_has_required_fields(self):
        """AgentState should have all required fields."""
        required_fields = [
            "messages", "goals", "background", "time_availability",
            "learner_id", "constraints", "rag_context", "rag_metadata",
            "lms_completed", "lms_enrolled", "current_path",
            "is_refinement", "turn_number", "error",
        ]
        annotations = AgentState.__annotations__
        for field in required_fields:
            assert field in annotations, f"AgentState missing required field: {field}"
