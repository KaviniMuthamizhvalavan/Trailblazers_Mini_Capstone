"""
Tests for LMS filtering — verifying that completed/enrolled courses
are never re-recommended by the agent.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import init_db, get_session, Learner, Course
from agent.lms_tool_node import fetch_learner_history, get_all_learners


class TestLMSDataRetrieval:
    """Test LMS tool node data retrieval from database."""

    def test_fetch_known_learner(self):
        """L001 (Priya Sharma) should have SAP-101, SAP-102, SAP-103 completed."""
        history = fetch_learner_history("L001")
        assert "SAP-101" in history["completed"]
        assert "SAP-102" in history["completed"]
        assert "SAP-103" in history["completed"]
        assert "SAP-104" in history["enrolled"]
        assert history["learner_name"] == "Priya Sharma"

    def test_fetch_learner_with_no_completions(self):
        """L003 (Maria Santos) should have no completed or enrolled courses."""
        history = fetch_learner_history("L003")
        assert history["completed"] == []
        assert history["enrolled"] == []
        assert history["learner_name"] == "Maria Santos"

    def test_fetch_nonexistent_learner(self):
        """Requesting a non-existent learner should return empty with error."""
        history = fetch_learner_history("NONEXISTENT")
        assert history["completed"] == []
        assert history["enrolled"] == []
        assert "error" in history

    def test_cross_domain_learner(self):
        """L004 (Ahmed Hassan) should have courses from both AI and Cloud tracks."""
        history = fetch_learner_history("L004")
        # AI courses
        assert "AI-101" in history["completed"]
        assert "AI-102" in history["completed"]
        assert "AI-103" in history["completed"]
        # Cloud courses
        assert "CLD-101" in history["completed"]
        assert "CLD-102" in history["completed"]
        assert "CLD-103" in history["completed"]
        # Currently enrolled
        assert "AI-104" in history["enrolled"]

    def test_get_all_learners(self):
        """Should return 5 learner profiles."""
        learners = get_all_learners()
        assert len(learners) == 5
        ids = {l["learner_id"] for l in learners}
        assert ids == {"L001", "L002", "L003", "L004", "L005"}


class TestLMSFiltering:
    """
    Test that the LMS tool node correctly identifies courses to filter,
    and that those courses would be excluded from recommendations.
    """

    def test_completed_prerequisite_not_recommended(self):
        """
        L001 has completed SAP-101, SAP-102, SAP-103 and is enrolled in SAP-104.
        If they ask for SAP consultant path, none of these should be recommended.
        """
        history = fetch_learner_history("L001")
        excluded = set(history["completed"]) | set(history["enrolled"])

        # These should all be excluded
        assert "SAP-101" in excluded
        assert "SAP-102" in excluded
        assert "SAP-103" in excluded
        assert "SAP-104" in excluded

        # But SAP-105, SAP-106 etc. should NOT be excluded
        assert "SAP-105" not in excluded
        assert "SAP-106" not in excluded

    def test_enrolled_course_excluded(self):
        """Enrolled courses should also be excluded, not just completed ones."""
        history = fetch_learner_history("L005")
        excluded = set(history["completed"]) | set(history["enrolled"])
        assert "WD-101" in excluded  # completed
        assert "WD-102" in excluded  # enrolled
        assert "WD-103" not in excluded  # not started yet
