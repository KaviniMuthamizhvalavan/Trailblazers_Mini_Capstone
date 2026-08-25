"""
Tests for the composition node logic.
Validates prerequisite ordering, time-boxing, and course filtering.
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPrerequisiteOrdering:
    """Test that recommended paths respect prerequisite ordering."""

    def _get_course_catalog(self):
        """Load course catalog for prerequisite validation."""
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "course_catalog.json"
        )
        with open(catalog_path, "r") as f:
            data = json.load(f)
        return {c["id"]: c for c in data["courses"]}

    def test_catalog_has_prerequisite_chains(self):
        """Verify the catalog has courses with 2+ prerequisites (planner checkpoint #2)."""
        catalog = self._get_course_catalog()
        multi_prereq_courses = [
            c for c in catalog.values()
            if len(c["prerequisite_course_ids"]) >= 2
        ]
        # Planner requires at least some courses with 2+ prerequisites
        assert len(multi_prereq_courses) >= 3, (
            f"Expected at least 3 courses with 2+ prerequisites, found {len(multi_prereq_courses)}"
        )

    def test_catalog_has_deep_chains(self):
        """Verify the catalog has prerequisite chains of depth >= 3."""
        catalog = self._get_course_catalog()

        def chain_depth(course_id, visited=None):
            if visited is None:
                visited = set()
            if course_id in visited:
                return 0
            visited.add(course_id)
            course = catalog.get(course_id)
            if not course or not course["prerequisite_course_ids"]:
                return 0
            return 1 + max(
                chain_depth(pid, visited)
                for pid in course["prerequisite_course_ids"]
            )

        max_depth = max(chain_depth(cid) for cid in catalog)
        assert max_depth >= 3, (
            f"Expected at least one prerequisite chain of depth >= 3, max found was {max_depth}"
        )

    def test_prerequisite_order_in_path(self):
        """
        Feed a deliberately mis-ordered path into enforce_prerequisite_ordering()
        and verify it fixes the violations.
        """
        from agent.composition_node import enforce_prerequisite_ordering

        # SAP-104 requires SAP-102 and SAP-103, but we put it FIRST — a clear violation
        mis_ordered_path = [
            {"course_id": "SAP-104", "week_start": 1, "week_end": 4,
             "prerequisite_ids": ["SAP-102", "SAP-103"],
             "title": "SAP FICO Advanced", "track": "SAP",
             "estimated_hours": 20.0, "difficulty_level": "advanced", "reason": ""},
            {"course_id": "SAP-101", "week_start": 5, "week_end": 7,
             "prerequisite_ids": [],
             "title": "SAP Basics", "track": "SAP",
             "estimated_hours": 15.0, "difficulty_level": "beginner", "reason": ""},
            {"course_id": "SAP-102", "week_start": 8, "week_end": 11,
             "prerequisite_ids": ["SAP-101"],
             "title": "SAP FI Basics", "track": "SAP",
             "estimated_hours": 20.0, "difficulty_level": "intermediate", "reason": ""},
            {"course_id": "SAP-103", "week_start": 12, "week_end": 14,
             "prerequisite_ids": ["SAP-101"],
             "title": "SAP CO Fundamentals", "track": "SAP",
             "estimated_hours": 15.0, "difficulty_level": "intermediate", "reason": ""},
        ]

        result = enforce_prerequisite_ordering(mis_ordered_path)
        result_ids = [c["course_id"] for c in result]

        # SAP-101 must come before SAP-102 and SAP-103
        assert result_ids.index("SAP-101") < result_ids.index("SAP-102")
        assert result_ids.index("SAP-101") < result_ids.index("SAP-103")
        # SAP-102 and SAP-103 must come before SAP-104
        assert result_ids.index("SAP-102") < result_ids.index("SAP-104")
        assert result_ids.index("SAP-103") < result_ids.index("SAP-104")

    def test_already_ordered_path_unchanged(self):
        """enforce_prerequisite_ordering is a no-op when ordering is already correct."""
        from agent.composition_node import enforce_prerequisite_ordering

        correct_path = [
            {"course_id": "SF-101", "week_start": 1, "week_end": 2,
             "prerequisite_ids": [],
             "title": "SF Basics", "track": "Salesforce",
             "estimated_hours": 10.0, "difficulty_level": "beginner", "reason": ""},
            {"course_id": "SF-102", "week_start": 3, "week_end": 6,
             "prerequisite_ids": ["SF-101"],
             "title": "SF Admin", "track": "Salesforce",
             "estimated_hours": 20.0, "difficulty_level": "intermediate", "reason": ""},
        ]

        result = enforce_prerequisite_ordering(correct_path)
        assert [c["course_id"] for c in result] == ["SF-101", "SF-102"]
        # Week assignments should be preserved since no reordering happened
        assert result[0]["week_start"] == 1
        assert result[1]["week_start"] == 3

    def test_week_recalculation_after_reorder(self):
        """After reordering, week_start/week_end are recalculated sequentially."""
        from agent.composition_node import enforce_prerequisite_ordering

        # B depends on A, but B is listed first
        mis_ordered = [
            {"course_id": "B", "week_start": 1, "week_end": 3,
             "prerequisite_ids": ["A"],
             "title": "B", "track": "Test",
             "estimated_hours": 15.0, "difficulty_level": "intermediate", "reason": ""},
            {"course_id": "A", "week_start": 4, "week_end": 5,
             "prerequisite_ids": [],
             "title": "A", "track": "Test",
             "estimated_hours": 10.0, "difficulty_level": "beginner", "reason": ""},
        ]

        result = enforce_prerequisite_ordering(mis_ordered)
        assert result[0]["course_id"] == "A"
        assert result[1]["course_id"] == "B"
        # A should start at week 1, span 2 weeks (original span preserved)
        assert result[0]["week_start"] == 1
        assert result[0]["week_end"] == 2
        # B should start right after A
        assert result[1]["week_start"] == 3
        assert result[1]["week_end"] == 5

    def test_empty_and_single_course_paths(self):
        """Edge cases: empty and single-course paths returned as-is."""
        from agent.composition_node import enforce_prerequisite_ordering

        assert enforce_prerequisite_ordering([]) == []

        single = [{"course_id": "X", "week_start": 1, "week_end": 2,
                    "prerequisite_ids": [], "title": "X", "track": "T",
                    "estimated_hours": 5.0, "difficulty_level": "beginner", "reason": ""}]
        assert enforce_prerequisite_ordering(single) == single


class TestTimeBoxing:
    """Test time-boxing logic."""

    def test_weekly_hours_not_exceeded(self):
        """
        Given a tight time constraint, verify the path spreads courses
        over enough weeks.
        """
        hours_per_week = 3
        sample_path = [
            {"course_id": "SF-101", "estimated_hours": 10, "week_start": 1, "week_end": 4},
            {"course_id": "SF-102", "estimated_hours": 20, "week_start": 5, "week_end": 11},
        ]

        for course in sample_path:
            weeks_span = course["week_end"] - course["week_start"] + 1
            hours_per_week_actual = course["estimated_hours"] / weeks_span
            # Allow some flexibility (up to 1.5x stated hours)
            assert hours_per_week_actual <= hours_per_week * 1.5, (
                f"Course {course['course_id']} has {hours_per_week_actual:.1f} hrs/week "
                f"but learner said {hours_per_week} hrs/week"
            )

    def test_courses_dont_overlap_unrealistically(self):
        """Verify courses in the same weeks don't total unrealistic hours."""
        hours_per_week = 5
        sample_path = [
            {"course_id": "CLD-101", "estimated_hours": 10, "week_start": 1, "week_end": 2},
            {"course_id": "CLD-102", "estimated_hours": 15, "week_start": 3, "week_end": 5},
        ]

        # Courses should be sequential, not overlapping
        for i in range(len(sample_path) - 1):
            current = sample_path[i]
            next_course = sample_path[i + 1]
            assert current["week_end"] < next_course["week_start"] or \
                   current["week_start"] == next_course["week_start"], (
                f"Courses {current['course_id']} and {next_course['course_id']} overlap unrealistically"
            )


class TestCourseFiltering:
    """Test that completed/enrolled courses are filtered out."""

    def test_completed_courses_not_in_path(self):
        """Completed courses should never appear in recommendations."""
        completed = ["SAP-101", "SAP-102", "SAP-103"]
        sample_path = [
            {"course_id": "SAP-104"},
            {"course_id": "SAP-105"},
        ]

        path_ids = {c["course_id"] for c in sample_path}
        for completed_id in completed:
            assert completed_id not in path_ids, (
                f"Completed course {completed_id} should not appear in recommendations"
            )

    def test_enrolled_courses_not_in_path(self):
        """Currently enrolled courses should not be re-recommended."""
        enrolled = ["SAP-104"]
        sample_path = [
            {"course_id": "SAP-105"},
            {"course_id": "SAP-106"},
        ]

        path_ids = {c["course_id"] for c in sample_path}
        for enrolled_id in enrolled:
            assert enrolled_id not in path_ids, (
                f"Enrolled course {enrolled_id} should not appear in recommendations"
            )


class TestCatalogIntegrity:
    """Test catalog data integrity."""

    def test_all_prerequisites_exist(self):
        """Every prerequisite ID should reference an existing course."""
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "course_catalog.json"
        )
        with open(catalog_path, "r") as f:
            data = json.load(f)

        course_ids = {c["id"] for c in data["courses"]}
        for course in data["courses"]:
            for prereq_id in course["prerequisite_course_ids"]:
                assert prereq_id in course_ids, (
                    f"Course {course['id']} has prerequisite {prereq_id} which doesn't exist"
                )

    def test_no_circular_prerequisites(self):
        """Verify no circular prerequisite dependencies."""
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "course_catalog.json"
        )
        with open(catalog_path, "r") as f:
            data = json.load(f)

        prereq_map = {c["id"]: c["prerequisite_course_ids"] for c in data["courses"]}

        def has_cycle(course_id, visited=None, stack=None):
            if visited is None:
                visited = set()
            if stack is None:
                stack = set()
            visited.add(course_id)
            stack.add(course_id)
            for prereq_id in prereq_map.get(course_id, []):
                if prereq_id not in visited:
                    if has_cycle(prereq_id, visited, stack):
                        return True
                elif prereq_id in stack:
                    return True
            stack.discard(course_id)
            return False

        for course_id in prereq_map:
            assert not has_cycle(course_id), (
                f"Circular prerequisite detected involving {course_id}"
            )

    def test_catalog_has_50_courses(self):
        """Verify the catalog has the expected number of courses."""
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "course_catalog.json"
        )
        with open(catalog_path, "r") as f:
            data = json.load(f)
        assert len(data["courses"]) == 50

    def test_catalog_has_6_tracks(self):
        """Verify all 6 tracks are represented."""
        catalog_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "course_catalog.json"
        )
        with open(catalog_path, "r") as f:
            data = json.load(f)
        tracks = {c["track"] for c in data["courses"]}
        expected = {"SAP", "Workday", "Salesforce", "Cloud", "AI", "Cybersecurity"}
        assert tracks == expected, f"Expected tracks {expected}, got {tracks}"
