"""
Seed the database with synthetic course catalog and learner records.
Loads data from JSON files in the data/ directory.
Idempotent — safe to re-run.
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.models import init_db, get_session, Course, Learner, course_prerequisites, learner_completed, learner_enrolled


def seed_courses(session, catalog_path):
    """Load courses and prerequisites from JSON into the database."""
    with open(catalog_path, "r") as f:
        data = json.load(f)

    courses = data["courses"]

    # First pass: create all course records (without prerequisites)
    for course_data in courses:
        existing = session.query(Course).filter_by(id=course_data["id"]).first()
        if existing:
            continue
        course = Course(
            id=course_data["id"],
            title=course_data["title"],
            description=course_data["description"],
            track=course_data["track"],
            estimated_hours=course_data["estimated_hours"],
            difficulty_level=course_data["difficulty_level"],
        )
        session.add(course)

    session.commit()
    print(f"  Loaded {len(courses)} courses.")

    # Second pass: set up prerequisite relationships
    prereq_count = 0
    for course_data in courses:
        if not course_data["prerequisite_course_ids"]:
            continue

        course = session.query(Course).filter_by(id=course_data["id"]).first()
        for prereq_id in course_data["prerequisite_course_ids"]:
            prereq = session.query(Course).filter_by(id=prereq_id).first()
            if prereq and prereq not in course.prerequisites:
                course.prerequisites.append(prereq)
                prereq_count += 1

    session.commit()
    print(f"  Loaded {prereq_count} prerequisite relationships.")


def seed_learners(session, learners_path):
    """Load learner profiles and their course histories from JSON."""
    with open(learners_path, "r") as f:
        data = json.load(f)

    learners = data["learners"]

    for learner_data in learners:
        existing = session.query(Learner).filter_by(learner_id=learner_data["learner_id"]).first()
        if existing:
            continue

        learner = Learner(
            learner_id=learner_data["learner_id"],
            name=learner_data["name"],
            email=learner_data["email"],
            department=learner_data["department"],
        )
        session.add(learner)
        session.flush()

        # Add completed courses
        for course_id in learner_data.get("completed_courses", []):
            course = session.query(Course).filter_by(id=course_id).first()
            if course:
                learner.completed_courses.append(course)

        # Add enrolled courses
        for course_id in learner_data.get("enrolled_courses", []):
            course = session.query(Course).filter_by(id=course_id).first()
            if course:
                learner.enrolled_courses.append(course)

    session.commit()
    print(f"  Loaded {len(learners)} learner profiles.")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(project_root, "data", "course_catalog.json")
    learners_path = os.path.join(project_root, "data", "learner_records.json")

    print("Initializing database...")
    init_db()

    session = get_session()
    try:
        print("Seeding courses...")
        seed_courses(session, catalog_path)

        print("Seeding learners...")
        seed_learners(session, learners_path)

        print("Database seeded successfully!")
    finally:
        session.close()


if __name__ == "__main__":
    main()
