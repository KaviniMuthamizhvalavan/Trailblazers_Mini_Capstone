"""
Database models for the Learning Path Recommender.
SQLAlchemy ORM models for courses, prerequisites, and learner records.
"""

import os
import json
from sqlalchemy import create_engine, Column, String, Integer, Float, ForeignKey, Table, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db/learning_path.db")

# Handle relative SQLite paths
if DATABASE_URL.startswith("sqlite:///./"):
    db_path = DATABASE_URL.replace("sqlite:///./", "")
    abs_db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), db_path)
    os.makedirs(os.path.dirname(abs_db_path), exist_ok=True)
    DATABASE_URL = f"sqlite:///{abs_db_path}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# Association table for course prerequisites (many-to-many)
course_prerequisites = Table(
    "course_prerequisites",
    Base.metadata,
    Column("course_id", String, ForeignKey("courses.id"), primary_key=True),
    Column("prerequisite_id", String, ForeignKey("courses.id"), primary_key=True),
)

# Association table for completed courses (many-to-many)
learner_completed = Table(
    "learner_completed_courses",
    Base.metadata,
    Column("learner_id", String, ForeignKey("learners.learner_id"), primary_key=True),
    Column("course_id", String, ForeignKey("courses.id"), primary_key=True),
)

# Association table for enrolled courses (many-to-many)
learner_enrolled = Table(
    "learner_enrolled_courses",
    Base.metadata,
    Column("learner_id", String, ForeignKey("learners.learner_id"), primary_key=True),
    Column("course_id", String, ForeignKey("courses.id"), primary_key=True),
)


class Course(Base):
    __tablename__ = "courses"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    track = Column(String, nullable=False)
    estimated_hours = Column(Float, nullable=False)
    difficulty_level = Column(String, nullable=False)

    # Many-to-many self-referential for prerequisites
    prerequisites = relationship(
        "Course",
        secondary=course_prerequisites,
        primaryjoin=id == course_prerequisites.c.course_id,
        secondaryjoin=id == course_prerequisites.c.prerequisite_id,
        backref="required_by",
    )

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "track": self.track,
            "estimated_hours": self.estimated_hours,
            "difficulty_level": self.difficulty_level,
            "prerequisite_ids": [p.id for p in self.prerequisites],
        }


class Learner(Base):
    __tablename__ = "learners"

    learner_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String)
    department = Column(String)

    completed_courses = relationship(
        "Course", secondary=learner_completed, backref="completed_by"
    )
    enrolled_courses = relationship(
        "Course", secondary=learner_enrolled, backref="enrolled_by"
    )

    def to_dict(self):
        return {
            "learner_id": self.learner_id,
            "name": self.name,
            "email": self.email,
            "department": self.department,
            "completed_course_ids": [c.id for c in self.completed_courses],
            "enrolled_course_ids": [c.id for c in self.enrolled_courses],
        }


def init_db():
    """Create all tables."""
    Base.metadata.create_all(engine)


def get_session():
    """Get a new database session."""
    return SessionLocal()


if __name__ == "__main__":
    init_db()
    print("Database tables created successfully.")
