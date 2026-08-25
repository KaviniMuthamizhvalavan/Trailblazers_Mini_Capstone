"""
RAG Node: Retrieves relevant course content from the FAISS vector store
based on the learner's goals and track preferences.
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langsmith import traceable
import faiss


# Cache the model and index at module level
_model = None
_index = None
_documents = None
_metadata = None


def _get_faiss_resources():
    """Load and cache FAISS index, documents, and metadata."""
    global _model, _index, _documents, _metadata

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    if _index is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        index_dir = os.path.join(project_root, "data", "faiss_index")

        _index = faiss.read_index(os.path.join(index_dir, "course_index.faiss"))

        with open(os.path.join(index_dir, "documents.json"), "r") as f:
            data = json.load(f)
            _documents = data["documents"]
            _metadata = data["metadata"]

    return _model, _index, _documents, _metadata


@traceable(name="retrieve_courses", run_type="retriever")
def retrieve_courses(query: str, top_k: int = 15) -> tuple[list[str], list[dict]]:
    """
    Retrieve the top-k most relevant courses for a given query.
    Returns (document_texts, metadata_list).
    """
    model, index, documents, metadata = _get_faiss_resources()

    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = index.search(query_embedding, min(top_k, len(documents)))

    retrieved_docs = []
    retrieved_meta = []
    for i, idx in enumerate(indices[0]):
        if idx < len(documents) and scores[0][i] > 0.0:
            retrieved_docs.append(documents[idx])
            retrieved_meta.append(metadata[idx])

    return retrieved_docs, retrieved_meta


@traceable(name="rag_node", run_type="retriever")
def rag_node(state: dict) -> dict:
    """
    RAG retrieval node. Queries FAISS with the learner's goals/track.
    Stores full retrieved context in state for RAGAS evaluation and
    for persistence across refinement turns.
    """
    # Build a retrieval query from goals + background + constraints
    goals = state.get("goals", "")
    background = state.get("background", "")
    constraints = state.get("constraints", [])
    active_constraints = [c["constraint"] for c in constraints if not c.get("superseded", False)]

    query_parts = []
    if goals:
        query_parts.append(f"Learning goal: {goals}")
    if background:
        query_parts.append(f"Background: {background}")
    if active_constraints:
        query_parts.append(f"Constraints: {', '.join(active_constraints)}")

    query = ". ".join(query_parts) if query_parts else "corporate learning courses"

    # Retrieve courses
    docs, meta = retrieve_courses(query, top_k=15)

    return {
        "rag_context": docs,
        "rag_metadata": meta,
    }
