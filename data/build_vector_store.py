"""
Build a FAISS vector store from the course catalog.
Embeds course descriptions + prerequisite info for RAG retrieval.
"""

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
import faiss


def build_course_documents(catalog_path):
    """Create text documents from course catalog for embedding."""
    with open(catalog_path, "r") as f:
        data = json.load(f)

    # Build a lookup for course titles
    title_lookup = {c["id"]: c["title"] for c in data["courses"]}

    documents = []
    metadata_list = []

    for course in data["courses"]:
        # Build prerequisite text
        prereq_names = []
        for pid in course["prerequisite_course_ids"]:
            if pid in title_lookup:
                prereq_names.append(f"{pid}: {title_lookup[pid]}")

        prereq_text = ""
        if prereq_names:
            prereq_text = f" Prerequisites: {', '.join(prereq_names)}."
        else:
            prereq_text = " No prerequisites required."

        # Build the document text
        doc_text = (
            f"Course: {course['title']} (ID: {course['id']}). "
            f"Track: {course['track']}. "
            f"Difficulty: {course['difficulty_level']}. "
            f"Estimated hours: {course['estimated_hours']}.{prereq_text} "
            f"Description: {course['description']}"
        )

        documents.append(doc_text)
        metadata_list.append({
            "course_id": course["id"],
            "title": course["title"],
            "track": course["track"],
            "difficulty_level": course["difficulty_level"],
            "estimated_hours": course["estimated_hours"],
            "prerequisite_course_ids": course["prerequisite_course_ids"],
        })

    return documents, metadata_list


def build_vector_store(documents, metadata_list, output_dir):
    """Embed documents and build FAISS index."""
    print(f"Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Embedding {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build FAISS index (Inner Product after normalization = cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "course_index.faiss"))

    # Save documents and metadata alongside
    with open(os.path.join(output_dir, "documents.json"), "w") as f:
        json.dump({"documents": documents, "metadata": metadata_list}, f, indent=2)

    print(f"FAISS index saved to {output_dir} ({index.ntotal} vectors, dimension {dimension})")
    return index


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    catalog_path = os.path.join(project_root, "data", "course_catalog.json")
    output_dir = os.path.join(project_root, "data", "faiss_index")

    documents, metadata_list = build_course_documents(catalog_path)
    build_vector_store(documents, metadata_list, output_dir)


if __name__ == "__main__":
    main()
