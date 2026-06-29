"""One-time script to build and persist the Career Mentor FAISS index.

Run this script once to pre-compute the FAISS index so the application
loads it instantly at startup instead of rebuilding it every time.

Usage:
    python scripts/build_mentor_index.py

Output:
    vectorstore/mentor/mentor.index
    vectorstore/mentor/mentor_docs.json
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CAREER_NOTES_DIR, VECTORSTORE_DIR
from src.mentor.rag_chain import CareerMentorRAG


def build_index() -> None:
    """Load knowledge-base documents, build FAISS index, and save to disk."""
    knowledge_dir = str(CAREER_NOTES_DIR)
    output_dir = str(VECTORSTORE_DIR / "mentor")

    print(f"Knowledge base directory : {knowledge_dir}")
    print(f"Output directory         : {output_dir}")

    # We only need the index -- pass a dummy API key since we never call the LLM here.
    mentor = CareerMentorRAG(api_key="DUMMY_KEY", knowledge_dir=knowledge_dir)

    docs = mentor.load_documents(knowledge_dir)
    if not docs:
        print("ERROR: No documents found. Check the career_notes directory.")
        sys.exit(1)

    print(f"Loaded {len(docs)} document chunks")

    ok = mentor.build_index()
    if not ok:
        print("ERROR: Failed to build the FAISS index.")
        sys.exit(1)

    saved = mentor.save_index(output_dir)
    if saved:
        print(f"SUCCESS -- Index saved to {output_dir}")
        print(f"  mentor.index     ({mentor.index.ntotal} vectors)")
        print(f"  mentor_docs.json ({len(mentor.documents)} documents)")
    else:
        print("ERROR: Failed to save the index.")
        sys.exit(1)


if __name__ == "__main__":
    build_index()
