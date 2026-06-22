"""FAISS-based job search and matching."""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.search.embed import generate_embedding, generate_embeddings_batch

logger = logging.getLogger(__name__)


class JobSearchEngine:
    """Semantic job search engine using FAISS."""

    def __init__(self, index_path: Optional[str] = None):
        """Initialize the job search engine.

        Args:
            index_path: Path to load existing FAISS index.
        """
        self.index: Optional[faiss.IndexFlatIP] = None
        self.job_data: list[dict] = []
        self.index_path = index_path

        if index_path and Path(index_path).exists():
            self.load_index(index_path)

    def load_jobs(self, jobs_path: str) -> list[dict]:
        """Load jobs from a JSON file.

        Args:
            jobs_path: Path to the jobs JSON file.

        Returns:
            List of job dictionaries.
        """
        try:
            with open(jobs_path, "r", encoding="utf-8") as f:
                self.job_data = json.load(f)
            logger.info(f"Loaded {len(self.job_data)} jobs from {jobs_path}")
            return self.job_data
        except Exception as e:
            logger.error(f"Error loading jobs: {e}")
            return []

    def build_index(self, jobs: Optional[list[dict]] = None) -> bool:
        """Build FAISS index from job data.

        Args:
            jobs: Optional list of job dicts. Uses loaded jobs if not provided.

        Returns:
            True if index built successfully, False otherwise.
        """
        if jobs:
            self.job_data = jobs

        if not self.job_data:
            logger.error("No job data available to build index")
            return False

        try:
            texts = []
            for job in self.job_data:
                parts = [
                    job.get("title", ""),
                    job.get("skills", ""),
                    job.get("description", ""),
                    job.get("experience", ""),
                    job.get("education", ""),
                ]
                texts.append(" ".join(p for p in parts if p))

            embeddings = generate_embeddings_batch(texts)
            if embeddings is None:
                return False

            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype(np.float32))

            logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for similar jobs using semantic similarity.

        Args:
            query: Query text (e.g., candidate profile).
            top_k: Number of results to return.

        Returns:
            List of matched jobs with scores.
        """
        if self.index is None:
            logger.error("FAISS index not built")
            return []

        try:
            query_embedding = generate_embedding(query)
            if query_embedding is None:
                return []

            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.job_data):
                    job = self.job_data[idx].copy()
                    job["match_score"] = round(float(score), 4)
                    results.append(job)

            logger.info(f"Found {len(results)} matching jobs")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []

    def save_index(self, path: str) -> bool:
        """Save FAISS index and job data to disk.

        Args:
            path: Directory path to save index files.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, f"{path}/jobs.index")
            with open(f"{path}/jobs.json", "w", encoding="utf-8") as f:
                json.dump(self.job_data, f, indent=2)
            logger.info(f"Saved index to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False

    def load_index(self, path: str) -> bool:
        """Load FAISS index and job data from disk.

        Args:
            path: Directory path containing index files.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            self.index = faiss.read_index(f"{path}/jobs.index")
            with open(f"{path}/jobs.json", "r", encoding="utf-8") as f:
                self.job_data = json.load(f)
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False


def calculate_skill_match(
    candidate_skills: list[str],
    job_skills: str,
) -> tuple[list[str], list[str]]:
    """Calculate matched and missing skills.

    Uses exact match and normalized overlap to avoid false positives
    (e.g., "c" matching "css", or "ai" matching "data analyst").

    Args:
        candidate_skills: List of candidate's skills.
        job_skills: Comma-separated string of required job skills.

    Returns:
        Tuple of (matched_skills, missing_skills).
    """
    job_skills_list = [s.strip().lower() for s in job_skills.split(",") if s.strip()]
    candidate_normalized = [s.strip().lower() for s in candidate_skills]

    matched = []
    missing = []

    for skill in job_skills_list:
        is_matched = False
        for cs in candidate_normalized:
            if skill == cs:
                is_matched = True
                break
            if len(skill) >= 3 and len(cs) >= 3:
                if skill in cs or cs in skill:
                    is_matched = True
                    break

        if is_matched:
            matched.append(skill.title())
        else:
            missing.append(skill.title())

    return matched, missing
