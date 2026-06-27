"""FAISS-based job search and matching with hybrid scoring."""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from src.search.embed import generate_embedding, generate_embeddings_batch
from src.normalization import normalize_skills, normalize_skill_name

logger = logging.getLogger(__name__)

ROLE_SIMILARITY_MAP: dict[str, list[str]] = {
    "software engineer": ["software engineer", "software developer", "backend engineer", "full stack engineer", "fullstack engineer", "sde", "software development engineer"],
    "software developer": ["software developer", "software engineer", "programmer", "application developer"],
    "data scientist": ["data scientist", "machine learning engineer", "ml engineer", "data analyst", "applied scientist", "research scientist"],
    "machine learning engineer": ["machine learning engineer", "ml engineer", "ml ops", "data scientist", "ai engineer", "deep learning engineer"],
    "data analyst": ["data analyst", "data scientist", "business analyst", "analytics engineer", "data engineer"],
    "data engineer": ["data engineer", "data architect", "data platform engineer", "etl developer", "big data engineer"],
    "devops engineer": ["devops engineer", "sre", "platform engineer", "cloud engineer", "infrastructure engineer", "site reliability engineer"],
    "backend engineer": ["backend engineer", "back end engineer", "backend developer", "server side developer", "software engineer"],
    "frontend engineer": ["frontend engineer", "front end engineer", "frontend developer", "ui developer", "web developer"],
    "full stack engineer": ["full stack engineer", "fullstack engineer", "full stack developer", "fullstack developer"],
    "product manager": ["product manager", "technical product manager", "product owner", "program manager"],
    "project manager": ["project manager", "program manager", "project coordinator", "scrum master"],
    "qa engineer": ["qa engineer", "quality assurance engineer", "test engineer", "software test engineer", "automation engineer"],
    "data architect": ["data architect", "data engineer", "data platform engineer", "solution architect"],
    "cloud engineer": ["cloud engineer", "devops engineer", "cloud architect", "cloud developer", "platform engineer"],
    "security engineer": ["security engineer", "cybersecurity engineer", "information security engineer", "soc analyst"],
    "ai engineer": ["ai engineer", "machine learning engineer", "deep learning engineer", "ai ml engineer", "generative ai engineer"],
    "research scientist": ["research scientist", "applied scientist", "research engineer", "ml researcher"],
    "business analyst": ["business analyst", "data analyst", "analytics manager", "business intelligence analyst"],
    "technical writer": ["technical writer", "documentation engineer", "technical content writer"],
}


def _normalize_role_title(title: str) -> str:
    """Normalize a role title for comparison.

    Args:
        title: Raw job title string.

    Returns:
        Normalized lowercase title.
    """
    return title.lower().strip().rstrip(".")


def _compute_role_similarity(candidate_role: str, job_title: str) -> float:
    """Compute role similarity between candidate target role and job title.

    Uses a combination of exact match, synonym map lookup, and keyword overlap.

    Args:
        candidate_role: Candidate's target role.
        job_title: Job posting title.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not candidate_role or not job_title:
        return 0.0

    cand_norm = _normalize_role_title(candidate_role)
    job_norm = _normalize_role_title(job_title)

    if cand_norm == job_norm:
        return 1.0

    for canonical, aliases in ROLE_SIMILARITY_MAP.items():
        cand_match = cand_norm in aliases or cand_norm == canonical
        job_match = job_norm in aliases or job_norm == canonical
        if cand_match and job_match:
            if cand_norm == job_norm:
                return 1.0
            if cand_norm == canonical or job_norm == canonical:
                return 0.85
            return 0.7

    cand_words = set(cand_norm.replace("/", " ").replace("-", " ").split())
    job_words = set(job_norm.replace("/", " ").replace("-", " ").split())
    if not cand_words or not job_words:
        return 0.0

    intersection = cand_words & job_words
    if not intersection:
        return 0.0

    common = len(intersection)
    total = max(len(cand_words), len(job_words))
    return common / total


def _compute_experience_similarity(candidate_experience: list[str], job_experience: str) -> float:
    """Compute experience relevance between candidate and job requirements.

    Args:
        candidate_experience: List of candidate experience entries.
        job_experience: Job experience requirement string.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not candidate_experience:
        return 0.3

    if not job_experience:
        return 0.8

    job_years_pattern = r"(\d+)\s*(?:\+|\-)?\s*(?:years?|yrs?)"
    import re
    job_match = re.search(job_years_pattern, job_experience.lower())
    if not job_match:
        return 0.5

    job_years = int(job_match.group(1))

    total_years = 0.0
    for exp in candidate_experience:
        exp_match = re.search(job_years_pattern, exp.lower())
        if exp_match:
            total_years += float(exp_match.group(1))
        else:
            total_years += 1.0

    if total_years >= job_years * 1.5:
        return 1.0
    if total_years >= job_years:
        return 0.9
    if total_years >= job_years * 0.75:
        return 0.7
    if total_years >= job_years * 0.5:
        return 0.5
    return 0.3


def _compute_education_similarity(candidate_education: list[str], job_education: str) -> float:
    """Compute education relevance between candidate and job requirements.

    Args:
        candidate_education: List of candidate education entries.
        job_education: Job education requirement string.

    Returns:
        Similarity score between 0.0 and 1.0.
    """
    if not candidate_education:
        return 0.3

    if not job_education:
        return 0.8

    edu_text = " ".join(candidate_education).lower()
    job_edu_lower = job_education.lower()

    degree_levels = [
        ("ph.d", "doctorate", "phd"),
        ("master", "ms", "m.s", "ma", "m.a", "mba"),
        ("bachelor", "bs", "b.s", "ba", "b.a", "btech", "b.tech"),
        ("associate", "diploma"),
    ]

    job_degree_level = -1
    for level, keywords in enumerate(degree_levels):
        for kw in keywords:
            if kw in job_edu_lower:
                job_degree_level = level
                break
        if job_degree_level >= 0:
            break

    cand_degree_level = -1
    for level, keywords in enumerate(degree_levels):
        for kw in keywords:
            if kw in edu_text:
                cand_degree_level = level
                break
        if cand_degree_level >= 0:
            break

    if job_degree_level == -1:
        return 0.8
    if cand_degree_level == -1:
        return 0.3
    if cand_degree_level <= job_degree_level:
        return 1.0
    return 0.5


def calculate_skill_match(
    candidate_skills: list[str],
    job_skills: str,
) -> tuple[list[str], list[str], list[str], list[str], float]:
    """Calculate matched and missing skills using normalized comparison.

    Uses centralized normalization to handle aliases, casing, and duplicates.

    Args:
        candidate_skills: List of candidate's skills.
        job_skills: Comma-separated string of required job skills.

    Returns:
        Tuple of (matched_skills, missing_skills, required_missing, preferred_missing, skill_score).
    """
    job_skills_raw = [s.strip() for s in job_skills.split(",") if s.strip()]
    job_normalized = normalize_skills(job_skills_raw)
    candidate_normalized = normalize_skills(candidate_skills)

    cand_lower = set(s.lower() for s in candidate_normalized)

    matched = []
    missing = []

    for skill in job_normalized:
        skill_lower = skill.lower()
        if skill_lower in cand_lower:
            matched.append(skill)
        else:
            missing.append(skill)

    required_missing = missing[:]
    preferred_missing = []

    total = len(job_normalized)
    if total > 0:
        skill_score = len(matched) / total
    else:
        skill_score = 0.5

    return matched, missing, required_missing, preferred_missing, skill_score


def compute_hybrid_score(
    semantic_score: float,
    skill_score: float,
    role_similarity: float,
    experience_similarity: float,
    education_similarity: float,
    candidate_role: str,
    job_title: str,
) -> dict:
    """Compute a hybrid match score combining multiple signals.

    The final score combines:
    - Semantic similarity (FAISS)
    - Skill overlap
    - Role similarity
    - Experience relevance
    - Education relevance

    A penalty is applied for major role mismatches.

    Args:
        semantic_score: FAISS cosine similarity score.
        skill_score: Fraction of job skills matched.
        role_similarity: Role title similarity (0-1).
        experience_similarity: Experience relevance (0-1).
        education_similarity: Education relevance (0-1).
        candidate_role: Candidate's target role.
        job_title: Job posting title.

    Returns:
        Dict with final_score, components, penalty info.
    """
    role_penalty = 1.0
    if candidate_role and role_similarity < 0.2:
        role_penalty = 0.3
    elif candidate_role and role_similarity < 0.4:
        role_penalty = 0.6

    weights = {
        "semantic": 0.30,
        "skill": 0.25,
        "role": 0.20,
        "experience": 0.15,
        "education": 0.10,
    }

    raw_score = (
        weights["semantic"] * semantic_score
        + weights["skill"] * skill_score
        + weights["role"] * role_similarity
        + weights["experience"] * experience_similarity
        + weights["education"] * education_similarity
    )

    final_score = raw_score * role_penalty

    final_score = max(0.0, min(1.0, final_score))

    return {
        "final_score": round(final_score, 4),
        "components": {
            "semantic": round(semantic_score, 4),
            "skill_overlap": round(skill_score, 4),
            "role_similarity": round(role_similarity, 4),
            "experience_similarity": round(experience_similarity, 4),
            "education_similarity": round(education_similarity, 4),
        },
        "role_penalty": round(role_penalty, 4),
    }


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

    def search_by_embedding(self, embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search for similar jobs using a pre-computed embedding vector.

        Avoids re-embedding when the same candidate profile is used.

        Args:
            embedding: Pre-computed embedding vector (1D numpy array).
            top_k: Number of results to return.

        Returns:
            List of matched jobs with scores.
        """
        if self.index is None:
            logger.error("FAISS index not built")
            return []

        try:
            query_vector = embedding.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.job_data):
                    job = self.job_data[idx].copy()
                    job["match_score"] = round(float(score), 4)
                    results.append(job)

            logger.info(f"Found {len(results)} matching jobs from stored embedding")
            return results

        except Exception as e:
            logger.error(f"Error during embedding search: {e}")
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

    def hybrid_search(
        self,
        resume_data,
        top_k: int = 5,
    ) -> list[dict]:
        """Perform hybrid search with skill matching and reranking.

        Runs FAISS semantic search first, then reranks results using
        hybrid scoring that considers skill overlap, role similarity,
        experience, and education.

        Args:
            resume_data: ResumeData object with parsed resume.
            top_k: Number of results to return.

        Returns:
            List of jobs with hybrid scores and match details.
        """
        raw_results = self.search_by_embedding(
            resume_data.candidate_embedding, top_k=top_k * 2
        ) if hasattr(resume_data, "candidate_embedding") and resume_data.candidate_embedding is not None else []

        if not raw_results and hasattr(resume_data, "skills"):
            query = f"{resume_data.target_role} {' '.join(resume_data.skills)}"
            raw_results = self.search(query, top_k=top_k * 2)

        if not raw_results:
            return []

        enriched = []
        for job in raw_results:
            matched, missing, req_missing, pref_missing, skill_score = calculate_skill_match(
                resume_data.skills, job.get("skills", "")
            )

            role_sim = _compute_role_similarity(
                resume_data.target_role, job.get("title", "")
            )
            exp_sim = _compute_experience_similarity(
                resume_data.experience, job.get("experience", "")
            )
            edu_sim = _compute_education_similarity(
                resume_data.education, job.get("education", "")
            )

            hybrid = compute_hybrid_score(
                semantic_score=job.get("match_score", 0),
                skill_score=skill_score,
                role_similarity=role_sim,
                experience_similarity=exp_sim,
                education_similarity=edu_sim,
                candidate_role=resume_data.target_role,
                job_title=job.get("title", ""),
            )

            job["match_score"] = hybrid["final_score"]
            job["hybrid_score_components"] = hybrid["components"]
            job["role_penalty"] = hybrid["role_penalty"]
            job["matched_skills"] = matched
            job["missing_skills"] = missing
            job["required_missing_skills"] = req_missing
            job["preferred_missing_skills"] = pref_missing
            job["skill_overlap_score"] = round(skill_score, 4)
            job["role_similarity"] = round(role_sim, 4)
            enriched.append(job)

        enriched.sort(key=lambda j: j["match_score"], reverse=True)
        enriched = enriched[:top_k]

        return enriched
