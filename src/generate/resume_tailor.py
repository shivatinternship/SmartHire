"""Explainable Resume Tailoring Module - generates job-aligned resume with full transparency."""

import json
import logging
import re
import time
from difflib import SequenceMatcher
from typing import Optional

import httpx

from src.generate.prompts import RESUME_TAILOR_PROMPT
from src.parsing.resume_parser import ResumeData, _parse_llm_json_response
from src.config import GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2

REQUIRED_RESULT_KEYS = {
    "tailored_resume",
    "section_explanations",
    "change_log",
    "integrity_report",
    "tailoring_plan",
    "confidence_scores",
}

_tailor_cache: dict[str, Optional[dict]] = {}


def _cache_key(resume_id: str, job_id: str) -> str:
    """Generate cache key for tailoring results.

    Args:
        resume_id: Resume identifier.
        job_id: Job identifier.

    Returns:
        Cache key string.
    """
    return f"tailor|{resume_id}|{job_id}"


def _validate_result(result: Optional[dict]) -> bool:
    """Check that the parsed result has the expected top-level keys."""
    if not isinstance(result, dict):
        return False
    return REQUIRED_RESULT_KEYS.issubset(result.keys())


def _extract_experience_identifiers(exp_text: str) -> set[str]:
    """Extract key identifiers from an experience entry for fuzzy matching.

    Extracts company names, dates, job titles, and other key phrases that
    should be preserved when an entry is rewritten.

    Args:
        exp_text: Experience entry text.

    Returns:
        Set of normalized identifier tokens.
    """
    text = exp_text.lower().strip()
    identifiers = set()

    date_patterns = re.findall(
        r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b'
        r'|\b\d{4}\b'
        r'|\d{1,2}/\d{4}'
        r'|\d{1,2}-\d{4}',
        text,
    )
    for d in date_patterns:
        identifiers.add(d.strip())

    words = re.findall(r'\b[a-z]{3,}\b', text)
    significant_words = [w for w in words if len(w) >= 4]
    identifiers.update(significant_words[:15])

    return identifiers


def _is_experience_rewrite(tailored_exp: str, original_exp_list: list[str], threshold: float = 0.35) -> bool:
    """Check if a tailored experience entry is a legitimate rewrite of an original.

    Uses multiple strategies:
    1. Direct substring containment (fast path)
    2. SequenceMatcher ratio (fuzzy string similarity)
    3. Identifier overlap (key phrases preserved)

    Args:
        tailored_exp: The tailored experience entry text.
        original_exp_list: List of original experience entry texts.
        threshold: Minimum similarity ratio to consider a match.

    Returns:
        True if the tailored entry matches some original entry.
    """
    t_lower = tailored_exp.lower().strip()

    for orig in original_exp_list:
        o_lower = orig.lower().strip()
        if o_lower in t_lower or t_lower in o_lower:
            return True

    for orig in original_exp_list:
        o_lower = orig.lower().strip()
        ratio = SequenceMatcher(None, o_lower, t_lower).ratio()
        if ratio >= threshold:
            return True

    t_identifiers = _extract_experience_identifiers(tailored_exp)
    for orig in original_exp_list:
        o_identifiers = _extract_experience_identifiers(orig)
        if not o_identifiers:
            continue
        overlap = t_identifiers & o_identifiers
        overlap_ratio = len(overlap) / max(len(o_identifiers), 1)
        if overlap_ratio >= 0.4:
            return True

    return False


def _validate_integrity(result: dict, original_resume: ResumeData) -> list[str]:
    """Validate that no fabricated content was introduced.

    Args:
        result: The tailoring result from the LLM.
        original_resume: Original parsed resume data.

    Returns:
        List of validation errors (empty if all checks pass).
    """
    errors = []
    tailored = result.get("tailored_resume", {})

    if tailored.get("name", "").strip() != (original_resume.name or "").strip():
        errors.append("Name was modified (forbidden)")

    if tailored.get("email", "").strip() != (original_resume.email or "").strip():
        errors.append("Email was modified (forbidden)")

    original_skills = set(s.lower().strip() for s in (original_resume.skills or []))
    new_skills = set(s.lower().strip() for s in (tailored.get("skills") or []))
    added_skills = new_skills - original_skills
    if added_skills:
        errors.append(f"Unsupported skills added: {', '.join(added_skills)}")

    original_exp_list = original_resume.experience or []
    tailored_exp_list = tailored.get("experience") or []

    if len(tailored_exp_list) > len(original_exp_list):
        added_count = len(tailored_exp_list) - len(original_exp_list)
        errors.append(f"Unsupported experience entries added: {added_count} new entries")
    else:
        for t_exp in tailored_exp_list:
            if not _is_experience_rewrite(t_exp, original_exp_list):
                errors.append(f"Unsupported experience entry (not a rewrite of any original): {t_exp[:80]}...")
                break

    original_certs = set(c.lower().strip() for c in (original_resume.certifications or []))
    new_certs = set(c.lower().strip() for c in (tailored.get("certifications") or []))
    added_certs = new_certs - original_certs
    if added_certs:
        errors.append(f"Unsupported certifications added: {', '.join(added_certs)}")

    integrity = result.get("integrity_report", {})
    if integrity.get("employers_added"):
        errors.append("Integrity report indicates employers were added")
    if integrity.get("projects_invented"):
        errors.append("Integrity report indicates projects were invented")
    if integrity.get("certifications_fabricated"):
        errors.append("Integrity report indicates certifications were fabricated")
    if integrity.get("numerical_achievements_invented"):
        errors.append("Integrity report indicates achievements were invented")
    if not integrity.get("resume_factually_accurate", True):
        errors.append("Integrity report flags resume as not factually accurate")

    return errors


class ResumeTailor:
    """Generate a tailored version of the resume for a target job description."""

    def __init__(self, api_key: str, model_name: str = GROQ_MODEL):
        """Initialize the resume tailor.

        Args:
            api_key: Groq API key.
            model_name: Name of the Groq model.
        """
        self.api_key = api_key
        self.model_name = model_name

    def tailor_resume(
        self,
        resume_data: ResumeData,
        job_title: str,
        job_description: str,
        job_skills: str,
        match_analysis: dict,
        ats_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
        recommendations: list[str],
        section_analysis: Optional[dict] = None,
        resume_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Generate tailored resume for a target job.

        Uses caching: if the same (resume_id, job_id) pair was already processed,
        returns the cached result without calling the LLM again.

        Args:
            resume_data: Parsed resume data.
            job_title: Target job title.
            job_description: Job description text.
            job_skills: Required job skills.
            match_analysis: Resume-JD match analysis results.
            ats_score: ATS compatibility score.
            matched_skills: Skills that match the job.
            missing_skills: Skills missing from resume.
            recommendations: Improvement recommendations.
            section_analysis: Optional section-level analysis.
            resume_id: Optional resume identifier for caching.
            job_id: Optional job identifier for caching.

        Returns:
            Dictionary with tailored resume and supporting data or None if failed.
        """
        if resume_id and job_id:
            key = _cache_key(resume_id, job_id)
            if key in _tailor_cache:
                logger.info(f"Returning cached tailoring result for {key}")
                return _tailor_cache[key]

        from groq import Groq

        client = Groq(
            api_key=self.api_key,
            timeout=httpx.Timeout(90.0, read=30.0, connect=10.0),
            max_retries=0,
        )

        resume_text = self._format_resume(resume_data)
        match_analysis_text = self._format_match_analysis(
            match_analysis, ats_score, matched_skills, missing_skills, recommendations
        )

        prompt = RESUME_TAILOR_PROMPT.format(
            resume_text=resume_text,
            job_title=job_title,
            job_description=job_description,
            job_skills=job_skills,
            match_analysis=match_analysis_text,
        )

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.debug(f"Tailoring attempt {attempt}/{MAX_RETRIES}")

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=4000,
                )

                response_text = response.choices[0].message.content
                if not response_text:
                    logger.warning(f"Attempt {attempt}: Empty response from LLM")
                    last_error = "Empty LLM response"
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue

                logger.debug(f"Raw LLM response length: {len(response_text)}")
                result = _parse_llm_json_response(response_text)

                if result is None:
                    logger.warning(f"Attempt {attempt}: Failed to parse JSON from LLM response")
                    last_error = "JSON parse failed"
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue

                if not _validate_result(result):
                    logger.warning(
                        f"Attempt {attempt}: Parsed JSON missing required keys. "
                        f"Got keys: {list(result.keys())}"
                    )
                    last_error = f"Missing required keys: {REQUIRED_RESULT_KEYS - set(result.keys())}"
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue

                integrity_errors = _validate_integrity(result, resume_data)
                if integrity_errors:
                    logger.warning(
                        f"Attempt {attempt}: Integrity validation failed: {integrity_errors}"
                    )
                    last_error = f"Integrity check failed: {'; '.join(integrity_errors)}"
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)
                    continue

                if resume_id and job_id:
                    key = _cache_key(resume_id, job_id)
                    _tailor_cache[key] = result
                    logger.info(f"Cached tailoring result for {key}")

                logger.info("Successfully tailored resume")
                return result

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(f"Attempt {attempt} failed: {last_error}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        logger.error(f"All {MAX_RETRIES} tailoring attempts failed. Last error: {last_error}")
        return None

    def _format_resume(self, resume_data: ResumeData) -> str:
        """Format resume data into structured text for LLM.

        Args:
            resume_data: Structured resume data.

        Returns:
            Formatted resume text.
        """
        parts = []
        if resume_data.name:
            parts.append(f"Name: {resume_data.name}")
        if resume_data.email:
            parts.append(f"Email: {resume_data.email}")
        if resume_data.target_role:
            parts.append(f"Target Role: {resume_data.target_role}")
        if resume_data.summary:
            parts.append(f"Professional Summary: {resume_data.summary}")
        if resume_data.skills:
            parts.append(f"Skills: {', '.join(resume_data.skills)}")
        if resume_data.experience:
            parts.append(f"Experience: {'; '.join(resume_data.experience)}")
        if resume_data.projects:
            parts.append(f"Projects: {'; '.join(resume_data.projects)}")
        if resume_data.education:
            parts.append(f"Education: {'; '.join(resume_data.education)}")
        if resume_data.certifications:
            parts.append(f"Certifications: {'; '.join(resume_data.certifications)}")
        if resume_data.awards:
            parts.append(f"Awards: {'; '.join(resume_data.awards)}")
        if resume_data.languages:
            parts.append(f"Languages: {'; '.join(resume_data.languages)}")
        return "\n".join(parts)

    def _format_match_analysis(
        self,
        match_analysis: dict,
        ats_score: float,
        matched_skills: list[str],
        missing_skills: list[str],
        recommendations: list[str],
    ) -> str:
        """Format match analysis data for LLM context.

        Args:
            match_analysis: Resume-JD match analysis.
            ats_score: ATS score.
            matched_skills: Matched skills.
            missing_skills: Missing skills.
            recommendations: Recommendations.

        Returns:
            Formatted analysis text.
        """
        parts = []
        if match_analysis:
            parts.append(f"Match Analysis: {json.dumps(match_analysis, indent=2)}")
        parts.append(f"ATS Score: {ats_score}/100")
        parts.append(f"Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}")
        parts.append(f"Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}")
        if recommendations:
            parts.append(f"Recommendations: {'; '.join(recommendations)}")
        return "\n".join(parts)


def tailor_resume(
    parsed_resume: ResumeData,
    selected_job: dict,
    match_analysis: dict,
    ats_score: float,
    matched_skills: list[str],
    missing_skills: list[str],
    recommendations: list[str],
    section_analysis: Optional[dict] = None,
    api_key: str = "",
    model_name: str = GROQ_MODEL,
    resume_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> Optional[dict]:
    """Convenience function to tailor a resume for a job.

    Args:
        parsed_resume: Parsed resume data from Module 1.
        selected_job: Selected job from Module 2.
        match_analysis: Resume-JD match analysis.
        ats_score: ATS compatibility score.
        matched_skills: Skills that match the job.
        missing_skills: Skills missing from resume.
        recommendations: Improvement recommendations.
        section_analysis: Optional section-level analysis.
        api_key: Groq API key.
        model_name: Groq model name.
        resume_id: Optional resume identifier for caching.
        job_id: Optional job identifier for caching.

    Returns:
        Tailoring result dictionary or None if failed.
    """
    tailor = ResumeTailor(api_key, model_name)
    return tailor.tailor_resume(
        parsed_resume,
        selected_job.get("title", ""),
        selected_job.get("description", ""),
        selected_job.get("skills", ""),
        match_analysis,
        ats_score,
        matched_skills,
        missing_skills,
        recommendations,
        section_analysis,
        resume_id,
        job_id,
    )
