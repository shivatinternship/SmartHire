"""CV improvement suggestion generator with skill differentiation and caching."""

import logging
from typing import Optional

from src.generate.prompts import RESUME_SUGGESTIONS_PROMPT
from src.parsing.resume_parser import ResumeData, _parse_llm_json_response
from src.config import GROQ_MODEL, LLM_MAX_TOKENS
from src.utils import generate_cache_key, format_resume_text

logger = logging.getLogger(__name__)

_suggestions_cache: dict[str, Optional[dict]] = {}


class CVSuggestionGenerator:
    """Generate resume improvement suggestions using Groq."""

    def __init__(self, api_key: str, model_name: str = GROQ_MODEL):
        """Initialize the CV suggestion generator.

        Args:
            api_key: Groq API key.
            model_name: Name of the Groq model.
        """
        self.api_key = api_key
        self.model_name = model_name

    def generate_suggestions(
        self,
        resume_data: ResumeData,
        job_title: str,
        job_description: str,
        job_skills: str,
        matched_skills: Optional[list[str]] = None,
        missing_skills: Optional[list[str]] = None,
        resume_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Generate improvement suggestions for a resume.

        Uses caching: if the same (resume_id, job_id) pair was already processed,
        returns the cached result.

        Args:
            resume_data: Parsed resume data.
            job_title: Target job title.
            job_description: Job description.
            job_skills: Required job skills.
            matched_skills: Optional pre-computed matched skills.
            missing_skills: Optional pre-computed missing skills.
            resume_id: Optional resume identifier for caching.
            job_id: Optional job identifier for caching.

        Returns:
            Dictionary with missing_skills, summary_rewrite, improvements.
        """
        if resume_id and job_id:
            key = generate_cache_key("cv", resume_id, job_id)
            if key in _suggestions_cache:
                logger.info(f"Returning cached CV suggestions for {key}")
                return _suggestions_cache[key]

        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)

            resume_text = format_resume_text(resume_data)
            prompt = RESUME_SUGGESTIONS_PROMPT.format(
                resume_text=resume_text,
                job_title=job_title,
                job_description=job_description,
                job_skills=job_skills,
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=LLM_MAX_TOKENS,
            )

            response_text = response.choices[0].message.content
            suggestions = _parse_llm_json_response(response_text)

            if suggestions is None:
                return None

            suggestions = self._enrich_with_structured_data(
                suggestions, matched_skills, missing_skills
            )

            if resume_id and job_id:
                key = generate_cache_key("cv", resume_id, job_id)
                _suggestions_cache[key] = suggestions
                logger.info(f"Cached CV suggestions for {key}")

            logger.info("Successfully generated CV suggestions")
            return suggestions

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return None

    def _enrich_with_structured_data(
        self,
        suggestions: dict,
        matched_skills: Optional[list[str]],
        missing_skills: Optional[list[str]],
    ) -> dict:
        """Enrich LLM suggestions with structured skill analysis.

        Differentiates between required, preferred, and transferable skills.
        Prioritizes meaningful skill gaps over exhaustive keyword lists.

        Args:
            suggestions: Raw suggestions from LLM.
            matched_skills: Pre-computed matched skills from job search.
            missing_skills: Pre-computed missing skills from job search.

        Returns:
            Enriched suggestions dict.
        """
        if matched_skills:
            suggestions["matched_skills"] = matched_skills

        if missing_skills:
            suggestions["missing_skills_analysis"] = {
                "required_missing": [],
                "preferred_missing": [],
                "transferable": [],
            }

            llm_missing = suggestions.get("missing_skills", [])
            llm_missing_norm = set(s.lower().strip() for s in llm_missing) if llm_missing else set()
            structured_missing_norm = set(s.lower().strip() for s in missing_skills)

            combined = llm_missing_norm | structured_missing_norm
            suggestions["missing_skills"] = sorted(combined)

            all_skills_text = " ".join(missing_skills).lower()
            preferred_keywords = [
                "preferred", "nice to have", "plus", "bonus", "desired",
                "optional", "familiarity with", "exposure to",
            ]
            for kw in preferred_keywords:
                if kw in all_skills_text:
                    suggestions["missing_skills_analysis"]["preferred_missing"] = list(
                        llm_missing_norm | structured_missing_norm
                    )
                    break
            else:
                suggestions["missing_skills_analysis"]["required_missing"] = list(
                    llm_missing_norm | structured_missing_norm
                )

        return suggestions


