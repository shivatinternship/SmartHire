"""CV improvement suggestion generator."""

import json
import logging
from typing import Optional

from src.generate.prompts import RESUME_SUGGESTIONS_PROMPT
from src.parsing.resume_parser import ResumeData, _parse_llm_json_response
from src.config import GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


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
    ) -> Optional[dict]:
        """Generate improvement suggestions for a resume.

        Args:
            resume_data: Parsed resume data.
            job_title: Target job title.
            job_description: Job description.
            job_skills: Required job skills.

        Returns:
            Dictionary with missing_skills, summary_rewrite, improvements.
        """
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)

            resume_text = self._format_resume(resume_data)
            prompt = RESUME_SUGGESTIONS_PROMPT.format(
                resume_text=resume_text,
                job_title=job_title,
                job_description=job_description,
                job_skills=job_skills,
            )

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )

            response_text = response.choices[0].message.content
            suggestions = _parse_llm_json_response(response_text)

            if suggestions is None:
                return None

            logger.info("Successfully generated CV suggestions")
            return suggestions

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return None

    def _format_resume(self, resume_data: ResumeData) -> str:
        """Format resume data into readable text.

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
        if resume_data.skills:
            parts.append(f"Skills: {', '.join(resume_data.skills)}")
        if resume_data.education:
            parts.append(f"Education: {'; '.join(resume_data.education)}")
        if resume_data.experience:
            parts.append(f"Experience: {'; '.join(resume_data.experience)}")
        return "\n".join(parts)
