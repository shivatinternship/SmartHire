"""Resume parser using Groq for structured extraction."""

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.config import GROQ_MODEL, LLM_TEMPERATURE, RESUME_TEXT_MAX_CHARS

logger = logging.getLogger(__name__)


class ResumeData(BaseModel):
    """Structured resume data schema."""

    name: str = Field(default="", description="Full name of the candidate")
    email: str = Field(default="", description="Email address")
    skills: list[str] = Field(default_factory=list, description="List of technical and soft skills")
    education: list[str] = Field(default_factory=list, description="Educational background")
    experience: list[str] = Field(default_factory=list, description="Work experience details")
    target_role: str = Field(default="", description="Target job role or career objective")


RESUME_EXTRACTION_PROMPT = """Extract structured information from the following resume text.

Return a JSON object with these fields:
- name: Full name of the candidate
- email: Email address
- skills: Array of technical and soft skills mentioned
- education: Array of educational qualifications
- experience: Array of work experience entries
- target_role: Target job role or career objective if mentioned

Resume text:
{text}

Return ONLY the JSON object, no other text. If a field is not found, use an empty string or empty array.
"""


def _parse_llm_json_response(response_text: str) -> Optional[dict]:
    """Parse JSON from an LLM response, stripping markdown code fences.

    Args:
        response_text: Raw LLM response text.

    Returns:
        Parsed dictionary or None if parsing failed.
    """
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from LLM response: {text[:200]}...")
        return None


def parse_resume(
    text: str,
    api_key: str,
    model_name: str = GROQ_MODEL,
) -> Optional[ResumeData]:
    """Parse resume text and extract structured data using Groq.

    Args:
        text: Raw resume text content.
        api_key: Groq API key.
        model_name: Name of the Groq model to use.

    Returns:
        Structured ResumeData or None if parsing failed.
    """
    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        truncated = text[:RESUME_TEXT_MAX_CHARS]
        if len(text) > RESUME_TEXT_MAX_CHARS:
            logger.warning(
                f"Resume text truncated from {len(text)} to {RESUME_TEXT_MAX_CHARS} chars"
            )

        prompt = RESUME_EXTRACTION_PROMPT.format(text=truncated)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=2000,
        )

        response_text = response.choices[0].message.content
        data = _parse_llm_json_response(response_text)
        if data is None:
            return None

        resume_data = ResumeData(**data)

        logger.info(f"Successfully parsed resume for: {resume_data.name}")
        return resume_data

    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        return None
