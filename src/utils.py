"""Shared utility functions for SmartHire GenAI.

Consolidates duplicate implementations across modules.
"""

import re
from typing import Optional

from src.parsing.resume_parser import ResumeData


def format_resume_text(resume_data: ResumeData) -> str:
    """Format resume data into readable text for LLM prompts.

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


def generate_cache_key(prefix: str, resume_id: str, job_id: str) -> str:
    """Generate a cache key for CV suggestions or tailoring results.

    Args:
        prefix: Cache namespace (e.g., "cv" or "tailor").
        resume_id: Resume identifier.
        job_id: Job identifier.

    Returns:
        Cache key string.
    """
    return f"{prefix}|{resume_id}|{job_id}"


GREETING_WORDS = frozenset({
    "hi", "hello", "hey", "hola", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good day",
    "what's up", "whats up", "greetings",
})

GREETING_RESPONSE = (
    "Hello! Great to meet you. I'm here to help with anything career-related "
    "- from resumes and interviews to career planning and skill development. "
    "What would you like to explore?"
)


def is_greeting(text: str) -> bool:
    """Check if text is a simple greeting.

    Args:
        text: Input text to check.

    Returns:
        True if text is a greeting.
    """
    if not text:
        return False
    return text.strip().lower().rstrip("!.") in GREETING_WORDS


def build_skill_tags_html(skills: list[str], css_class: str = "matched-skill") -> str:
    """Build HTML for skill tags.

    Args:
        skills: List of skill names.
        css_class: CSS class for the skill tags.

    Returns:
        HTML string with skill tags.
    """
    return " ".join([
        f'<span class="skill-tag {css_class}">{s}</span>'
        for s in skills
    ])
