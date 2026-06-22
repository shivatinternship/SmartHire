"""Input validation and safety guardrails."""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

BLOCKED_PATTERNS = [
    r"\bhack\b(?!athon)",
    r"crack\s*password",
    r"\billegal\b",
    r"\bviolence\b",
    r"\bbomb\b",
    r"\bweapon\b",
    r"drug\s*deal",
    r"\bsteal\b",
    r"\bmurder\b",
    r"\bmalware\b",
    r"\bphish\b",
    r"\bdox(ing|ed)?\b",
    r"\bswat(ting)?\b",
    r"\bporn\b",
    r"\bsex\b",
    r"\bgambling\b",
    r"\bterroris\b",
    r"\bself.harm\b",
    r"\bsuicide\b",
]

CAREER_HINTS = [
    r"\b(resume|cv|career|job|interview|skill|education|experience|salary|negotiat)\w*",
    r"\b(portfolio|linkedin|network|mentor|coach|professional|work|hiring|recruit)\w*",
    r"\b(apply|application|cover.letter|reference|certification|training|course|degree)\w*",
    r"\b(university|college|internship|freelance|remote|onsite|technical|leadership)\w*",
    r"\b(communication|team|project|achievement|promotion|resign|quit|fired|laid.off)\w*",
    r"\b(onboarding|mentorship|networking|github|leetcode|system.design|behavioral)\w*",
    r"\b(headhunter|recruiter|offer.letter|compensation|benefits|equity|stock.option)\w*",
    r"\b(401k|insurance|vacation|sick.leave|work.life|burnout|imposter|career.change)\w*",
    r"\b(upskill|reskill|layoff|data|analyst|analytics|science|scientist|machine.learning)\w*",
    r"\b(ml|ai|artificial.intelligence|deep.learning|nlp|computer.vision|sql|python)\w*",
    r"\b(tableau|power.bi|excel|statistics|modeling|pipeline|etl|dashboard|visualization)\w*",
    r"\b(business.intelligence|bi|big.data|spark|hadoop|cloud|aws|azure|gcp|devops)\w*",
    r"\b(engineering|developer|software|frontend|backend|fullstack|mobile|web|api)\w*",
    r"\b(database|architecture|design|product|management|agile|scrum|qa|testing)\w*",
    r"\b(security|cybersecurity|blockchain|crypto|fintech|healthtech|edtech|saas)\w*",
    r"\b(startup|entrepreneur|founder|cto|vp|director|manager|lead|senior|junior|entry|level)\w*",
    r"\b(transition|switch|pivot|grow|advance|develop|learn|study|prepare|plan|goal)\w*",
    r"\b(weakness|strength|improve|build|create|write|review|feedback|tip|advice|guide)\w*",
    r"\b(how.to|what.is|best.way|strategy|approach|method|technique|resource|tool)\w*",
]


def check_input_safety(text: str) -> dict:
    """Check if input text is safe.

    Uses blocklist for harmful content. Allows by default - the LLM's prompt
    handles career-topic relevance naturally.

    Args:
        text: User input text to validate.

    Returns:
        Dictionary with 'allowed' boolean and 'reason' string.
    """
    if not text or not text.strip():
        return {"allowed": False, "reason": "Empty input provided."}

    text_lower = text.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"Blocked harmful content: {text[:50]}...")
            return {
                "allowed": False,
                "reason": "This query contains content outside the scope of a career platform.",
            }

    has_career_hint = any(re.search(pattern, text_lower) for pattern in CAREER_HINTS)
    if not has_career_hint:
        logger.info(f"Non-career query allowed (LLM will redirect): {text[:50]}...")

    return {"allowed": True, "reason": ""}


def validate_resume_text(text: str) -> dict:
    """Validate that extracted text looks like a resume.

    Args:
        text: Extracted text from a document.

    Returns:
        Dictionary with 'valid' boolean and 'reason' string.
    """
    if not text or len(text.strip()) < 50:
        return {
            "valid": False,
            "reason": "The document appears to be too short or empty to be a valid resume.",
        }

    resume_indicators = [
        "experience",
        "education",
        "skills",
        "work",
        "university",
        "college",
        "degree",
        "bachelor",
        "master",
        "proficiency",
        "qualification",
        "employment",
        "career",
        "objective",
        "summary",
    ]

    text_lower = text.lower()
    found = sum(1 for ind in resume_indicators if ind in text_lower)

    if found < 2:
        return {
            "valid": False,
            "reason": "The document does not appear to contain typical resume content.",
        }

    return {"valid": True, "reason": ""}
