"""Resume parser using Groq for structured extraction."""

import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from src.config import GROQ_MODEL, RESUME_TEXT_MAX_CHARS
from src.normalization import normalize_skills, deduplicate_skills

logger = logging.getLogger(__name__)


class ResumeData(BaseModel):
    """Structured resume data schema."""

    name: str = Field(default="", description="Full name of the candidate")
    email: str = Field(default="", description="Email address")
    summary: str = Field(default="", description="Professional summary or profile")
    skills: list[str] = Field(default_factory=list, description="List of technical and soft skills")
    education: list[str] = Field(default_factory=list, description="Educational background")
    experience: list[str] = Field(default_factory=list, description="Work experience details")
    projects: list[str] = Field(default_factory=list, description="Personal or academic projects")
    certifications: list[str] = Field(default_factory=list, description="Professional certifications")
    awards: list[str] = Field(default_factory=list, description="Awards and honors")
    languages: list[str] = Field(default_factory=list, description="Languages spoken")
    target_role: str = Field(default="", description="Target job role or career objective")


RESUME_EXTRACTION_PROMPT = """You are a precise resume parser. Extract structured information from the following resume text.

Return ONLY a JSON object with these exact fields (no markdown, no explanation):
{
  "name": "Full name of the candidate (empty string if not found)",
  "email": "Email address (empty string if not found)",
  "skills": ["skill1", "skill2", ...],
  "education": ["degree, institution, dates", ...],
  "experience": ["role, company, dates, description", ...],
  "target_role": "string",
  "summary": "Professional summary (2-3 lines, empty string if not found)",
  "projects": ["project name, technologies, description", ...],
  "certifications": ["certification name, issuer", ...],
  "awards": ["award name", ...],
  "languages": ["language (proficiency)", ...]
}

TARGET ROLE EXTRACTION RULES (critical):
Extract the target role explicitly from the resume in this priority order:
1. Resume headline or objective line (e.g., "Senior Software Engineer" at the top)
2. Most recent job title from experience
3. The most frequently occurring professional title across all experience entries
4. Education specialization if clearly tied to a professional role (e.g., "B.S. in Computer Science" -> "Software Engineer")
5. Professional summary or career objective statement

Only return an empty string "" if absolutely no evidence of a target role exists in the entire resume.
Do NOT fabricate roles. Do NOT return generic labels like "Not Specified" or "Unknown".

SKILL EXTRACTION RULES:
- Extract both technical skills (Python, Java, AWS, etc.) and relevant soft skills
- Be comprehensive but accurate - only extract skills that are explicitly mentioned
- Include skill proficiency levels where explicitly stated (e.g., "Advanced Python")
- Return empty array [] if no skills are found

EXPERIENCE EXTRACTION RULES:
- Return each role as a single string: "Role, Company, Dates - Description"
- Include all roles in chronological order (most recent first)
- Do not truncate or summarize - include the full description

EDUCATION EXTRACTION RULES:
- Return each entry as a single string: "Degree, Institution, Dates"
- Include GPA or honors only if explicitly stated

Resume text:
{text}

Return ONLY the JSON object, no other text. If a field is not found, use an empty string "" or empty array [].
"""


def _parse_llm_json_response(response_text: str) -> Optional[dict]:
    """Parse JSON from an LLM response, stripping markdown code fences.

    Args:
        response_text: Raw LLM response text.

    Returns:
        Parsed dictionary or None if parsing failed.
    """
    if not response_text or not response_text.strip():
        logger.warning("Empty response text, cannot parse JSON")
        return None

    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start:end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError as e:
            logger.debug(f"Brace extraction parse failed: {e}")

        for candidate_end in range(end, start, -1):
            candidate = text[start:candidate_end + 1]
            for suffix in ["}", "}]", "}}"]:
                try:
                    result = json.loads(candidate + suffix)
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue

    if start != -1 and end == -1:
        for candidate_end in range(start + 1, min(start + 5000, len(text))):
            if text[candidate_end] == "}":
                try:
                    result = json.loads(text[start:candidate_end + 1])
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue

    logger.error(f"Failed to parse JSON dict from LLM response: {text[:300]}...")
    return None


def _infer_target_role(data: dict, resume_text: str) -> str:
    """Infer target role from parsed data and raw text if not explicitly found.

    Uses priority-based inference:
    1. Resume headline / first line
    2. Most recent professional title
    3. Most frequent professional title
    4. Education specialization
    5. Professional summary

    Args:
        data: Parsed resume data dict.
        resume_text: Raw resume text for fallback headline extraction.

    Returns:
        Inferred target role or empty string.
    """
    existing = data.get("target_role", "")
    if existing and existing.strip() and existing.strip().lower() not in ("not specified", "unknown", "n/a", ""):
        return existing.strip()

    experience = data.get("experience", [])
    education = data.get("education", [])
    summary = data.get("summary", "")

    roles_found: list[str] = []

    for exp in experience:
        title_match = re.match(r"^([^,]+)", exp)
        if title_match:
            title = title_match.group(1).strip()
            if title:
                roles_found.append(title)

    if roles_found:
        most_recent = roles_found[0]
        return most_recent

    role_words = [
        "engineer", "developer", "scientist", "analyst", "manager",
        "architect", "designer", "consultant", "specialist", "lead",
        "director", "head", "coordinator", "administrator", "associate",
    ]

    lines = resume_text.strip().split("\n")
    for line in lines[:10]:
        line_lower = line.lower().strip()
        for word in role_words:
            if word in line_lower:
                if len(line.strip()) < 100:
                    candidate = line.strip().rstrip(".,")
                    return candidate

    if summary:
        for word in role_words:
            if word in summary.lower():
                summary_line = summary.split(".")[0].strip()
                return summary_line

    if education:
        edu_text = " ".join(education).lower()
        if "computer science" in edu_text:
            return "Software Engineer"
        if "data science" in edu_text or "data" in edu_text:
            return "Data Scientist"
        if "business" in edu_text or "management" in edu_text:
            return "Business Analyst"
        if "electrical" in edu_text or "mechanical" in edu_text or "engineering" in edu_text:
            return "Engineer"

    return ""


def _validate_and_clean_resume_data(data: dict, resume_text: str) -> dict:
    """Validate, clean, and normalize parsed resume data.

    - Removes empty values
    - Deduplicates and normalizes skills
    - Normalizes string fields
    - Infers target role if missing
    - Validates experience and education entries

    Args:
        data: Raw parsed data dict from LLM.
        resume_text: Original resume text for fallback inference.

    Returns:
        Cleaned and validated dict.
    """
    if not isinstance(data, dict):
        logger.warning("_validate_and_clean_resume_data received non-dict input, returning empty dict")
        return {}

    for field in ["name", "email", "summary"]:
        if field in data and isinstance(data[field], str):
            data[field] = data[field].strip()
        elif field not in data:
            data[field] = ""

    if data.get("target_role") and isinstance(data["target_role"], str):
        cleaned = data["target_role"].strip()
        if cleaned.lower() in ("not specified", "unknown", "n/a", "none", ""):
            data["target_role"] = ""
        else:
            data["target_role"] = cleaned

    list_fields = [
        "skills", "education", "experience", "projects",
        "certifications", "awards", "languages",
    ]
    for field in list_fields:
        if field in data and isinstance(data[field], list):
            cleaned = [str(item).strip() for item in data[field] if item and str(item).strip()]
            data[field] = cleaned
        elif field not in data:
            data[field] = []

    if data.get("skills"):
        normalized = normalize_skills(data["skills"])
        data["skills"] = normalized

    for field in ["education", "experience", "projects", "certifications", "awards", "languages"]:
        if data.get(field):
            data[field] = deduplicate_skills(data[field])

    if not data.get("target_role"):
        inferred = _infer_target_role(data, resume_text)
        if inferred:
            logger.info(f"Inferred target role: {inferred}")
            data["target_role"] = inferred

    return data


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

        prompt = RESUME_EXTRACTION_PROMPT.replace("{text}", truncated)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )

        response_text = response.choices[0].message.content
        data = _parse_llm_json_response(response_text)
        if data is None or not isinstance(data, dict):
            return None

        data = _validate_and_clean_resume_data(data, truncated)
        data = _normalize_fields(data)
        data = _validate_resume_data_structure(data)

        logger.debug(f"Data types before ResumeData: name={type(data.get('name')).__name__}, email={type(data.get('email')).__name__}")
        resume_data = ResumeData(**data)

        logger.info(f"Successfully parsed resume for: {resume_data.name}")
        return resume_data

    except Exception as e:
        logger.error(f"Error parsing resume: {type(e).__name__}: {e}")
        logger.debug(f"Response text preview: {response_text[:500] if 'response_text' in locals() else 'N/A'}")
        logger.debug(f"Parsed data: {data if 'data' in locals() else 'N/A'}")
        return None


def _format_education(edu) -> str:
    """Format education entry to string."""
    if isinstance(edu, str):
        return edu
    if isinstance(edu, dict):
        parts = []
        if edu.get("degree"):
            parts.append(edu["degree"])
        if edu.get("school") or edu.get("institution"):
            parts.append(edu.get("school") or edu.get("institution"))
        if edu.get("dates"):
            parts.append(edu["dates"])
        return ", ".join(parts) if parts else str(edu)
    return str(edu)


def _format_experience(exp) -> str:
    """Format experience entry to string."""
    if isinstance(exp, str):
        return exp
    if isinstance(exp, dict):
        parts = []
        if exp.get("role") or exp.get("title"):
            parts.append(exp.get("role") or exp.get("title"))
        if exp.get("company"):
            parts.append(exp["company"])
        if exp.get("dates"):
            parts.append(exp["dates"])
        if exp.get("description"):
            parts.append(exp["description"])
        return ", ".join(parts) if parts else str(exp)
    return str(exp)


def _normalize_fields(data: dict) -> dict:
    """Normalize LLM output so education/experience are string arrays."""
    if "education" in data and isinstance(data["education"], list):
        data["education"] = [_format_education(e) for e in data["education"]]
    if "experience" in data and isinstance(data["experience"], list):
        data["experience"] = [_format_experience(e) for e in data["experience"]]
    if "skills" in data and isinstance(data["skills"], list):
        data["skills"] = [str(s) for s in data["skills"]]
    if "projects" in data and isinstance(data["projects"], list):
        data["projects"] = [str(p) for p in data["projects"]]
    if "certifications" in data and isinstance(data["certifications"], list):
        data["certifications"] = [str(c) for c in data["certifications"]]
    if "awards" in data and isinstance(data["awards"], list):
        data["awards"] = [str(a) for a in data["awards"]]
    if "languages" in data and isinstance(data["languages"], list):
        data["languages"] = [str(l) for l in data["languages"]]
    if "summary" in data and isinstance(data["summary"], str):
        data["summary"] = data["summary"].strip()
    return data


def _validate_resume_data_structure(data: dict) -> dict:
    """Validate and coerce resume data to match ResumeData schema before Pydantic validation."""
    string_fields = ["name", "email", "summary", "target_role"]
    list_fields = ["skills", "education", "experience", "projects", "certifications", "awards", "languages"]

    for field in string_fields:
        val = data.get(field)
        if val is None or val is False:
            data[field] = ""
        elif not isinstance(val, str):
            logger.warning(f"Field '{field}' coerced from {type(val).__name__} to str: {val}")
            data[field] = str(val)
        else:
            data[field] = val.strip()

    for field in list_fields:
        val = data.get(field)
        if val is None or val is False:
            data[field] = []
        elif not isinstance(val, list):
            logger.warning(f"Field '{field}' coerced from {type(val).__name__} to list: {val}")
            if isinstance(val, str):
                data[field] = [val] if val.strip() else []
            else:
                data[field] = []
        else:
            data[field] = [str(item).strip() for item in val if item is not None and str(item).strip()]

    return data
