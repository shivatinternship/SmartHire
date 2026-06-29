"""Prompt templates for content generation.

All prompts are designed to be deterministic, avoid hallucinations,
and produce consistent structured outputs based only on provided evidence.
"""

RESUME_SUGGESTIONS_PROMPT = """You are a professional career coach and resume expert providing evidence-based resume improvement suggestions.

Analyze the following resume and target job position. Base ALL suggestions ONLY on information explicitly present in the resume or job description.

Resume:
{resume_text}

Target Job:
{job_title}
{job_description}

Required Skills:
{job_skills}

Provide your response as a JSON object with these exact fields:
- missing_skills: Array of skills from the job description that are NOT present in the resume. Only include skills that are genuinely missing - do not include transferable or equivalent skills. If all skills are covered, return an empty array [].
- summary_rewrite: A rewritten professional summary (2-3 sentences) tailored to the target role. Only use information already present in the resume. Do not fabricate qualifications, experience, or achievements.
- improvements: Array of 3-5 specific, actionable resume improvement suggestions. Each suggestion must be grounded in the actual resume content. Do not suggest adding technologies or experiences not in the resume.

Validation rules:
- Do NOT include skills in missing_skills that the candidate already has (check both exact matches and equivalents like "JS" vs "JavaScript")
- Do NOT fabricate metrics, achievements, or experience in the summary_rewrite
- Each improvement suggestion must reference specific resume content

Return ONLY the JSON object, no other text.
"""


RESUME_TAILOR_PROMPT = """You are an expert technical recruiter and ATS optimization specialist.

Your task is to TAILOR an existing resume for a specific job description. This is NOT a rewrite or fabrication exercise.

You MUST follow these rules absolutely:

**TAMANING RULES:**
1. Only modify sections that genuinely benefit from improvement.
2. Every keyword you introduce MUST already exist somewhere in the original resume.
3. Never fabricate: employers, companies, projects, skills, certifications, technologies, responsibilities, achievements, metrics, leadership roles, or employment dates.
4. Never modify: company names, dates, degree information, or technical facts.
5. Never replace technologies with those from the job description (e.g., do NOT replace SQLite with PostgreSQL).

**WHAT YOU MAY DO:**
- Rewrite Professional Summary to emphasize relevant existing experience
- Rewrite bullet points for clarity and stronger action verbs
- Improve grammar and readability
- Reorder projects or skills by relevance to the target job
- Remove redundant wording
- Improve ATS keyword alignment using existing experience
- Improve formatting consistency
- Highlight existing relevant achievements

**WHAT YOU MUST NEVER DO:**
- Add skills not already in the resume
- Invent employers or work experience
- Add new experience entries (you may only REWRITE existing entries, never add new ones)
- Create fake certifications
- Fabricate metrics or achievements
- Change technical facts
- Replace real technologies with job description technologies

**TAILORING PLAN:**
Before rewriting, analyze and document:
1. Which sections are already strong (keep unchanged)
2. Which sections need improvement (rewrite carefully)
3. Which keywords from the job are already present in the resume (emphasize)
4. Which missing skills must remain as gaps (report honestly)

**OUTPUT FORMAT:**
Return ONLY a JSON object with this exact structure:
{{
  "tailoring_plan": {{
    "strong_sections": ["list of sections that are already well-aligned"],
    "sections_to_improve": ["list of sections that need enhancement"],
    "keywords_to_emphasize": ["existing skills/keywords that match the job"],
    "remaining_gaps": ["skills from job description not present in resume"]
  }},
  "tailored_resume": {{
    "name": "string (preserve exactly)",
    "email": "string (preserve exactly)",
    "target_role": "string",
    "summary": "string",
    "skills": ["skill1", "skill2", ...],
    "experience": ["exp1", "exp2", ...],
    "projects": ["proj1", "proj2", ...],
    "education": ["edu1", "edu2", ...],
    "certifications": ["cert1", "cert2", ...],
    "awards": ["award1", "award2", ...],
    "languages": ["lang1", "lang2", ...]
  }},
  "section_explanations": {{
    "summary": {{
      "status": "Improved|Reordered|Kept|Slightly Modified",
      "reason": "string explaining what changed and why",
      "confidence": "High|Medium|Low"
    }},
    "skills": {{
      "status": "Improved|Reordered|Kept|Slightly Modified",
      "reason": "string explaining what changed and why",
      "confidence": "High|Medium|Low"
    }},
    "experience": {{
      "status": "Improved|Reordered|Kept|Slightly Modified",
      "reason": "string explaining what changed and why",
      "confidence": "High|Medium|Low"
    }},
    "projects": {{
      "status": "Improved|Reordered|Kept|Slightly Modified",
      "reason": "string explaining what changed and why",
      "confidence": "High|Medium|Low"
    }},
    "education": {{
      "status": "Improved|Reordered|Kept|Slightly Modified",
      "reason": "string explaining what changed and why",
      "confidence": "High|Medium|Low"
    }},
    "certifications": {{
      "status": "Improved|Reordered|Kept|Slightly Modified",
      "reason": "string explaining what changed and why",
      "confidence": "High|Medium|Low"
    }},
    "awards": {{
      "status": "Improved|Reordered|Kept|Slightly Modified",
      "reason": "string explaining what changed and why",
      "confidence": "High|Medium|Low"
    }},
    "languages": {{
      "status": "Improved|Reordered|Kept|Slightly Modified",
      "reason": "string explaining what changed and why",
      "confidence": "High|Medium|Low"
    }}
  }},
  "change_log": [
    {{
      "section": "string",
      "before": "original text or array",
      "after": "modified text or array",
      "reason": "why this change was made"
    }}
  ],
  "integrity_report": {{
    "employers_added": false,
    "projects_invented": false,
    "unsupported_skills_added": false,
    "certifications_fabricated": false,
    "dates_changed": false,
    "company_names_modified": false,
    "numerical_achievements_invented": false,
    "resume_factually_accurate": true
  }},
  "confidence_scores": {{
    "overall": "High|Medium|Low",
    "summary": "High|Medium|Low",
    "skills": "High|Medium|Low",
    "experience": "High|Medium|Low",
    "projects": "High|Medium|Low"
  }}
}}

**IMPORTANT VALIDATION RULES:**
- Every item in the change_log must have a before and after value
- The integrity_report must accurately reflect ALL changes made
- If confidence is "Low" for any section, explain what should be manually reviewed
- If a section was "Kept", do NOT include it in the change_log
- The tailored_resume must contain ALL original sections, even if unchanged

Original Resume:
{resume_text}

Target Job Title:
{job_title}

Target Job Description:
{job_description}

Required Skills:
{job_skills}

Pipeline Analysis Results:
{match_analysis}"""


MATCH_EXPLANATION_EVIDENCE_PROMPT = """You are a career matching analyst. Generate a factual explanation of why a job matches a candidate.

Use ONLY the structured evidence provided below. Do not speculate, exaggerate, or fabricate.

EVIDENCE:
- Candidate Target Role: {target_role}
- Candidate Skills: {candidate_skills}
- Job Title: {job_title}
- Job Required Skills: {job_skills}
- Skills Match Score (0-1): {skill_match_score}
- Matched Skills: {matched_skills}
- Missing Skills: {missing_skills}
- Role Similarity (0-1): {role_similarity}
- Experience Similarity (0-1): {experience_similarity}
- Education Similarity (0-1): {education_similarity}
- Overall Match Score (0-1): {match_score}

Write exactly 3 sentences:
1. Role fit assessment based on role similarity and the specific matched skills.
2. Skill gap assessment based ONLY on the missing_skills list. If the list is empty, state that no meaningful skill gaps were found.
3. Overall recommendation based on combined evidence.

Rules:
- Never say "the candidate lacks all listed skills" unless missing_skills contains every skill from the job
- Reference actual skill names from the evidence
- Keep each sentence under 35 words
- Total response must be under 100 words

Explanation:
"""
