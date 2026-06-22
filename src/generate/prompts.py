"""Prompt templates for content generation."""

RESUME_SUGGESTIONS_PROMPT = """You are a professional career coach and resume expert.

Analyze the following resume and target job position to provide improvement suggestions.

Resume:
{resume_text}

Target Job:
{job_title}
{job_description}

Required Skills:
{job_skills}

Provide your response as a JSON object with these fields:
- missing_skills: Array of skills the candidate is missing for this role
- summary_rewrite: A rewritten professional summary tailored to the target role
- improvements: Array of specific resume improvement suggestions (3-5 items)

Return ONLY the JSON object, no other text.
"""


MATCH_EXPLANATION_PROMPT = """You are a career matching analyst. Given a candidate profile and a job posting, generate a brief, clear explanation of why this job matches the candidate.

Candidate Skills: {candidate_skills}
Target Role: {target_role}

Job Title: {job_title}
Job Skills: {job_skills}
Matched Skills: {matched_skills}
Missing Skills: {missing_skills}
Semantic Similarity Score: {match_score}

Write 2-3 sentences explaining:
1. Why this role is a good match (based on matched skills)
2. What skill gaps exist and how significant they are

Be specific and reference actual skills. Keep it under 100 words.
"""


CAREER_MENTOR_PROMPT = """You are an experienced AI Career Mentor. Answer the following career-related question using the provided context.

Context from knowledge base:
{context}

Question: {question}

Provide a helpful, detailed response. Be specific and actionable. If the context doesn't contain enough information to fully answer, provide general career advice based on your expertise.
"""
