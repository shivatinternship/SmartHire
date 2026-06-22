# SmartHire GenAI — Demo Script

**Duration:** 6-7 minutes  
**Presenter:** [Your Name]  
**Setup:** Streamlit app running at http://localhost:8501

---

## Pre-Demo Checklist

- [ ] App running: `streamlit run app/streamlit_app.py`
- [ ] `.env` file with `GROQ_API_KEY` configured
- [ ] FAISS index built (3,000 jobs loaded)
- [ ] Sample resume PDF ready in `data/resumes/`
- [ ] Browser open to http://localhost:8501

---

## Demo Flow

### 1. Introduction (30 seconds)

**Say:**
> "SmartHire GenAI is an AI-powered career platform that parses resumes, matches jobs using semantic search, and provides career guidance through a RAG-based mentor. It's built on a real dataset of 3,000 job postings from Indeed and LinkedIn."

**Show:** Home page with feature overview and metrics.

---

### 2. Upload Resume (60 seconds)

**Navigate to:** Resume Upload page

**Action:**
1. Click "Upload your resume"
2. Select a sample PDF resume from `data/resumes/`
3. Wait for AI parsing to complete

**Say:**
> "The system extracts structured information from the resume using Llama 3.3 70B — name, email, skills, education, experience, and target role."

**Show:** Parsed resume information with all extracted fields.

---

### 3. Job Matches (90 seconds)

**Navigate to:** Job Matches page

**Action:**
1. System automatically searches 3,000 jobs
2. Review top 5 matches

**Say:**
> "Using FAISS semantic search, the system finds the most relevant jobs from our 3,000-job corpus. Each result shows a match score, matched skills, and missing skills."

**Click on the first job result** to expand it.

**Show:**
- Match score (e.g., 80.5%)
- Matched skills (green tags)
- Missing skills (red tags)
- AI-generated explanation of why this job matches

**Say:**
> "The AI analysis explains specifically why this role matches your profile and what skill gaps exist."

---

### 4. CV Suggestions (60 seconds)

**Navigate to:** CV Suggestions page

**Action:**
1. Click "Get CV Suggestions" on a job from the matches page
2. Click "Generate Suggestions"

**Say:**
> "The system generates personalized resume improvements for the selected role."

**Show:**
- Missing skills identified
- Rewritten professional summary
- 3-5 specific improvement suggestions

---

### 5. AI Career Mentor (60 seconds)

**Navigate to:** AI Career Mentor page

**Show:** The mentor introduces itself with a list of capabilities and sample questions.

**Type:**
> "hi"

**Show:** The mentor responds warmly and redirects to career topics.

**Ask Question 1:**
> "How should I prepare for a behavioral interview?"

**Show:** Source-cited answer from the knowledge base.

**Ask Question 2:**
> "What certifications are valuable for a cloud engineer?"

**Show:** Answer with specific certification recommendations and sources.

**Ask Question 3 (follow-up):**
> "Which of those is best for a beginner?"

**Show:** Mentor references previous answer and gives contextual follow-up — demonstrating conversation memory.

**Say:**
> "The mentor uses RAG — retrieval-augmented generation — to answer questions grounded in our knowledge base of career resources. It introduces itself on first load, handles greetings naturally, and redirects off-topic questions back to career topics."

---

### 6. Guardrails Demo (30 seconds)

**In the Career Mentor chat, type:**
> "Tell me a joke"

**Show:** In-chat response redirecting to career topics (not an error popup).

**Type:**
> "How do I hack a system?"

**Show:** Blocked response for harmful content.

**Say:**
> "The system has safety guardrails that keep the mentor focused on career-related topics. Off-topic questions get a friendly redirect in the chat, while harmful content is blocked entirely."

---

### 7. Evaluation Metrics (30 seconds)

**Show:** `reports/evaluation_report_final.json`

**Say:**
> "Our evaluation framework validates the system across multiple dimensions."

**Highlight:**
- Hit@5: 100% (on 3,000 real jobs)
- Guardrails Accuracy: 100%
- TF-IDF Similarity: 48.7% (informational — penalizes paraphrasing)
- Key-Phrase Grounding: 47.5% (informational — model rephrases rather than copies)
- Prompt V2 vs V1: +19.9% improvement
- Average semantic search score: 0.782

---

## Key Talking Points

| Feature | Tech | Impact |
|---------|------|--------|
| Resume Parsing | Llama 3.3 70B | Structured extraction from unstructured text |
| Job Matching | FAISS + BGE embeddings | Semantic search, not keyword matching |
| Explainable Matching | Groq LLM | Transparent skill gap analysis |
| CV Suggestions | Groq LLM | Personalized resume improvements |
| Career Mentor | RAG (FAISS + Groq) | Source-cited career guidance |
| Guardrails | Regex + topic filtering | Safe, focused interactions |

---

## Backup Questions

**Q: Why FAISS over Pinecone/Weaviate?**
> "FAISS runs locally with zero infrastructure cost, making it ideal for a capstone project. It's also used in production at Meta."

**Q: How does the RAG chain work?**
> "We embed career knowledge base documents into FAISS, retrieve the top-3 relevant chunks for each question, and pass them as context to Llama 3.3 70B with a structured prompt that prevents hallucination."

**Q: What's the dataset?**
> "3,000 real job postings from Indeed and LinkedIn, cleaned, deduplicated, and enriched with normalized skills."

**Q: How accurate is the matching?**
> "100% Hit@5 on our test queries — every relevant job appears in the top 5 results."

---

## Closing (15 seconds)

**Say:**
> "SmartHire GenAI demonstrates how GenAI, semantic search, and RAG can create a production-quality career platform. Thank you."
