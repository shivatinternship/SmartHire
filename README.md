# SmartHire GenAI — Resume Matching & AI Career Mentor

An AI-powered career platform that analyzes resumes, matches jobs, and provides career guidance using Generative AI. Built with LangChain, Groq, and FAISS on a real-world dataset of 3,000 job postings from Indeed and LinkedIn.

## Live Demo

- Streamlit Application: <DEPLOYED_STREAMLIT_URL>
- GitHub Repository: <GITHUB_REPOSITORY_URL>

## Features

- **Resume Parsing**: Extract structured information from PDF/DOCX resumes using Llama 3.3 70B via Groq API. Documents are loaded, cleaned, and chunked before embedding generation to improve retrieval quality and semantic search performance.
- **Semantic Job Matching**: FAISS-powered vector search to find jobs based on meaning, not keywords
- **Explainable Matching**: Every job match shows matched skills, missing skills, match score, and AI-generated analysis
- **CV Improvement Suggestions**: AI-powered resume rewrite, missing skills identification, and improvement tips
- **AI Career Mentor**: RAG-based chatbot with introduction message, greeting handling, source-cited answers, and session conversation memory. Documents are loaded, cleaned, and chunked before embedding generation to improve retrieval quality and semantic search performance.
- **Safety Guardrails**: Input validation with off-topic redirect to keep the mentor focused on career topics
- **Evaluation Framework**: Automated metrics for retrieval, RAG quality, and guardrails accuracy

### Document Processing Pipeline

1. Upload PDF/DOCX document
2. Extract text using PyMuPDF or python-docx
3. Clean and preprocess content
4. Split documents into chunks
5. Generate embeddings
6. Store vectors in FAISS
7. Retrieve relevant chunks during search or RAG

## Dataset

| Metric | Value |
|--------|-------|
| Total jobs | 3,000 |
| Source | Indeed + LinkedIn (Kaggle) |
| Unique titles | 1,916 |
| Unique skills | 187 |
| FAISS index size | 4.4 MB |
| Embedding model | BAAI/bge-small-en-v1.5 (384-dim) |

See `DATASET_REPORT.md` for full dataset details.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit 1.39 |
| LLM | Groq — Llama 3.3 70B Versatile |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector DB | FAISS (IndexFlatIP) |
| RAG Framework | LangChain + Groq client |
| Resume Parsing | PyMuPDF, python-docx |
| Validation | Pydantic 2.9 |
| Env Variables | python-dotenv |

## Generative AI Techniques Used

This project demonstrates the core Generative AI techniques covered in the course:

- Structured Output — LLM extracts resume information into validated JSON format
- Prompt Engineering — Custom prompts for resume parsing, CV improvement, and job-match explanations
- LLM APIs — Groq-hosted Llama 3.3 70B used for resume analysis and content generation
- Embeddings — BAAI/bge-small-en-v1.5 converts jobs and career documents into vector representations
- Semantic Search — Candidate profiles are matched to jobs based on meaning rather than keywords
- Vector Database (FAISS) — Stores and retrieves embeddings efficiently for similarity search
- Retrieval-Augmented Generation (RAG) — Career Mentor retrieves relevant documents before generating answers
- LangChain Orchestration — Coordinates retrieval and generation workflows
- Safety Guardrails — Input validation and off-topic filtering before LLM execution
- Evaluation Framework — Measures retrieval quality, grounding, helpfulness, and guardrail performance

## Project Structure

```
smarthire-genai/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── DEMO_SCRIPT.md              # 6-7 minute demo walkthrough
├── PRESENTATION_CONTENT.md     # Presentation-ready content
├── DATASET_REPORT.md           # Full dataset documentation
├── PROJECT_AUDIT.md            # Project audit report
│
├── app/
│   └── streamlit_app.py        # Main Streamlit UI (6 pages)
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized configuration
│   ├── evaluate.py             # Evaluation framework
│   ├── evaluate_retrieval.py   # Retrieval evaluation on real data
│   │
│   ├── parsing/
│   │   ├── __init__.py
│   │   ├── loader.py           # PDF & DOCX text extraction
│   │   └── resume_parser.py    # LLM-based structured resume parsing
│   │
│   ├── search/
│   │   ├── __init__.py
│   │   ├── embed.py            # Embedding generation (bge-small)
│   │   ├── job_search.py       # FAISS index + semantic search + skill matching
│   │   ├── preprocess_jobs.py  # Dataset preprocessing pipeline
│   │   └── generate_jobs.py    # Job data generation
│   │
│   ├── generate/
│   │   ├── __init__.py
│   │   ├── prompts.py          # All prompt templates
│   │   └── cv_suggestions.py   # CV improvement generator
│   │
│   ├── mentor/
│   │   ├── __init__.py
│   │   └── rag_chain.py        # RAG career mentor (FAISS + Groq)
│   │
│   └── safety/
│       ├── __init__.py
│       └── guardrails.py       # Input validation & safety checks
│
├── data/
│   ├── jobs/
│   │   ├── jobs.json           # 3,000 real job listings
│   │   ├── indeed_raw.csv      # Raw Indeed dataset
│   │   └── linkedin_raw.csv    # Raw LinkedIn dataset
│   ├── resumes/                # Upload directory for resumes
│   └── career_notes/           # RAG knowledge base (25 files)
│       ├── career_roadmap.txt
│       ├── resume_guide.txt
│       ├── interview_prep.txt
│       ├── behavioral_interview_guide.txt
│       ├── salary_negotiation.txt
│       ├── salary_negotiation_advanced.txt
│       ├── networking_guide.txt
│       ├── technical_interview.txt
│       ├── job_search_strategy.txt
│       ├── portfolio_guide.txt
│       ├── internship_guide.txt
│       ├── ats_resume_guide.txt
│       ├── ai_agent_guide.txt
│       ├── rag_guide.txt
│       ├── prompt_engineering_guide.txt
│       ├── machine_learning_engineer_roadmap.txt
│       ├── ml_engineer_guide.txt
│       ├── data_scientist_roadmap.txt
│       ├── data_analyst_roadmap.txt
│       ├── backend_engineer_roadmap.txt
│       ├── frontend_engineer_roadmap.txt
│       ├── fullstack_engineer_roadmap.txt
│       ├── devops_engineer_roadmap.txt
│       ├── cloud_engineer_roadmap.txt
│       └── genai_engineer_roadmap.txt
│
├── vectorstore/                # FAISS indices (auto-generated)
│   ├── jobs.index              # 4.4 MB, 3,000 vectors
│   └── jobs.json               # Job metadata
│
└── reports/                    # Evaluation reports
    ├── evaluation_report_final.json
    ├── evaluation_report.json
    └── retrieval_evaluation.json
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd smarthire-genai
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

Get your free API key from: https://console.groq.com

### 4. Run the Application

```bash
streamlit run app/streamlit_app.py
```

Opens at: http://localhost:8501

## Usage Guide

### Navigation Pages

1. **Home** — Overview of the platform with metrics and feature highlights
2. **Resume Upload** — Upload a PDF or DOCX resume; AI extracts name, email, skills, education, experience, and target role. After successful parsing, click "Go to Job Matches" to navigate directly to matching jobs
3. **Job Matches** — View top-matching jobs with explainable scores (matched/missing skills and AI analysis). Click "Get CV Suggestions for this role" to get personalized improvement tips
4. **CV Suggestions** — Select a job from Job Matches to get AI-powered improvement suggestions, a rewritten professional summary, and missing skills
5. **AI Career Mentor** — Ask career questions and get expert advice powered by RAG with source citations
6. **Evaluation** — View system evaluation metrics and retrieval test results

### Workflow

1. **Upload Resume** → AI parses and extracts structured information
2. **View Job Matches** → See top 5 matching jobs with match scores and skill analysis
3. **Get AI Analysis** → Each job match includes an AI-generated explanation of why the role fits
4. **Improve Resume** → Click on any job to get specific improvement suggestions
5. **Ask Mentor** → Get career advice from the AI mentor with source-cited answers

## Standout Features

### Explainable Job Matching
Every job result displays:
- **Match Score** — Semantic similarity percentage (0-100%)
- **Matched Skills** — Skills you have that the job requires (green tags)
- **Missing Skills** — Skills you should develop (red tags)
- **AI Analysis** — Natural language explanation of why this role matches your profile

### Source-Cited Career Mentor
- **Introduction** — Mentor greets the user and lists available topics with sample questions
- **Greeting Handling** — Responds warmly to greetings, then steers toward career topics
- **Off-Topic Redirect** — Gently redirects non-career questions back to relevant topics
- **Conversation Memory** — Maintains chat history within the session
- **Source Citations** — Answers include references to knowledge base documents

### Safety Guardrails
- 13 blocked patterns for harmful content (violence, illegal activities, etc.)
- 60+ allowed career topics with keyword detection
- 100% accuracy on test suite (harmful, off-topic, and career queries)
- Automatic redirect for non-career questions

### Knowledge Base
- 25 career guidance documents covering resumes, interviews, career roadmaps, and more
- Covers roles: Data Scientist, ML Engineer, Backend/Frontend/Fullstack Developer, DevOps, Cloud Engineer, Data Analyst, GenAI Engineer
- Topics: ATS optimization, salary negotiation, behavioral interviews, technical interviews, networking, portfolio building, and more


## System Architecture

Resume Upload (PDF/DOCX)
        │
        ▼
Document Loading & Chunking
        │
        ▼
LLM Resume Parser
        │
        ▼
Structured Candidate Profile
        │
        ├──► FAISS Semantic Job Search
        │           │
        │           ▼
        │      Top Job Matches
        │
        ├──► CV Improvement Generator
        │           │
        │           ▼
        │      Resume Suggestions
        │
        └──► AI Career Mentor (RAG)
                    │
                    ▼
            Career Guidance

Career Notes + Knowledge Base
                    │
                    ▼
                 FAISS
                    │
                    ▼
              LangChain RAG

## Evaluation

### Run Retrieval Evaluation

```bash
python -m src.evaluate_retrieval
```

Tests 10 domain-specific queries against 3,000 real jobs.

### Run Full Evaluation

```bash
python -m src.evaluate
```

Reports are saved in the `reports/` directory.

### Written Evaluation Report

See `FINAL_REPORT.md` for the comprehensive final evaluation report covering all evaluation requirements.

### System Metrics

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Retrieval Hit@5 | 100% | 80% | PASS |
| Retrieval Hit@10 | 100% | 90% | PASS |
| Guardrails Accuracy | 100% | 90% | PASS |
| RAG Helpfulness | 83.3% | 70% | PASS |
| Semantic Search Quality | 78.2% | 70% | PASS |
| Prompt V2 vs V1 | +19.9% | — | PASS |
| RAG Correctness (TF-IDF) | 48.7% | — | Informational* |
| RAG Groundedness (Key-Phrase) | 47.5% | — | Informational* |

*\* Informational: These metrics use heuristic word-overlap evaluation that penalizes paraphrasing. Actual quality is higher.*

### Retrieval Test Results

| Query | Top Match | Score |
|-------|-----------|-------|
| Python developer with ML experience | Machine Learning Software Engineer | 0.805 |
| Data analyst SQL Tableau | Data Analyst | 0.787 |
| DevOps engineer Docker Kubernetes | Senior Cloud Infrastructure/Devops Engineer | 0.768 |
| Frontend developer React JavaScript | Front-End Software Engineer, React | 0.759 |
| Product manager Agile Scrum | Product Manager | 0.800 |
| Security engineer cybersecurity | Software Engineer, Cyber Vulnerability Researcher | 0.757 |
| Data engineer ETL pipeline Spark | Data Engineer | 0.847 |
| UX designer Figma user research | User Experience Researcher | 0.688 |
| ML engineer deep learning TensorFlow | Machine Learning Engineer (Tensor Flow) | 0.837 |
| Java backend Spring Boot microservices | Remote Spring Boot Developer | 0.732 |

## Deployment to Streamlit Cloud

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect your repository
4. Set `app/streamlit_app.py` as the main file
5. Add `GROQ_API_KEY` in Streamlit Cloud secrets

## Troubleshooting

### `httpx` / `groq` compatibility error
The project uses `groq==0.37.1` for compatibility with `httpx==0.27.2`. If you encounter proxy-related errors, ensure your `groq` version is up to date:
```bash
pip install --upgrade groq
```

### `ModuleNotFoundError: No module named 'src'`
The `streamlit_app.py` adds the project root to `sys.path` automatically. If you run into this, ensure you're running from the project root:
```bash
cd smarthire-genai
streamlit run app/streamlit_app.py
```

### `FAISS index not found`
The FAISS index is built automatically on first run if the `vectorstore/jobs.index` file doesn't exist. This takes approximately 77 seconds for 3,000 jobs.

### `GROQ_API_KEY is not configured`
Create a `.env` file in the project root with your Groq API key:
```
GROQ_API_KEY=your_api_key_here
```

## License

MIT License
