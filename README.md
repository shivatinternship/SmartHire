# SmartHire GenAI — Resume Matching & AI Career Mentor

An AI-powered career platform that analyzes resumes, matches jobs, and provides career guidance using Generative AI.

## Features

- **Resume Parsing**: Extract structured information from PDF/DOCX resumes using LLM
- **Semantic Job Matching**: FAISS-powered vector search to find jobs based on meaning, not keywords
- **Explainable Matching**: Every job match shows matched skills, missing skills, and match score
- **CV Improvement Suggestions**: AI-powered resume rewrite, missing skills, and improvement tips
- **AI Career Mentor**: RAG-based chatbot with introduction message, greeting handling, source-cited answers, and session conversation memory
- **Guardrails**: Input validation with off-topic redirect to keep the mentor focused on career topics
- **Evaluation Framework**: Automated metrics for retrieval, RAG quality, and guardrails accuracy

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
│
├── app/
│   └── streamlit_app.py        # Main Streamlit UI (6 pages)
│
├── src/
│   ├── config.py               # Centralized configuration
│   ├── evaluate.py             # Evaluation framework
│   ├── evaluate_retrieval.py   # Retrieval evaluation on real data
│   │
│   ├── parsing/
│   │   ├── loader.py           # PDF & DOCX text extraction
│   │   └── resume_parser.py    # LLM-based structured resume parsing
│   │
│   ├── search/
│   │   ├── embed.py            # Embedding generation (bge-small)
│   │   ├── job_search.py       # FAISS index + semantic search + skill matching
│   │   ├── preprocess_jobs.py  # Dataset preprocessing pipeline
│   │   └── generate_jobs.py    # Job data generation
│   │
│   ├── generate/
│   │   ├── prompts.py          # All prompt templates
│   │   └── cv_suggestions.py   # CV improvement generator
│   │
│   ├── mentor/
│   │   └── rag_chain.py        # RAG career mentor (FAISS + Groq)
│   │
│   └── safety/
│       └── guardrails.py       # Input validation & safety checks
│
├── data/
│   ├── jobs/
│   │   └── jobs.json           # 3,000 real job listings
│   ├── resumes/                # Upload directory for resumes
│   └── career_notes/           # RAG knowledge base (7 files)
│       ├── career_roadmap.txt
│       ├── resume_guide.txt
│       ├── interview_prep.txt
│       ├── ml_engineer_guide.txt
│       ├── salary_negotiation.txt
│       ├── networking_guide.txt
│       └── technical_interview.txt
│
├── vectorstore/                # FAISS indices (auto-generated)
│   ├── jobs.index              # 4.4 MB, 3,000 vectors
│   └── jobs.json               # Job metadata
│
├── reports/                    # Evaluation reports
│   ├── evaluation_report_final.json
│   ├── evaluation_report.json
│   └── retrieval_evaluation.json
│
└── notebooks/                  # Jupyter exploration notebooks
    ├── 01_embeddings_explore.ipynb
    ├── 02_build_faiss.ipynb
    └── 03_rag_prototype.ipynb
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

1. **Home** — Overview of the platform and metrics
2. **Resume Upload** — Upload a PDF or DOCX resume; AI extracts name, email, skills, education, experience, and target role
3. **Job Matches** — View top-matching jobs with explainable scores (matched/missing skills)
4. **CV Suggestions** — Select a job to get AI-powered improvement tips, rewritten summary, and missing skills
5. **AI Career Mentor** — Ask career questions; the mentor introduces itself, handles greetings, and provides source-cited answers from the knowledge base
6. **Evaluation** — View system evaluation metrics and retrieval test results

## Standout Features

### Explainable Job Matching
Every job result displays:
- **Match Score** — Semantic similarity percentage
- **Matched Skills** — Skills you have that the job requires
- **Missing Skills** — Skills you should develop
- **AI Analysis** — Natural language explanation of why this role matches your profile

### Source-Cited Career Mentor
- **Introduction** — Mentor greets the user and lists available topics with sample questions
- **Greeting Handling** — Responds warmly to greetings, then steers toward career topics
- **Off-Topic Redirect** — Gently redirects non-career questions back to relevant topics
- **Answer** — Grounded response from RAG with source citations

### Safety Guardrails
- 13 blocked patterns for harmful content
- 60 allowed career topics
- 100% accuracy on test suite

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

Reports are saved in the `reports/` directory. Metrics include:

| Metric | Score | Target |
|--------|-------|--------|
| Retrieval Hit@5 | 100% | 80% |
| Retrieval Hit@10 | 100% | 90% |
| Guardrails Accuracy | 100% | 90% |
| RAG Helpfulness | 83.3% | 70% |
| TF-IDF Similarity | 48.7% | Informational |
| Key-Phrase Grounding | 47.5% | Informational |
| Prompt V2 vs V1 | +19.9% | — |

## Screenshots

> Add screenshots of each page here after running the application.

### Home Page
<!-- ![Home Page](screenshots/home.png) -->

### Resume Upload
<!-- ![Resume Upload](screenshots/resume_upload.png) -->

### Job Matches
<!-- ![Job Matches](screenshots/job_matches.png) -->

### CV Suggestions
<!-- ![CV Suggestions](screenshots/cv_suggestions.png) -->

### Career Mentor
<!-- ![Career Mentor](screenshots/career_mentor.png) -->

### Evaluation
<!-- ![Evaluation](screenshots/evaluation.png) -->

## Deployment to Streamlit Cloud

1. Push code to GitHub
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect your repository
4. Set `app/streamlit_app.py` as the main file
5. Add `GROQ_API_KEY` in Streamlit Cloud secrets

## Troubleshooting

### `httpx` / `groq` compatibility error
The project uses `groq>=0.37.1` for compatibility with `httpx==0.27.2`. If you encounter proxy-related errors, ensure your `groq` version is up to date:
```bash
pip install --upgrade groq
```

### `ModuleNotFoundError: No module named 'src'`
The `streamlit_app.py` adds the project root to `sys.path` automatically. If you run into this, ensure you're running from the project root:
```bash
cd smarthire-genai
streamlit run app/streamlit_app.py
```

## License

MIT License
