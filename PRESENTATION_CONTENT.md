# SmartHire GenAI — Presentation Content

---

## 1. Problem Statement

Job seekers face three core challenges:

1. **Resume Optimization**: Tailoring resumes for specific roles without knowing what employers prioritize
2. **Job Discovery**: Finding relevant positions among thousands of postings with inconsistent formatting
3. **Career Guidance**: Getting personalized, actionable advice without expensive coaching

Traditional job platforms rely on keyword matching, missing semantic relationships between skills and roles. A "Python developer" and a "machine learning engineer" share significant skill overlap, but keyword search treats them as unrelated.

**SmartHire GenAI** solves this with AI-powered semantic matching, explainable results, and a RAG-based career mentor.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │  Resume   │ │   Job    │ │   CV     │ │  Career    │  │
│  │  Upload   │ │ Matches  │ │ Suggest  │ │  Mentor    │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬──────┘  │
└───────┼────────────┼────────────┼──────────────┼─────────┘
        │            │            │              │
        ▼            ▼            ▼              ▼
┌─────────────────────────────────────────────────────────┐
│                     Backend Pipeline                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Resume   │ │  FAISS   │ │  Groq    │ │  RAG Chain │  │
│  │ Parser   │ │  Search  │ │  LLM     │ │  (FAISS +  │  │
│  │ (Groq)   │ │  Engine  │ │          │ │   Groq)    │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└─────────────────────────────────────────────────────────┘
        │            │            │              │
        ▼            ▼            ▼              ▼
┌─────────────────────────────────────────────────────────┐
│                      Data Layer                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐  │
│  │ Resume   │ │  Jobs    │ │ FAISS    │ │ Career     │  │
│  │ Text     │ │  JSON    │ │ Index    │ │ Knowledge  │  │
│  │ (PDF/    │ │ (3,000)  │ │ (3,000)  │ │ Base (7)   │  │
│  │  DOCX)   │ │          │ │ vectors  │ │            │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit 1.39 | Interactive web UI with 5 pages |
| **LLM** | Groq — Llama 3.3 70B Versatile | Resume parsing, CV suggestions, match explanations |
| **Embeddings** | BAAI/bge-small-en-v1.5 | 384-dim semantic embeddings |
| **Vector DB** | FAISS (IndexFlatIP) | Local vector storage and similarity search |
| **RAG** | LangChain 0.3 + Groq | Document retrieval and context-aware generation |
| **Parsing** | PyMuPDF + python-docx | PDF and DOCX text extraction |
| **Validation** | Pydantic 2.9 | Structured data schemas |
| **Safety** | Custom regex + topic filtering | Input validation and guardrails |

---

## 4. Dataset

### Source
- **Indeed Job Listings**: 967 jobs (Bright Data API scrape)
- **LinkedIn Job Postings**: 2,033 jobs (public dataset)
- **Total**: 3,000 unique jobs after deduplication

### Statistics
| Metric | Value |
|--------|-------|
| Total jobs | 3,000 |
| Unique job titles | 1,916 |
| Unique skills | 187 |
| Jobs with skills | 99% |
| Jobs with descriptions | 100% |
| FAISS index size | 4.4 MB |
| Embedding dimension | 384 |

### Preprocessing Pipeline
1. Load raw CSVs (26,114 rows)
2. Column mapping to unified schema
3. Text cleaning (HTML, whitespace, encoding)
4. Skills normalization (200+ synonym mappings)
5. Deduplication (title + company)
6. Cap at 3,000 jobs
7. Generate embeddings with BGE-small

---

## 5. Workflow

### Resume Processing
```
PDF/DOCX → Text Extraction → LLM Parsing → Structured ResumeData
                                              ├── name
                                              ├── email
                                              ├── skills[]
                                              ├── education[]
                                              ├── experience[]
                                              └── target_role
```

### Job Matching
```
ResumeData → Query Construction → FAISS Search → Top-5 Jobs
                                                  ├── match_score
                                                  ├── matched_skills
                                                  └── missing_skills
```

### Career Mentor (RAG)
```
User Question → Safety Check → FAISS Retrieval (top-3)
                                    ↓
                    Context + Question + Chat History → Groq LLM → Answer
                                    ↓
                              Source Citations
```

**Conversation Memory**: Session-only (5-turn window), excludes greetings from LLM context, uses `RunnableWithMessageHistory` wrapping the RAG chain.

---

## 6. Evaluation Results

### Retrieval Quality
| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Hit@5 | 100% | 80% | PASS |
| Hit@10 | 100% | 90% | PASS |
| Avg Top-1 Score | 0.782 | 0.70 | PASS |

### System Quality
| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Guardrails Accuracy | 100% | 90% | PASS |
| Prompt V2 Quality | 80% | 70% | PASS |
| RAG Helpfulness | 83.3% | 70% | PASS |
| TF-IDF Similarity | 48.7% | Informational | INFO |
| Key-Phrase Grounding | 47.5% | Informational | INFO |

### Retrieval Test Queries
| Query | Top Match | Score |
|-------|-----------|-------|
| Python developer with ML | Machine Learning Software Engineer | 0.805 |
| Data analyst SQL Tableau | Data Analyst | 0.787 |
| Data engineer ETL Spark | Data Engineer | 0.847 |
| ML engineer TensorFlow | ML Engineer (TensorFlow) | 0.837 |
| Product manager Agile | Product Manager | 0.800 |

---

## 7. Key Innovations

### 1. Semantic Job Matching
Unlike keyword-based platforms, SmartHire uses FAISS inner-product similarity on 384-dim BGE embeddings. This means "React developer" matches "Front-End Software Engineer" even without exact keyword overlap.

### 2. Explainable Matching
Every job result includes:
- Match score (semantic similarity percentage)
- Matched skills (green tags)
- Missing skills (red tags)
- AI-generated explanation of why the role matches

### 3. Grounded Career Mentor (RAG)
The career mentor uses retrieval-augmented generation:
- 7 knowledge base files covering resumes, interviews, salary negotiation
- FAISS retrieval of top-3 relevant chunks
- Structured prompt with hallucination prevention
- Source citations for every answer
- Introduction message with sample questions on first load
- Greeting handling and off-topic redirect

### 4. Safety Guardrails
- 13 blocked patterns (harmful content)
- 60 allowed career topics
- Input validation for resume text
- 100% accuracy on test suite

---

## 8. Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| LLM hallucination | Structured prompts with explicit "answer only from context" instruction |
| Skill matching false positives | Normalized substring matching with minimum length threshold |
| FAISS cold start | Streamlit caching with `@st.cache_resource` |
| Large embedding computation | Batch processing with BGE-small (fast, 384-dim) |
| Resume format diversity | PyMuPDF (PDF) + python-docx (DOCX) with fallback extraction |
| API rate limits | Groq free tier with optimized prompt token usage |

---

## 9. Future Work

1. **Multi-format support**: Add HTML, plain text resume parsing
2. **Persistent sessions**: Database-backed user profiles
3. **Knowledge base expansion**: Automated career content ingestion
4. **Real-time job feeds**: Integration with job board APIs
5. **ATS compatibility scoring**: How well a resume passes ATS filters
6. **Interview simulation**: AI-powered mock interview practice
7. **Skill gap learning paths**: Curated courses for missing skills
8. **Multi-language support**: Resume parsing in non-English languages

---

## 10. Metrics Summary

| Category | Metric | Score |
|----------|--------|-------|
| **Retrieval** | Hit@5 | 100% |
| **Retrieval** | Hit@10 | 100% |
| **Retrieval** | Avg Score | 0.782 |
| **RAG** | Helpfulness | 83.3% |
| **RAG** | Key-Phrase Grounding | 47.5%* |
| **RAG** | TF-IDF Similarity | 48.7%* |
| **Safety** | Guardrails | 100% |
| **Prompts** | V2 vs V1 | +19.9% |

*These are informational metrics that penalize paraphrasing. Actual answer quality is higher — the RAG chain uses Llama 3.3 70B with structured prompts that produce detailed, contextual answers.
