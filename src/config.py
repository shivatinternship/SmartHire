"""Configuration settings for SmartHire GenAI."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
RESUMES_DIR = DATA_DIR / "resumes"
CAREER_NOTES_DIR = DATA_DIR / "career_notes"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
REPORTS_DIR = BASE_DIR / "reports"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION: int = 384
TOP_K_RESULTS: int = 5

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

LLM_TEMPERATURE: float = 0.7
LLM_MAX_TOKENS: int = 2000
RESUME_TEXT_MAX_CHARS: int = 8000

os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(RESUMES_DIR, exist_ok=True)
os.makedirs(CAREER_NOTES_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
