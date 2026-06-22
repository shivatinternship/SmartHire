"""Preprocessing pipeline for real-world job datasets.

Loads raw CSV files from Kaggle/public sources, cleans them, normalizes skills,
and produces a unified jobs.json for the SmartHire GenAI application.

Supported sources:
- Indeed Job Listings (luminati-io/Indeed-dataset-samples)
- LinkedIn Job Postings (nenad0707/analyzing-job-market)
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SKILL_SYNONYMS = {
    "js": "JavaScript",
    "ts": "TypeScript",
    "py": "Python",
    "tf": "TensorFlow",
    "sklearn": "Scikit-learn",
    "k8s": "Kubernetes",
    "k8S": "Kubernetes",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "react.js": "React",
    "reactjs": "React",
    "react js": "React",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "vue.js": "Vue.js",
    "vuejs": "Vue.js",
    "angular.js": "Angular",
    "angularjs": "Angular",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "express.js": "Express.js",
    "expressjs": "Express.js",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "fast api": "FastAPI",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "mssql": "SQL Server",
    "sql server": "SQL Server",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "GCP",
    "google cloud": "GCP",
    "microsoft azure": "Azure",
    "ms azure": "Azure",
    "ci/cd": "CI/CD",
    "ci cd": "CI/CD",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "natural language processing": "NLP",
    "nlp": "NLP",
    "computer vision": "Computer Vision",
    "data science": "Data Science",
    "data analytics": "Data Analysis",
    "data analysis": "Data Analysis",
    "data visualization": "Data Visualization",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "excel": "Excel",
    "spreadsheets": "Excel",
    "agile": "Agile",
    "scrum": "Scrum",
    "kanban": "Kanban",
    "jira": "JIRA",
    "confluence": "Confluence",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "bitbucket": "Bitbucket",
    "jenkins": "Jenkins",
    "travis ci": "Travis CI",
    "circle ci": "CircleCI",
    "circleci": "CircleCI",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "puppet": "Puppet",
    "chef": "Chef",
    "prometheus": "Prometheus",
    "grafana": "Grafana",
    "datadog": "Datadog",
    "elasticsearch": "Elasticsearch",
    "kibana": "Kibana",
    "logstash": "Logstash",
    "spark": "Apache Spark",
    "apache spark": "Apache Spark",
    "hadoop": "Apache Hadoop",
    "apache hadoop": "Apache Hadoop",
    "kafka": "Apache Kafka",
    "apache kafka": "Apache Kafka",
    "airflow": "Apache Airflow",
    "apache airflow": "Apache Airflow",
    "hbase": "HBase",
    "cassandra": "Cassandra",
    "redis": "Redis",
    "memcached": "Memcached",
    "nginx": "Nginx",
    "apache": "Apache",
    "linux": "Linux",
    "unix": "Unix",
    "bash": "Bash",
    "shell scripting": "Shell Scripting",
    "powershell": "PowerShell",
    "matlab": "MATLAB",
    "r": "R",
    "sql": "SQL",
    "nosql": "NoSQL",
    "rest api": "REST API",
    "restful api": "REST API",
    "rest": "REST",
    "graphql": "GraphQL",
    "grpc": "gRPC",
    "websocket": "WebSocket",
    "html": "HTML",
    "css": "CSS",
    "sass": "SASS",
    "less": "LESS",
    "bootstrap": "Bootstrap",
    "tailwind": "Tailwind CSS",
    "tailwind css": "Tailwind CSS",
    "jquery": "jQuery",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "python": "Python",
    "java": "Java",
    "c++": "C++",
    "c#": "C#",
    "csharp": "C#",
    ".net": ".NET",
    "dot net": ".NET",
    "dotnet": ".NET",
    "ruby": "Ruby",
    "php": "PHP",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "swift": "Swift",
    "kotlin": "Kotlin",
    "scala": "Scala",
    "perl": "Perl",
    "lua": "Lua",
    "r": "R",
    "sass": "SAS",
    "spss": "SPSS",
    "stata": "Stata",
    "pytorch": "PyTorch",
    "py torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tensor flow": "TensorFlow",
    "keras": "Keras",
    "scikit-learn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "sklearn": "Scikit-learn",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "scipy": "SciPy",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "plotly": "Plotly",
    "opencv": "OpenCV",
    "nltk": "NLTK",
    "spacy": "spaCy",
    "hugging face": "Hugging Face",
    "huggingface": "Hugging Face",
    "bert": "BERT",
    "gpt": "GPT",
    "llm": "LLM",
    "large language model": "LLM",
    "transformers": "Transformers",
    "cuda": "CUDA",
    "gpu": "GPU",
    "tpu": "TPU",
    "mlops": "MLOps",
    "ml ops": "MLOps",
    "data pipeline": "Data Pipeline",
    "etl": "ETL",
    "data warehouse": "Data Warehousing",
    "data warehousing": "Data Warehousing",
    "data lake": "Data Lake",
    "data lakehouse": "Data Lakehouse",
    "bigquery": "BigQuery",
    "snowflake": "Snowflake",
    "redshift": "Redshift",
    "databricks": "Databricks",
    "dbt": "dbt",
    "delta lake": "Delta Lake",
    "iceberg": "Iceberg",
    "fhir": "FHIR",
    "hipaa": "HIPAA",
    "gdpr": "GDPR",
    "soc 2": "SOC 2",
    "iso 27001": "ISO 27001",
    "owasp": "OWASP",
    "penetration testing": "Penetration Testing",
    "cybersecurity": "Cybersecurity",
    "infosec": "Information Security",
    "information security": "Information Security",
    "cloud": "Cloud Computing",
    "cloud computing": "Cloud Computing",
    "microservices": "Microservices",
    "serverless": "Serverless",
    "lambda": "AWS Lambda",
    "s3": "AWS S3",
    "ec2": "AWS EC2",
    "ecs": "AWS ECS",
    "eks": "AWS EKS",
    "cloud formation": "CloudFormation",
    "cloudformation": "CloudFormation",
    "devops": "DevOps",
    "sre": "SRE",
    "site reliability": "SRE",
    "platform engineering": "Platform Engineering",
    "infrastructure": "Infrastructure",
    "monitoring": "Monitoring",
    "logging": "Logging",
    "observability": "Observability",
    "distributed systems": "Distributed Systems",
    "system design": "System Design",
    "architecture": "Architecture",
    "design patterns": "Design Patterns",
    "oop": "OOP",
    "object oriented": "OOP",
    "functional programming": "Functional Programming",
    "unit testing": "Unit Testing",
    "integration testing": "Integration Testing",
    "e2e testing": "E2E Testing",
    "test automation": "Test Automation",
    "tdd": "TDD",
    "test driven": "TDD",
    "bdd": "BDD",
    "behavior driven": "BDD",
    "code review": "Code Review",
    "pair programming": "Pair Programming",
    "technical writing": "Technical Writing",
    "presentation": "Presentation",
    "communication": "Communication",
    "leadership": "Leadership",
    "team management": "Team Management",
    "project management": "Project Management",
    "product management": "Product Management",
    "stakeholder management": "Stakeholder Management",
    "problem solving": "Problem Solving",
    "analytical": "Analytical Skills",
    "analytical skills": "Analytical Skills",
    "critical thinking": "Critical Thinking",
    "creativity": "Creativity",
    "creative": "Creativity",
    "adaptability": "Adaptability",
    "time management": "Time Management",
    "attention to detail": "Attention to Detail",
    "self-motivated": "Self-Motivated",
    "self motivated": "Self-Motivated",
    "remote": "Remote Work",
    "hybrid": "Hybrid Work",
    "onsite": "Onsite Work",
    "in-person": "Onsite Work",
}

DEGREE_PATTERNS = [
    r"\b(?:bachelor'?s?|b\.?s\.?|b\.?a\.?|b\.?tech)\b",
    r"\b(?:master'?s?|m\.?s\.?|m\.?a\.?|m\.?tech|m\.?b\.?a\.?)\b",
    r"\b(?:ph\.?d\.?|doctorate|doctoral)\b",
    r"\b(?:associate'?s?|a\.?s\.?|a\.?a\.?)\b",
    r"\b(?:diploma|certificate|certification)\b",
]


def _clean_skill_string(raw: str) -> str:
    """Clean a raw skills string by removing brackets, quotes, and normalizing.

    Args:
        raw: Raw skills string (may contain brackets, quotes, etc.).

    Returns:
        Cleaned comma-separated skills string.
    """
    if not raw:
        return ""

    cleaned = raw.strip()
    cleaned = cleaned.strip("[]()")
    cleaned = cleaned.replace("'", "").replace('"', "")
    cleaned = cleaned.replace("_", " ")

    parts = [s.strip() for s in cleaned.split(",") if s.strip()]
    seen = set()
    unique = []
    for part in parts:
        if len(part) > 1 and part.lower() not in seen:
            seen.add(part.lower())
            unique.append(part)

    return ", ".join(unique)


def _clean_text(text: Optional[str]) -> str:
    """Clean and normalize a text field.

    Args:
        text: Raw text input.

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_skills_from_text(text: str) -> list[str]:
    """Extract known skills from free-form text.

    Args:
        text: Text to search for skills.

    Returns:
        List of normalized skill names.
    """
    if not text:
        return []

    text_lower = text.lower()
    found_skills = set()

    for pattern, normalized in SKILL_SYNONYMS.items():
        if len(pattern) >= 2:
            regex = r"\b" + re.escape(pattern) + r"\b"
            if re.search(regex, text_lower):
                found_skills.add(normalized)

    return sorted(found_skills)


def _extract_experience(description: str) -> str:
    """Extract experience requirements from job description.

    Args:
        description: Job description text.

    Returns:
        Extracted experience string or empty string.
    """
    if not description:
        return ""

    patterns = [
        r"(\d+[\+\-]?\s*(?:to|or more|years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?)",
        r"(?:minimum|at least|requires?)\s*(\d+\s*(?:to\s*\d+\s*)?(?:years?|yrs?))",
        r"(\d+\s*(?:\+\s*)?years?(?:\s*of)?\s*(?:experience|exp|professional)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            return match.group(0).strip()

    return ""


def _extract_degree(description: str) -> str:
    """Extract education requirements from job description.

    Args:
        description: Job description text.

    Returns:
        Extracted degree string or empty string.
    """
    if not description:
        return ""

    text_lower = description.lower()

    for pattern in DEGREE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0).strip().title()

    return ""


def _build_salary(min_pay: Optional[str], max_pay: Optional[str], rate: Optional[str]) -> str:
    """Build a normalized salary string.

    Args:
        min_pay: Minimum pay value.
        max_pay: Maximum pay value.
        rate: Pay rate (hourly, yearly, etc.).

    Returns:
        Formatted salary string.
    """
    try:
        min_val = float(min_pay) if min_pay else None
        max_val = float(max_pay) if max_pay else None
    except (ValueError, TypeError):
        return ""

    if min_val is None and max_val is None:
        return ""

    rate_str = (rate or "").lower().strip()
    if "hour" in rate_str:
        if min_val and max_val:
            return f"${min_val:.0f}/hr - ${max_val:.0f}/hr"
        elif min_val:
            return f"${min_val:.0f}/hr"
    elif "month" in rate_str:
        if min_val and max_val:
            return f"${min_val * 12:,.0f} - ${max_val * 12:,.0f}"
    else:
        if min_val and max_val:
            if min_val > 1000:
                return f"${min_val:,.0f} - ${max_val:,.0f}"
            else:
                return f"${min_val * 1000:,.0f} - ${max_val * 1000:,.0f}"

    return ""


def process_indeed_dataset(csv_path: str) -> list[dict]:
    """Process the Indeed job listings CSV.

    Args:
        csv_path: Path to the Indeed CSV file.

    Returns:
        List of normalized job dictionaries.
    """
    jobs = []
    seen = set()

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            title = _clean_text(row.get("job_title", ""))
            company = _clean_text(row.get("company_name", ""))
            location = _clean_text(row.get("location", ""))
            description = _clean_text(row.get("description_text", "") or row.get("description", ""))
            job_type = _clean_text(row.get("job_type", ""))

            if not title or not description:
                continue

            dedup_key = f"{title.lower()}|{company.lower()}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            skills = _extract_skills_from_text(description)
            skills_str = ", ".join(skills) if skills else ""

            experience = _extract_experience(description)
            education = _extract_degree(description)
            salary = _build_salary(
                row.get("salary_formatted"),
                None,
                None,
            )

            jobs.append({
                "title": title,
                "company": company,
                "location": location,
                "skills": skills_str,
                "description": description[:2000],
                "salary": salary,
                "experience": experience,
                "education": education,
                "work_type": job_type,
                "source": "Indeed",
            })

    logger.info(f"Processed {len(jobs)} jobs from Indeed dataset")
    return jobs


def process_linkedin_dataset(csv_path: str) -> list[dict]:
    """Process the LinkedIn job postings CSV.

    Args:
        csv_path: Path to the LinkedIn CSV file.

    Returns:
        List of normalized job dictionaries.
    """
    jobs = []
    seen = set()

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            title = _clean_text(row.get("Job Title", ""))
            full_title = _clean_text(row.get("Job Title Full", ""))
            company = _clean_text(row.get("Company Name", ""))
            location = _clean_text(row.get("Job Location", ""))
            skills_raw = _clean_text(row.get("Job Skills", ""))
            skills_raw = _clean_skill_string(skills_raw)
            position_level = _clean_text(row.get("Job Position Level", ""))
            position_type = _clean_text(row.get("Job Position Type", ""))
            industry = _clean_text(row.get("Company Industry", ""))

            display_title = full_title if full_title else title
            if not display_title:
                continue

            dedup_key = f"{display_title.lower()}|{company.lower()}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            skills = _extract_skills_from_text(skills_raw)
            if not skills:
                skills = _extract_skills_from_text(display_title + " " + skills_raw)
            skills_str = ", ".join(skills) if skills else skills_raw

            min_pay = row.get("Minimum Pay", "")
            max_pay = row.get("Maximum Pay", "")
            pay_rate = row.get("Pay Rate", "")
            salary = _build_salary(min_pay, max_pay, pay_rate)

            experience = _clean_text(row.get("Years of Experience", ""))

            description_parts = [display_title]
            if skills_raw:
                description_parts.append(f"Required skills: {skills_raw}")
            if position_level:
                description_parts.append(f"Level: {position_level}")
            if industry:
                description_parts.append(f"Industry: {industry}")
            description = ". ".join(description_parts)

            jobs.append({
                "title": display_title,
                "company": company,
                "location": location,
                "skills": skills_str,
                "description": description[:2000],
                "salary": salary,
                "experience": experience,
                "education": "",
                "work_type": position_type,
                "level": position_level,
                "industry": industry,
                "source": "LinkedIn",
            })

    logger.info(f"Processed {len(jobs)} jobs from LinkedIn dataset")
    return jobs


def merge_and_deduplicate(
    *job_lists: list[dict],
    target_count: Optional[int] = None,
) -> list[dict]:
    """Merge multiple job lists and remove duplicates.

    Args:
        job_lists: Variable number of job lists to merge.
        target_count: Optional target count to cap the output.

    Returns:
        Merged and deduplicated list of jobs.
    """
    all_jobs = []
    seen = set()

    for job_list in job_lists:
        for job in job_list:
            key = f"{job['title'].lower()}|{job.get('company', '').lower()}"
            if key not in seen:
                seen.add(key)
                all_jobs.append(job)

    if target_count and len(all_jobs) > target_count:
        all_jobs = all_jobs[:target_count]

    logger.info(f"Merged dataset: {len(all_jobs)} unique jobs")
    return all_jobs


def run_preprocessing(
    indeed_path: Optional[str] = None,
    linkedin_path: Optional[str] = None,
    output_path: Optional[str] = None,
    target_count: int = 3000,
) -> str:
    """Run the full preprocessing pipeline.

    Args:
        indeed_path: Path to Indeed CSV file.
        linkedin_path: Path to LinkedIn CSV file.
        output_path: Path to save the output JSON.
        target_count: Target number of jobs in output.

    Returns:
        Path to the output file.
    """
    from src.config import JOBS_DIR

    if output_path is None:
        output_path = str(JOBS_DIR / "jobs.json")

    all_processed = []

    if indeed_path and Path(indeed_path).exists():
        indeed_jobs = process_indeed_dataset(indeed_path)
        all_processed.append(indeed_jobs)
        logger.info(f"Loaded {len(indeed_jobs)} jobs from Indeed")

    if linkedin_path and Path(linkedin_path).exists():
        linkedin_jobs = process_linkedin_dataset(linkedin_path)
        all_processed.append(linkedin_jobs)
        logger.info(f"Loaded {len(linkedin_jobs)} jobs from LinkedIn")

    if not all_processed:
        logger.error("No input files found")
        return output_path

    merged = merge_and_deduplicate(*all_processed, target_count=target_count)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(merged)} jobs to {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    base = Path(__file__).resolve().parent.parent.parent / "data" / "jobs"
    indeed = str(base / "indeed_raw.csv")
    linkedin = str(base / "linkedin_raw.csv")
    output = str(base / "jobs.json")

    result = run_preprocessing(
        indeed_path=indeed,
        linkedin_path=linkedin,
        output_path=output,
        target_count=3000,
    )
    print(f"Preprocessing complete: {result}")
