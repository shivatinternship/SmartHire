"""Centralized skill normalization for SmartHire GenAI.

Provides consistent normalization across resume parsing, job matching,
CV suggestions, and resume tailoring modules.
"""

import re
import logging

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
    "pytorch": "PyTorch",
    "py torch": "PyTorch",
    "tensorflow": "TensorFlow",
    "tensor flow": "TensorFlow",
    "keras": "Keras",
    "scikit-learn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
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

_lower_to_normalized: dict[str, str] = {}


def _build_lookup() -> dict[str, str]:
    """Build a case-insensitive lookup from synonym keys to canonical names."""
    lookup = {}
    for key, canonical in SKILL_SYNONYMS.items():
        lookup[key.lower().strip()] = canonical
        lookup[canonical.lower().strip()] = canonical
    return lookup


def _get_lookup() -> dict[str, str]:
    global _lower_to_normalized
    if not _lower_to_normalized:
        _lower_to_normalized = _build_lookup()
    return _lower_to_normalized


def normalize_skill_name(skill: str) -> str:
    """Normalize a single skill name.

    Applies casing, spacing, punctuation, and alias resolution.

    Args:
        skill: Raw skill name.

    Returns:
        Normalized skill name.
    """
    if not skill or not isinstance(skill, str):
        return str(skill) if skill else ""

    cleaned = skill.strip()

    lookup = _get_lookup()
    key = cleaned.lower().strip()
    if key in lookup:
        return lookup[key]

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    key = cleaned.lower()
    if key in lookup:
        return lookup[key]

    cleaned = cleaned.rstrip(".")
    key = cleaned.lower()
    if key in lookup:
        return lookup[key]

    if cleaned.islower() or cleaned.isupper():
        cleaned = cleaned.title()

    return cleaned


def normalize_skills(skills: list[str]) -> list[str]:
    """Normalize a list of skills.

    Applies normalization to each skill, removes duplicates (case-insensitive),
    removes empty strings, and preserves order.

    Args:
        skills: Raw list of skill names.

    Returns:
        Deduplicated, normalized skill list.
    """
    if not skills:
        return []

    seen: set[str] = set()
    result: list[str] = []

    for skill in skills:
        normalized = normalize_skill_name(skill)
        if not normalized:
            continue
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            result.append(normalized)

    return result


def deduplicate_skills(skills: list[str]) -> list[str]:
    """Remove duplicate skills while preserving order.

    Args:
        skills: List of skill names.

    Returns:
        Deduplicated list.
    """
    if not skills:
        return []

    seen: set[str] = set()
    result: list[str] = []

    for skill in skills:
        key = skill.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(skill)

    return result


def normalize_skill_list_for_comparison(skills: list[str]) -> list[str]:
    """Normalize skills for comparison purposes (case-insensitive, aliases resolved).

    Returns a deduplicated set of normalized skill names that can be used
    for matching across resume parsing, job matching, and CV suggestions.

    Args:
        skills: Raw list of skill names.

    Returns:
        Normalized, deduplicated list.
    """
    return normalize_skills(skills)
