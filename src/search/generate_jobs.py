"""Generate a scalable job dataset for SmartHire GenAI.

Generates 500+ realistic job listings programmatically without hardcoding
individual entries. Uses combinatorial generation from skill/title/location templates.
"""

import json
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

COMPANIES = [
    "TechCorp Inc.", "DataDriven Solutions", "AI Innovations Lab", "WebScale Systems",
    "CloudFirst Inc.", "InnovateTech", "Analytics Pro", "CodeCraft Studios",
    "BigData Corp", "DeepMind Labs", "Quantum Computing Co", "CyberShield Security",
    "GreenTech Solutions", "FinTech Global", "HealthTech AI", "EduTech Platforms",
    "RetailOps Inc.", "MediaStream Corp", "LogiTrack Systems", "BioData Labs",
    "NextGen Robotics", "SpaceTech Industries", "AutoPilot Systems", "GameDev Studios",
    "AgriTech Innovations", "SupplyChain AI", "LegalTech Solutions", "InsureTech Pro",
    "TravelTech Global", "FoodTech Systems", "EnergyTech Corp", "BuildTech Inc.",
    "WearTech Labs", "SoundTech Audio", "VisionTech AI", "RoboTech Industries",
    "NanoTech Research", "CryptoTech Solutions", "blockchain Labs", "CloudNine Systems",
    "DataForge Inc.", "CodeAlpha Labs", "ByteShift Corp", "PixelPerfect Studios",
    "NeuralNet Inc.", "TensorFlow Solutions", "PyTorch Labs", "SparkData Corp",
    "Kubernetes Cloud", "DockerHub Inc.", "MicroService Pro", "APIFirst Systems",
    "Serverless Co", "EdgeComputing Inc.", "IoTConnect Corp", "DigitalTwin Labs",
    "PredictiveAI Inc.", "MLOps Global", "DataPipeline Corp", "StreamData Inc.",
]

LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Remote", "Austin, TX", "Seattle, WA",
    "Boston, MA", "Chicago, IL", "Denver, CO", "Los Angeles, CA", "Portland, OR",
    "Miami, FL", "Atlanta, GA", "Dallas, TX", "Phoenix, AZ", "San Diego, CA",
    "Minneapolis, MN", "Detroit, MI", "Nashville, TN", "Charlotte, NC", "Raleigh, NC",
    "Salt Lake City, UT", "Pittsburgh, PA", "Columbus, OH", "Indianapolis, IN",
    "Remote (US)", "Remote (Global)", "Hybrid - San Francisco, CA", "Hybrid - New York, NY",
    "London, UK", "Berlin, Germany", "Toronto, Canada", "Singapore",
]

JOB_TEMPLATES = {
    "Software Engineering": {
        "titles": [
            "Software Engineer", "Senior Software Engineer", "Staff Software Engineer",
            "Backend Engineer", "Senior Backend Engineer", "Frontend Engineer",
            "Senior Frontend Engineer", "Full Stack Developer", "Senior Full Stack Developer",
            "Platform Engineer", "Systems Engineer", "Application Developer",
        ],
        "skills_pool": [
            "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#",
            "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask",
            "FastAPI", "Spring Boot", "ASP.NET", "Ruby on Rails",
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
            "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform",
            "Git", "CI/CD", "Jenkins", "GitHub Actions",
            "REST API", "GraphQL", "gRPC", "Microservices",
            "SQL", "NoSQL", "Data Modeling", "System Design",
            "Agile", "Scrum", "Unit Testing", "Integration Testing",
        ],
        "descriptions": [
            "Build and maintain scalable backend services serving millions of users.",
            "Design and implement RESTful APIs and microservices architectures.",
            "Develop responsive web applications using modern frameworks.",
            "Architect and build distributed systems for high-availability services.",
            "Lead technical design and implementation of core platform features.",
            "Optimize application performance and ensure code quality through reviews.",
            "Collaborate with product teams to deliver impactful features on schedule.",
            "Build real-time data processing pipelines for analytics and reporting.",
        ],
    },
    "Data Science": {
        "titles": [
            "Data Scientist", "Senior Data Scientist", "Lead Data Scientist",
            "Research Scientist", "Applied Scientist", "Data Analyst",
            "Senior Data Analyst", "Analytics Engineer", "Quantitative Analyst",
        ],
        "skills_pool": [
            "Python", "R", "SQL", "SAS", "SPSS",
            "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch",
            "Tableau", "Power BI", "Looker", "Excel",
            "Statistics", "Linear Algebra", "Probability", "A/B Testing",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "Data Visualization", "ETL", "Data Modeling",
            "BigQuery", "Snowflake", "Redshift", "Spark", "Hadoop",
            "Jupyter", "Git", "Docker", "Airflow",
        ],
        "descriptions": [
            "Extract actionable insights from complex datasets to drive business decisions.",
            "Build and validate statistical models for forecasting and prediction.",
            "Design and analyze A/B experiments to optimize product features.",
            "Develop machine learning models for recommendation systems.",
            "Create interactive dashboards and reports for stakeholder decision-making.",
            "Collaborate with engineering teams to deploy models to production.",
            "Perform deep-dive analyses into user behavior and product metrics.",
        ],
    },
    "Machine Learning Engineering": {
        "titles": [
            "Machine Learning Engineer", "Senior ML Engineer", "MLOps Engineer",
            "AI Engineer", "Deep Learning Engineer", "NLP Engineer",
            "Computer Vision Engineer", "Applied ML Engineer",
        ],
        "skills_pool": [
            "Python", "PyTorch", "TensorFlow", "Keras", "JAX",
            "Scikit-learn", "Hugging Face", "OpenCV", "NLTK", "spaCy",
            "MLflow", "Weights & Biases", "DVC", "Kubeflow",
            "Docker", "Kubernetes", "AWS SageMaker", "GCP Vertex AI",
            "SQL", "Spark", "Ray", "Dask",
            "FastAPI", "Flask", "TensorRT", "ONNX",
            "CI/CD", "Git", "Linux", "Bash",
            "Statistics", "Linear Algebra", "Deep Learning", "NLP",
        ],
        "descriptions": [
            "Design, train, and deploy machine learning models at scale.",
            "Build end-to-end ML pipelines from data ingestion to model serving.",
            "Optimize model inference for latency and throughput in production.",
            "Research and implement state-of-the-art deep learning architectures.",
            "Develop and maintain ML infrastructure and experiment tracking systems.",
            "Collaborate with data scientists to productionize research prototypes.",
        ],
    },
    "DevOps & Infrastructure": {
        "titles": [
            "DevOps Engineer", "Senior DevOps Engineer", "Site Reliability Engineer",
            "Senior SRE", "Platform Engineer", "Cloud Engineer",
            "Infrastructure Engineer", "Systems Administrator",
        ],
        "skills_pool": [
            "Docker", "Kubernetes", "Terraform", "Pulumi", "Ansible",
            "AWS", "GCP", "Azure", "Linux", "Bash", "Python",
            "Jenkins", "GitHub Actions", "GitLab CI", "ArgoCD",
            "Prometheus", "Grafana", "Datadog", "New Relic",
            "Nginx", "Apache", "HAProxy", "Load Balancing",
            "PostgreSQL", "MySQL", "Redis", "Elasticsearch",
            "Networking", "DNS", "SSL/TLS", "Security",
            "CI/CD", "Infrastructure as Code", "Monitoring",
        ],
        "descriptions": [
            "Design and maintain cloud infrastructure supporting millions of requests.",
            "Implement CI/CD pipelines for rapid, reliable software delivery.",
            "Ensure high availability and disaster recovery for production systems.",
            "Automate infrastructure provisioning and configuration management.",
            "Monitor system performance and respond to incidents promptly.",
            "Optimize cloud costs while maintaining performance and reliability.",
        ],
    },
    "Product Management": {
        "titles": [
            "Product Manager", "Senior Product Manager", "Technical Product Manager",
            "Group Product Manager", "Product Owner", "Associate Product Manager",
        ],
        "skills_pool": [
            "Product Strategy", "Agile", "Scrum", "Kanban", "JIRA",
            "Confluence", "SQL", "Data Analysis", "A/B Testing",
            "User Research", "Wireframing", "Figma", "Sketch",
            "Roadmapping", "OKRs", "KPIs", "Stakeholder Management",
            "Communication", "Leadership", "Prioritization",
            "Market Research", "Competitive Analysis", "Go-to-Market",
            "Python", "R", "Excel", "Tableau",
        ],
        "descriptions": [
            "Define product vision and strategy aligned with business objectives.",
            "Lead cross-functional teams from conception to launch.",
            "Analyze user data and market trends to prioritize feature development.",
            "Collaborate with engineering, design, and marketing teams.",
            "Drive product roadmap planning and execution.",
            "Conduct user research and translate insights into product requirements.",
        ],
    },
    "Data Engineering": {
        "titles": [
            "Data Engineer", "Senior Data Engineer", "Analytics Engineer",
            "ETL Developer", "Data Platform Engineer", "Big Data Engineer",
        ],
        "skills_pool": [
            "Python", "SQL", "Java", "Scala",
            "Apache Spark", "Apache Airflow", "Apache Kafka", "Apache Flink",
            "AWS", "GCP", "Azure", "Snowflake", "BigQuery", "Redshift",
            "ETL", "Data Modeling", "Data Warehousing", "Data Lake",
            "dbt", "Terraform", "Docker", "Kubernetes",
            "PostgreSQL", "MySQL", "MongoDB", "Cassandra",
            "Git", "CI/CD", "Linux", "Bash",
            "Pandas", "PySpark", "Delta Lake", "Iceberg",
        ],
        "descriptions": [
            "Design and maintain scalable data pipelines processing terabytes daily.",
            "Build data warehousing solutions for analytics and reporting.",
            "Develop real-time streaming data infrastructure.",
            "Optimize data quality and reliability across all pipelines.",
            "Collaborate with data scientists to provide clean, accessible datasets.",
            "Implement data governance and cataloging frameworks.",
        ],
    },
    "Cybersecurity": {
        "titles": [
            "Security Engineer", "Senior Security Engineer", "Cybersecurity Analyst",
            "Application Security Engineer", "DevSecOps Engineer", "Security Architect",
        ],
        "skills_pool": [
            "Python", "Bash", "PowerShell", "Go",
            "SIEM", "Splunk", "QRadar", "IDS/IPS",
            "OWASP", "Penetration Testing", "Vulnerability Assessment",
            "AWS Security", "Cloud Security", "Network Security",
            "Firewall", "VPN", "Encryption", "PKI",
            "Docker", "Kubernetes", "Terraform",
            "SOC", "Incident Response", "Forensics",
            "Compliance", "GDPR", "SOC 2", "ISO 27001",
            "Git", "CI/CD", "Linux",
        ],
        "descriptions": [
            "Protect critical infrastructure from cyber threats and vulnerabilities.",
            "Conduct security assessments and penetration testing.",
            "Implement security automation in CI/CD pipelines.",
            "Respond to security incidents and conduct forensic analysis.",
            "Design secure architecture for cloud-native applications.",
            "Ensure compliance with security standards and regulations.",
        ],
    },
    "UX/UI Design": {
        "titles": [
            "UX Designer", "Senior UX Designer", "UI Designer",
            "Product Designer", "UX Researcher", "Design Lead",
        ],
        "skills_pool": [
            "Figma", "Sketch", "Adobe XD", "InVision", "Zeplin",
            "User Research", "Usability Testing", "A/B Testing",
            "Wireframing", "Prototyping", "User Flows", "Information Architecture",
            "HTML", "CSS", "JavaScript", "React",
            "Design Systems", "Accessibility", "WCAG",
            "Typography", "Color Theory", "Layout",
            "Interaction Design", "Visual Design",
            "Miro", "JIRA", "Confluence",
        ],
        "descriptions": [
            "Design intuitive user experiences for web and mobile applications.",
            "Conduct user research and translate findings into design solutions.",
            "Create and maintain design systems for consistent user interfaces.",
            "Collaborate with product and engineering teams on feature design.",
            "Prototype and test new interaction patterns.",
            "Champion accessibility and inclusive design principles.",
        ],
    },
}

SKILL_SALARY_MULTIPLIERS = {
    "python": 1.05, "machine learning": 1.10, "deep learning": 1.12,
    "kubernetes": 1.08, "aws": 1.06, "terraform": 1.07,
    "scala": 1.08, "go": 1.07, "rust": 1.10,
    "spark": 1.06, "kafka": 1.05,
}


def _generate_skills(skills_pool: list[str], count: int = 8) -> list[str]:
    """Pick a random subset of skills from the pool."""
    k = min(count, len(skills_pool))
    return sorted(random.sample(skills_pool, k))


def _generate_salary(skills: list[str]) -> str:
    """Generate a realistic salary range based on skills."""
    base_min = random.randint(70, 100)
    base_max = base_min + random.randint(30, 60)

    multiplier = 1.0
    for skill in skills:
        if skill.lower() in SKILL_SALARY_MULTIPLIERS:
            multiplier = max(multiplier, SKILL_SALARY_MULTIPLIERS[skill.lower()])

    min_salary = int(base_min * multiplier * 1000)
    max_salary = int(base_max * multiplier * 1000)
    return f"${min_salary:,} - ${max_salary:,}"


def generate_jobs(target_count: int = 500, seed: int = 42) -> list[dict]:
    """Generate a job dataset of the specified size.

    Uses combinatorial generation from templates to create diverse,
    realistic job listings without hardcoding individual entries.

    Args:
        target_count: Desired number of jobs to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of job dictionaries.
    """
    random.seed(seed)
    jobs = []

    while len(jobs) < target_count:
        category = random.choice(list(JOB_TEMPLATES.keys()))
        template = JOB_TEMPLATES[category]

        title = random.choice(template["titles"])
        company = random.choice(COMPANIES)
        location = random.choice(LOCATIONS)
        skills = _generate_skills(template["skills_pool"])
        description = random.choice(template["descriptions"])
        salary = _generate_salary(skills)

        jobs.append({
            "title": title,
            "company": company,
            "location": location,
            "skills": ", ".join(skills),
            "description": description,
            "salary": salary,
            "category": category,
        })

    logger.info(f"Generated {len(jobs)} job listings")
    return jobs


def save_jobs(jobs: list[dict], output_path: str) -> None:
    """Save generated jobs to a JSON file.

    Args:
        jobs: List of job dictionaries.
        output_path: Path to the output JSON file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)
    logger.info(f"Saved {len(jobs)} jobs to {output_path}")


if __name__ == "__main__":
    from src.config import JOBS_DIR

    jobs = generate_jobs(target_count=500)
    save_jobs(jobs, str(JOBS_DIR / "jobs.json"))
    print(f"Generated and saved {len(jobs)} jobs to {JOBS_DIR / 'jobs.json'}")
