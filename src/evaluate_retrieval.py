"""Evaluate retrieval quality on real Kaggle dataset."""

import sys
import json

sys.path.insert(0, ".")
from src.config import VECTORSTORE_DIR
from src.search.job_search import JobSearchEngine


def main():
    engine = JobSearchEngine()
    if not engine.load_index(str(VECTORSTORE_DIR)):
        print(f"Error: Could not load index from {VECTORSTORE_DIR}")
        sys.exit(1)

    test_queries = [
        ("Python developer with machine learning experience", ["software engineer", "data scientist", "machine learning", "python"]),
        ("Data analyst SQL Tableau visualization", ["data analyst", "business intelligence", "analytics", "data"]),
        ("DevOps engineer Docker Kubernetes cloud", ["devops", "sre", "cloud", "infrastructure", "platform"]),
        ("Frontend developer React JavaScript", ["frontend", "software engineer", "web developer", "react"]),
        ("Product manager Agile Scrum roadmap", ["product manager", "product owner", "program manager"]),
        ("Security engineer cybersecurity penetration testing", ["security", "cybersecurity", "information security"]),
        ("Data engineer ETL pipeline Spark", ["data engineer", "etl", "data pipeline"]),
        ("UX designer Figma user research", ["ux", "ui", "design", "user experience"]),
        ("Machine learning engineer deep learning TensorFlow", ["machine learning", "ml engineer", "deep learning", "ai"]),
        ("Java backend developer Spring Boot microservices", ["java", "backend", "software engineer", "spring"]),
    ]

    print(f"Dataset: {engine.index.ntotal} jobs")
    print(f"Testing {len(test_queries)} queries\n")

    hit5_count = 0
    hit10_count = 0
    all_results = []

    for query, expected_keywords in test_queries:
        results = engine.search(query, top_k=10)

        relevant_in_top5 = False
        relevant_in_top10 = False

        for i, r in enumerate(results[:10]):
            title_lower = r["title"].lower()
            if any(kw in title_lower for kw in expected_keywords):
                if i < 5:
                    relevant_in_top5 = True
                relevant_in_top10 = True

        if relevant_in_top5:
            hit5_count += 1
        if relevant_in_top10:
            hit10_count += 1

        top1_title = results[0]["title"] if results else "N/A"
        top1_score = results[0]["match_score"] if results else 0

        all_results.append({
            "query": query,
            "top1_title": top1_title,
            "top1_score": top1_score,
            "hit5": relevant_in_top5,
            "hit10": relevant_in_top10,
        })

        status5 = "HIT" if relevant_in_top5 else "MISS"
        status10 = "HIT" if relevant_in_top10 else "MISS"
        print(f"Query: {query}")
        print(f"  Top: {top1_title} (score: {top1_score:.3f})")
        print(f"  Hit@5: [{status5}]  Hit@10: [{status10}]")
        print()

    hit5 = hit5_count / len(test_queries)
    hit10 = hit10_count / len(test_queries)

    print("=" * 60)
    print(f"Hit@5:  {hit5:.2%} ({hit5_count}/{len(test_queries)})")
    print(f"Hit@10: {hit10:.2%} ({hit10_count}/{len(test_queries)})")

    report = {
        "dataset": "Real Kaggle (Indeed + LinkedIn)",
        "total_jobs": engine.index.ntotal,
        "num_queries": len(test_queries),
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "results": all_results,
    }

    with open("reports/retrieval_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to reports/retrieval_evaluation.json")


if __name__ == "__main__":
    main()
