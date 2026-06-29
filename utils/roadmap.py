"""Roadmap and project generators for CareerCompass AI."""

from __future__ import annotations

from .text_processing import clean_text


def _safe_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    return [item.strip() for item in text.split("|") if item.strip()]


def _topic_tasks(missing: list[str], fallback: list[str]) -> list[str]:
    selected = [skill for skill in missing if skill][:3]
    if not selected:
        selected = fallback[:3]
    return [f"Practice {skill} with notes and small exercises." for skill in selected]


def generate_roadmap(career_name: object, missing_skills: object, existing_roadmap: object) -> list[dict]:
    career = clean_text(career_name).title() or "Target Role"
    missing = _safe_list(missing_skills)
    existing = _safe_list(existing_roadmap)
    base = existing[:4] or [
        f"Understand the responsibilities of a {career}.",
        "Build daily practice around the most important fundamentals.",
        "Create a small portfolio project with clear documentation.",
        "Prepare interview stories and revise core concepts.",
    ]

    joined_missing = " ".join(missing)
    if any(term in joined_missing for term in ["html", "css", "javascript", "react"]):
        base[0:1] = ["Frontend fundamentals: semantic HTML, CSS layout, and JavaScript interactions."]
    if any(term in joined_missing for term in ["sql", "database", "postgresql", "mysql"]):
        base[0:1] = ["Database practice: schema design, SQL joins, and query optimization basics."]
    if any(term in joined_missing for term in ["cloud", "docker", "kubernetes", "aws", "azure"]):
        base[1:2] = ["Deployment basics: Docker, cloud services, logs, and environment configuration."]
    if any(term in joined_missing for term in ["machine learning", "statistics", "scikit-learn"]):
        base[1:2] = ["ML foundations: statistics, model training, evaluation, and feature engineering."]
    if any(term in joined_missing for term in ["security", "network", "penetration"]):
        base[1:2] = ["Security labs: networking basics, threat modeling, and secure coding drills."]
    if any(term in joined_missing for term in ["figma", "user interface", "user experience"]):
        base[0:1] = ["Design practice: Figma flows, wireframes, usability heuristics, and component states."]
    if any(term in joined_missing for term in ["testing", "selenium", "quality"]):
        base[1:2] = ["QA automation: test cases, Selenium flows, API checks, and bug reporting."]
    if any(term in joined_missing for term in ["c", "c++", "embedded", "operating systems"]):
        base[0:1] = ["Systems basics: C/C++, memory, operating systems, and hardware-aware debugging."]

    while len(base) < 4:
        base.append("Build, document, and review one practical portfolio milestone.")

    return [
        {
            "week": f"Week {index + 1}",
            "title": ["Foundations", "Guided Practice", "Portfolio Build", "Interview Polish"][index],
            "tasks": [base[index], *_topic_tasks(missing[index:index + 3], ["documentation", "practice", "feedback"])[:2]],
        }
        for index in range(4)
    ]


def generate_project_recommendations(career_name: object, mini_project_ideas: object, missing_skills: object) -> list[dict]:
    career = clean_text(career_name).title() or "Software"
    ideas = _safe_list(mini_project_ideas)
    missing = _safe_list(missing_skills)[:4] or ["research", "implementation", "testing"]
    defaults = [
        f"{career} Starter Portfolio Project",
        f"{career} Case Study Dashboard",
        f"{career} Real-World Workflow Automation",
    ]
    titles = (ideas + defaults)[:3]
    difficulties = ["Beginner", "Intermediate", "Advanced"]
    return [
        {
            "title": title.title(),
            "description": f"Build a practical {career} project that demonstrates problem solving, clean documentation, and measurable outcomes.",
            "skills": missing[:3],
            "difficulty": difficulties[index],
        }
        for index, title in enumerate(titles)
    ]


def generate_interview_questions(career_name: object) -> list[str]:
    career = clean_text(career_name)
    question_bank = {
        "data": [
            "How would you clean and validate a messy dataset before analysis?",
            "Explain a dashboard metric you would track for a business stakeholder.",
            "How do you choose between correlation and causation in analysis?",
            "Describe a SQL query involving joins and aggregation.",
            "How would you explain a model result to a non-technical audience?",
        ],
        "security": [
            "How would you investigate a suspicious login alert?",
            "What is the difference between vulnerability, threat, and risk?",
            "How do you secure secrets in an application?",
            "Explain basic network segmentation.",
            "How would you document a penetration test finding?",
        ],
        "design": [
            "How do you validate whether a design solves the right user problem?",
            "Describe your process for turning requirements into wireframes.",
            "How do you handle accessibility in interface design?",
            "What makes a component reusable?",
            "How would you respond to conflicting stakeholder feedback?",
        ],
        "cloud": [
            "How would you deploy an application with rollback support?",
            "Explain containers and why teams use them.",
            "How would you monitor service health?",
            "What is the role of CI/CD?",
            "How would you control cloud costs?",
        ],
    }
    for key, questions in question_bank.items():
        if key in career:
            return questions
    return [
        f"What core skills make someone effective as a {career.title() or 'software professional'}?",
        "Describe a project where you debugged a difficult issue.",
        "How do you learn a new technology quickly?",
        "How do you test that your work is correct?",
        "Explain a technical concept from your portfolio in simple language.",
    ]
