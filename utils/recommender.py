"""Defensive recommendation engine for CareerCompass AI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .text_processing import clean_text, create_user_profile_text, normalize_skill, safe_split_pipe
from .validation import REQUIRED_COLUMNS, validate_career_dataframe, validate_user_inputs


def _row(**kwargs) -> dict:
    defaults = {
        "career_name": "Software Developer",
        "category": "Software Engineering",
        "required_skills": "python|git|problem solving|data structures|debugging",
        "nice_to_have_skills": "sql|testing|cloud",
        "tools": "vs code|github|terminal",
        "related_interests": "software|automation|building products",
        "work_style": "Technical|Analytical",
        "education_fit": "Undergraduate|Graduate|Self-taught",
        "experience_level": "Beginner|Intermediate",
        "goal_fit": "Fast hiring|Remote job",
        "beginner_friendly_score": 75,
        "salary_potential_score": 78,
        "remote_friendly_score": 78,
        "learning_difficulty_score": 55,
        "creativity_score": 55,
        "technical_depth_score": 72,
        "market_demand_score": 80,
        "short_description": "Builds useful software with clean logic, reliable tests, and maintainable code.",
        "roadmap": "Learn fundamentals|Build projects|Practice testing|Prepare interviews",
        "mini_project_ideas": "Portfolio app|Automation script|API project",
    }
    defaults.update(kwargs)
    return defaults


def get_fallback_career_data() -> pd.DataFrame:
    rows = [
        _row(career_name="Frontend Developer", category="Web Development", required_skills="html|css|javascript|react|git", nice_to_have_skills="typescript|tailwind css|accessibility", tools="vs code|chrome devtools|github", related_interests="web apps|user interface|visual design", beginner_friendly_score=88, remote_friendly_score=86, creativity_score=82, market_demand_score=84),
        _row(career_name="Backend Developer", category="Web Development", required_skills="python|sql|apis|databases|git", nice_to_have_skills="django|docker|testing", tools="postgresql|postman|github", related_interests="systems|apis|data flow", technical_depth_score=86, market_demand_score=85),
        _row(career_name="Full Stack Developer", category="Web Development", required_skills="html|css|javascript|python|sql|git", nice_to_have_skills="react|node.js|docker", tools="vs code|github|postman", related_interests="web apps|startups|product building", market_demand_score=88),
        _row(career_name="Python Developer", category="Software Engineering", required_skills="python|oop|git|debugging|testing", nice_to_have_skills="django|flask|sql", tools="pycharm|vs code|github", related_interests="automation|backend|scripting", beginner_friendly_score=86),
        _row(career_name="Data Analyst", category="Data & AI", required_skills="excel|sql|statistics|python|data visualization", nice_to_have_skills="pandas|power bi|tableau", tools="excel|power bi|jupyter", related_interests="dashboards|business insights|reports", work_style="Analytical|Business-Oriented", beginner_friendly_score=90, market_demand_score=86),
        _row(career_name="Data Scientist", category="Data & AI", required_skills="python|statistics|machine learning|sql|pandas", nice_to_have_skills="scikit-learn|deep learning|storytelling", tools="jupyter|scikit-learn|github", related_interests="prediction|experiments|research", experience_level="Intermediate|Advanced", salary_potential_score=90, technical_depth_score=88),
        _row(career_name="Machine Learning Engineer", category="Data & AI", required_skills="python|machine learning|statistics|apis|model deployment", nice_to_have_skills="docker|cloud|mlops", tools="scikit-learn|mlflow|github", related_interests="artificial intelligence|automation|models", experience_level="Intermediate|Advanced", salary_potential_score=92, technical_depth_score=92),
        _row(career_name="DevOps Engineer", category="Cloud & DevOps", required_skills="linux|docker|ci/cd|cloud|scripting", nice_to_have_skills="kubernetes|terraform|monitoring", tools="docker|github actions|aws", related_interests="deployment|automation|reliability", salary_potential_score=90, technical_depth_score=88, market_demand_score=88),
        _row(career_name="Cloud Engineer", category="Cloud & DevOps", required_skills="cloud|linux|networking|security|automation", nice_to_have_skills="aws|azure|terraform", tools="aws|azure|docker", related_interests="infrastructure|scaling|platforms", salary_potential_score=91, technical_depth_score=87),
        _row(career_name="Cybersecurity Analyst", category="Cybersecurity", required_skills="networking|security|linux|incident response|risk analysis", nice_to_have_skills="python|siem|penetration testing", tools="wireshark|splunk|nmap", related_interests="security|investigation|threats", work_style="Analytical|Technical", market_demand_score=87),
        _row(career_name="UI/UX Designer", category="Design & Product", required_skills="user interface|user experience|figma|wireframing|user research", nice_to_have_skills="prototyping|accessibility|html", tools="figma|miro|notion", related_interests="design|users|product", work_style="Creative|Business-Oriented", creativity_score=94, technical_depth_score=48),
        _row(career_name="QA Engineer", category="QA & Testing", required_skills="testing|test cases|bug reporting|quality assurance|sql", nice_to_have_skills="selenium|python|api testing", tools="jira|selenium|postman", related_interests="quality|debugging|reliability", beginner_friendly_score=88, market_demand_score=78),
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def load_career_data(path: str = "data/career_profiles.csv") -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    try:
        csv_path = Path(path)
        if not csv_path.exists():
            alt = Path(__file__).resolve().parents[1] / path
            csv_path = alt
        if not csv_path.exists() or os.path.getsize(csv_path) == 0:
            warnings.append("Career CSV was not found or was empty, so fallback career data was loaded.")
            return validate_career_dataframe(get_fallback_career_data())["df"], warnings
        df = pd.read_csv(csv_path)
        validation = validate_career_dataframe(df)
        if not validation["is_valid"]:
            warnings.extend(validation.get("errors", []))
            warnings.append("Fallback career data was loaded instead.")
            return validate_career_dataframe(get_fallback_career_data())["df"], warnings
        warnings.extend(validation.get("warnings", []))
        return validation["df"], warnings
    except Exception:
        warnings.append("Career CSV could not be read, so fallback career data was loaded.")
        return validate_career_dataframe(get_fallback_career_data())["df"], warnings


def safe_ratio(numerator: float, denominator: float) -> float:
    try:
        return 0.0 if float(denominator) == 0 else float(numerator) / float(denominator)
    except Exception:
        return 0.0


def calculate_overlap_score(user_items: object, target_items: object) -> tuple[float, list[str], list[str]]:
    users = {normalize_skill(item) for item in (user_items or []) if normalize_skill(item)}
    targets = [normalize_skill(item) for item in (target_items or []) if normalize_skill(item)]
    matched = [item for item in targets if item in users]
    missing = [item for item in targets if item not in users]
    return round(100 * safe_ratio(len(matched), len(targets)), 2), matched, missing


def calculate_skill_score(user_skills: list[str], required_skills: list[str], nice_to_have_skills: list[str]) -> dict:
    required = required_skills or nice_to_have_skills or []
    nice = nice_to_have_skills or []
    req_score, matched_required, missing_required = calculate_overlap_score(user_skills, required)
    nice_score, matched_nice, _ = calculate_overlap_score(user_skills, nice)
    if not required and not nice:
        score = 0
    elif not required_skills:
        score = nice_score
    else:
        score = 0.8 * req_score + 0.2 * nice_score
    return {
        "score": round(score, 2),
        "matched_required": matched_required,
        "matched_nice": matched_nice,
        "missing_required": missing_required,
    }


def calculate_interest_score(user_interests: list[str], career_interests: list[str]) -> float:
    score, _, _ = calculate_overlap_score(user_interests, career_interests)
    return score


def calculate_experience_score(user_experience: str, career_experience_levels: list[str]) -> float:
    user = clean_text(user_experience)
    if not user:
        return 60
    matrix = {
        "beginner": {"beginner": 100, "intermediate": 60, "advanced": 30},
        "intermediate": {"beginner": 80, "intermediate": 100, "advanced": 70},
        "advanced": {"beginner": 60, "intermediate": 90, "advanced": 100},
    }
    return max([matrix.get(user, {}).get(level, 60) for level in career_experience_levels] or [60])


def calculate_work_style_score(user_work_style: str, career_work_styles: list[str], career_category: str) -> float:
    user = clean_text(user_work_style).title()
    styles = [clean_text(item).title() for item in career_work_styles]
    category = clean_text(career_category).title()
    if not user:
        return 60
    if user in styles:
        return 100
    related = {
        "Analytical": ["Technical", "Research-Oriented", "Data & Ai", "Business Analyst", "Bi Analyst"],
        "Creative": ["Design & Product", "Web Development", "Game & Creative Tech", "Product Designer", "Ui/Ux Designer"],
        "Technical": ["Software Engineering", "Backend Developer", "Cloud & Devops", "Cybersecurity", "Systems & Infrastructure", "Data & Ai"],
        "Business-Oriented": ["Product Manager - Technical", "Product Analyst", "Business Analyst", "Bi Analyst", "Solutions Architect", "Enterprise Software"],
        "Research-Oriented": ["Data Scientist", "Ai Engineer", "Nlp Engineer", "Computer Vision Engineer", "Robotics Software Engineer", "Emerging Tech"],
    }
    return 70 if category in related.get(user, []) or any(style in related.get(user, []) for style in styles) else 40


def calculate_goal_score(user_goal: str, career_goal_fit: list[str]) -> float:
    goal = clean_text(user_goal).title()
    fits = [clean_text(item).title() for item in career_goal_fit]
    if not goal:
        return 60
    if goal in fits:
        return 100
    related = {
        "Fast Hiring": ["Frontend Developer", "Qa Engineer", "Data Analyst", "Business Analyst", "Technical Writer", "Low-Code / No-Code Developer"],
        "High Salary": ["Ai Engineer", "Machine Learning Engineer", "Cloud Engineer", "Devops Engineer", "Security Engineer", "Solutions Architect"],
        "Remote Job": ["Frontend Developer", "Backend Developer", "Data Analyst", "Devops Engineer", "Technical Writer"],
        "Freelancing": ["Frontend Developer", "Full Stack Developer", "Ui/Ux Designer", "Wordpress/Php", "Mobile App Developer"],
        "Startup Role": ["Full Stack Developer", "Backend Developer", "Product Manager - Technical", "Devops Engineer"],
        "Research Career": ["Data Scientist", "Ai Engineer", "Nlp Engineer", "Computer Vision Engineer", "Robotics Software Engineer"],
    }
    return 70 if any(item in related.get(goal, []) for item in fits) else 40


def calculate_category_score(category_preference: str, career_category: str) -> float:
    pref = clean_text(category_preference).title()
    category = clean_text(career_category).title()
    if not pref or pref == "No Preference":
        return 70
    if pref == category:
        return 100
    related = {
        "Web Development": ["Design & Product", "Software Engineering"],
        "Data & Ai": ["Cloud & Devops", "Software Engineering"],
        "Cloud & Devops": ["Systems & Infrastructure", "Cybersecurity"],
        "Cybersecurity": ["Systems & Infrastructure", "Cloud & Devops"],
        "Mobile Development": ["Software Engineering", "Design & Product"],
        "Game & Creative Tech": ["Design & Product", "Software Engineering"],
        "Enterprise Software": ["Software Engineering", "Systems & Infrastructure"],
    }
    return 60 if category in related.get(pref, []) else 25


def calculate_tfidf_scores(user_profile_text: str, career_profile_texts: list[str]) -> list[float]:
    try:
        if not clean_text(user_profile_text) or not career_profile_texts:
            return [0.0] * len(career_profile_texts)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2500)
        matrix = vectorizer.fit_transform([user_profile_text] + career_profile_texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        max_score = float(scores.max()) if len(scores) else 0
        if max_score <= 0:
            return [0.0] * len(career_profile_texts)
        return [round(float(score / max_score) * 100, 2) for score in scores]
    except Exception:
        return [0.0] * len(career_profile_texts)


def create_career_profile_text(row: pd.Series) -> str:
    parts = []
    for col in ["career_name", "category", "required_skills", "nice_to_have_skills", "tools", "related_interests", "work_style", "short_description"]:
        value = row.get(col, "")
        parts.append(" ".join(value) if isinstance(value, list) else clean_text(value))
    return clean_text(" ".join(parts))


def generate_explanation(row: pd.Series) -> str:
    matched = row.get("matched_required_skills", [])
    interests = row.get("interest_score", 0)
    category_score = row.get("category_score", 0)
    score = row.get("final_score", 0)
    notes = []
    if matched:
        notes.append(f"Strong skill overlap in {', '.join(matched[:3])}.")
    if interests >= 50:
        notes.append("Your interests line up well with this role.")
    if category_score >= 80:
        notes.append("It also matches your preferred career category.")
    if score < 50:
        notes.append("This is exploratory, so focus on the missing core skills first.")
    return " ".join(notes) or "This path is suggested from your profile text, interests, and general market fit."


def recommend_careers(user_inputs: dict, career_df: pd.DataFrame) -> dict:
    validation = validate_user_inputs(user_inputs)
    if not validation["is_valid"]:
        return {"success": False, "errors": validation["errors"], "warnings": validation["warnings"], "recommendations": pd.DataFrame(), "is_exploratory": True}
    cleaned = validation["cleaned_inputs"]
    data_validation = validate_career_dataframe(career_df)
    if not data_validation["is_valid"]:
        return {"success": False, "errors": data_validation["errors"], "warnings": validation["warnings"], "recommendations": pd.DataFrame(), "is_exploratory": True}
    df = data_validation["df"].copy()
    user_skills = list(dict.fromkeys(cleaned.get("skills", []) + cleaned.get("languages_tools", [])))
    user_profile_text = create_user_profile_text(
        user_skills, cleaned.get("interests", []), cleaned.get("education", ""), cleaned.get("experience", ""),
        cleaned.get("work_style", ""), cleaned.get("goal", ""), cleaned.get("category_preference", ""), cleaned.get("extra_text", ""),
    )
    tfidf_scores = calculate_tfidf_scores(user_profile_text, [create_career_profile_text(row) for _, row in df.iterrows()])
    rows = []
    for index, row in df.iterrows():
        skill = calculate_skill_score(user_skills, row["required_skills"], row["nice_to_have_skills"])
        interest_score = calculate_interest_score(cleaned.get("interests", []), row["related_interests"])
        category_score = calculate_category_score(cleaned.get("category_preference", ""), row["category"])
        experience_score = calculate_experience_score(cleaned.get("experience", ""), row["experience_level"])
        work_style_score = calculate_work_style_score(cleaned.get("work_style", ""), row["work_style"], row["category"])
        goal_score = calculate_goal_score(cleaned.get("goal", ""), row["goal_fit"])
        market_score = float(row.get("market_demand_score", 60))
        tfidf = tfidf_scores[index] if index < len(tfidf_scores) else 0
        final = (
            0.30 * skill["score"] + 0.18 * tfidf + 0.12 * interest_score + 0.10 * category_score
            + 0.09 * experience_score + 0.08 * work_style_score + 0.06 * goal_score + 0.07 * market_score
        )
        output = row.to_dict()
        output.update({
            "final_score": round(float(np.nan_to_num(final, nan=0, posinf=100, neginf=0)), 2),
            "skill_score": skill["score"],
            "tfidf_score": tfidf,
            "interest_score": interest_score,
            "category_score": category_score,
            "experience_score": experience_score,
            "work_style_score": work_style_score,
            "goal_score": goal_score,
            "matched_required_skills": skill["matched_required"],
            "matched_nice_skills": skill["matched_nice"],
            "missing_required_skills": skill["missing_required"],
        })
        rows.append(output)
    recs = pd.DataFrame(rows)
    recs["final_score"] = recs["final_score"].clip(0, 100)
    recs = recs.sort_values(
        ["final_score", "beginner_friendly_score", "market_demand_score", "salary_potential_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    recs["explanation"] = recs.apply(generate_explanation, axis=1)
    return {
        "success": True,
        "errors": [],
        "warnings": validation["warnings"] + data_validation["warnings"],
        "recommendations": recs,
        "is_exploratory": bool(recs.empty or recs.iloc[0]["final_score"] < 50),
    }
