"""Validation helpers for user input and career data."""

from __future__ import annotations

import pandas as pd

from .text_processing import clean_text, parse_list_input, safe_split_pipe, truncate_text


REQUIRED_COLUMNS = [
    "career_name", "category", "required_skills", "nice_to_have_skills", "tools",
    "related_interests", "work_style", "education_fit", "experience_level",
    "goal_fit", "beginner_friendly_score", "salary_potential_score",
    "remote_friendly_score", "learning_difficulty_score", "creativity_score",
    "technical_depth_score", "market_demand_score", "short_description",
    "roadmap", "mini_project_ideas",
]

NUMERIC_COLUMNS = [
    "beginner_friendly_score", "salary_potential_score", "remote_friendly_score",
    "learning_difficulty_score", "creativity_score", "technical_depth_score",
    "market_demand_score",
]

PIPE_COLUMNS = [
    "required_skills", "nice_to_have_skills", "tools", "related_interests",
    "work_style", "education_fit", "experience_level", "goal_fit", "roadmap",
    "mini_project_ideas",
]

TEXT_COLUMNS = ["career_name", "category", "short_description"]

CATEGORY_DISPLAY = {
    "web development": "Web Development",
    "mobile development": "Mobile Development",
    "software engineering": "Software Engineering",
    "data & ai": "Data & AI",
    "cloud & devops": "Cloud & DevOps",
    "cybersecurity": "Cybersecurity",
    "design & product": "Design & Product",
    "qa & testing": "QA & Testing",
    "game & creative tech": "Game & Creative Tech",
    "emerging tech": "Emerging Tech",
    "systems & infrastructure": "Systems & Infrastructure",
    "enterprise software": "Enterprise Software",
    "technical communication": "Technical Communication",
}


def validate_user_inputs(user_inputs: dict) -> dict:
    warnings: list[str] = []
    errors: list[str] = []
    cleaned = dict(user_inputs or {})

    skills = parse_list_input(cleaned.get("skills", ""))
    interests = parse_list_input(cleaned.get("interests", ""))
    tools = parse_list_input(cleaned.get("languages_tools", ""))
    extra_original = cleaned.get("extra_text", "")
    extra_text = truncate_text(extra_original, 3000)

    if len(clean_text(extra_original)) > 3000:
        warnings.append("Extra profile text was trimmed to 3000 characters.")
    if len(skills) > 100:
        skills = skills[:100]
        warnings.append("Skills were trimmed to the first 100 items.")
    if len(interests) > 100:
        interests = interests[:100]
        warnings.append("Interests were trimmed to the first 100 items.")

    cleaned.update({
        "name": clean_text(cleaned.get("name", "")).title(),
        "skills": skills,
        "interests": interests,
        "languages_tools": tools,
        "education": clean_text(cleaned.get("education", "")),
        "experience": clean_text(cleaned.get("experience", "")),
        "work_style": clean_text(cleaned.get("work_style", "")),
        "goal": clean_text(cleaned.get("goal", "")),
        "category_preference": clean_text(cleaned.get("category_preference", "")),
        "extra_text": extra_text,
    })

    meaningful = skills or interests or tools or extra_text
    if not meaningful:
        errors.append("Please enter at least a few skills, interests, tools, or a short profile description.")

    return {
        "is_valid": not errors,
        "warnings": warnings,
        "errors": errors,
        "cleaned_inputs": cleaned,
    }


def validate_career_dataframe(df: pd.DataFrame) -> dict:
    warnings: list[str] = []
    errors: list[str] = []
    try:
        if df is None or df.empty:
            return {"is_valid": False, "warnings": warnings, "errors": ["Career data is empty."], "df": pd.DataFrame()}

        clean_df = df.copy()
        missing_required = [col for col in REQUIRED_COLUMNS if col not in clean_df.columns and col not in NUMERIC_COLUMNS]
        if missing_required:
            return {
                "is_valid": False,
                "warnings": warnings,
                "errors": [f"Career data is missing required columns: {', '.join(missing_required)}."],
                "df": pd.DataFrame(),
            }

        for col in NUMERIC_COLUMNS:
            if col not in clean_df.columns:
                clean_df[col] = 60
                warnings.append(f"Missing score column '{col}' was filled with defaults.")
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce").fillna(60).clip(0, 100).astype(int)

        for col in PIPE_COLUMNS:
            if col not in clean_df.columns:
                clean_df[col] = "general"
                warnings.append(f"Missing list column '{col}' was filled with defaults.")
            clean_df[col] = clean_df[col].apply(safe_split_pipe)

        for col in TEXT_COLUMNS:
            if col not in clean_df.columns:
                clean_df[col] = "Unknown"
            clean_df[col] = clean_df[col].apply(clean_text)

        clean_df["career_name"] = clean_df["career_name"].str.title()
        clean_df["category"] = clean_df["category"].apply(lambda value: CATEGORY_DISPLAY.get(clean_text(value), str(value).title()))
        clean_df["short_description"] = clean_df["short_description"].replace("", "A practical technology career path.")
        clean_df = clean_df[clean_df["career_name"].astype(bool)]
        clean_df = clean_df.drop_duplicates(subset=["career_name"], keep="first")

        if clean_df.empty:
            errors.append("Career data had no usable rows after cleaning.")
        return {"is_valid": not errors, "warnings": warnings, "errors": errors, "df": clean_df}
    except Exception:
        return {"is_valid": False, "warnings": warnings, "errors": ["Career data could not be validated."], "df": pd.DataFrame()}
