# CareerCompass AI

CareerCompass AI is a free, open-source Streamlit demo that recommends software and technology career paths from a learner's skills, interests, education, experience level, work style, goals, and profile text.

It is designed for classroom presentations, portfolio demos, and beginner-friendly data science explanation. No API key is required.

## Overview

The app compares a user profile against a curated career dataset covering software development, data, AI, cloud, cybersecurity, design, QA, infrastructure, enterprise software, and technical writing roles.

It uses transparent scoring instead of paid APIs or black-box services. Every result includes matched skills, missing skills, score breakdowns, a 4-week roadmap, project ideas, and interview questions.

## Problem Statement

Students and early-career learners often know some tools, but do not know which software career paths fit them best. CareerCompass AI helps them explore realistic roles, understand skill gaps, and plan next learning steps.

## Features

- Career recommendations with explainable scoring
- 50+ software and technology career profiles
- Skill normalization for common aliases like `py`, `js`, `nodejs`, `k8s`, and `powerbi`
- TF-IDF text similarity using scikit-learn
- Skill gap analysis with matched and missing skills
- Personalized 4-week roadmap
- Mini project recommendations
- Role-specific interview questions
- Career explorer with filters and Plotly charts
- Free career toolkit for presentation and job-search preparation
- Markdown report download
- Defensive error handling and fallback data
- Streamlit Community Cloud friendly

## Career Fields Covered

- Web Development
- Mobile Development
- Software Engineering
- Data & AI
- Cloud & DevOps
- Cybersecurity
- Design & Product
- QA & Testing
- Game & Creative Tech
- Emerging Tech
- Systems & Infrastructure
- Enterprise Software
- Technical Communication

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly

## Folder Structure

```text
career-compass-ai/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── career_profiles.csv
├── utils/
│   ├── __init__.py
│   ├── recommender.py
│   ├── text_processing.py
│   ├── roadmap.py
│   └── validation.py
└── .streamlit/
    └── config.toml
```

## Recommendation Formula

```text
final_score =
0.30 * skill_score
+ 0.18 * tfidf_score
+ 0.12 * interest_score
+ 0.10 * category_score
+ 0.09 * experience_score
+ 0.08 * work_style_score
+ 0.06 * goal_score
+ 0.07 * market_demand_score
```

## How the Model Works

1. User text is cleaned and normalized.
2. Skills and interests are parsed from commas, semicolons, newlines, or sentence-like input.
3. Common skill variants are normalized, such as `py` to `python` and `k8s` to `kubernetes`.
4. Required and nice-to-have skill overlap is calculated.
5. A career profile text is built for every role.
6. TF-IDF vectorization and cosine similarity compare the user profile with career profile text.
7. Interest, category, experience, work style, goal, and market demand scores are calculated.
8. The final weighted score ranks career paths.
9. Missing skills are used to generate a roadmap and project ideas.

## Error Handling and Edge Cases

The app is built to keep running when:

- User inputs are blank, duplicated, noisy, emoji-heavy, or sentence-like
- Extra text is very long
- `career_profiles.csv` is missing, empty, corrupted, or has invalid values
- Numeric scores are missing, non-numeric, or outside the 0-100 range
- Plotly chart rendering fails
- Roadmap or project data is incomplete
- Download reports are generated from partial data

If the CSV cannot be used, the app loads fallback career data from `utils/recommender.py`.

## How to Run Locally

From the project folder:

```bash
uv venv
.\.venv\Scripts\activate
uv pip install -r requirements.txt
streamlit run app.py
```

If you do not activate the environment, you can still run:

```bash
uv run streamlit run app.py
```

## How to Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to Streamlit Community Cloud.
3. Click **New app**.
4. Select the GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy.

No API key is required.

## Demo Profile

Use the **Try Demo Profile** button in the sidebar.

```text
Name: Aisha
Skills: Python, SQL, Excel, Statistics, Pandas, HTML, CSS, Git
Interests: data analysis, dashboards, web apps, business insights, prediction
Education: Undergraduate
Experience: Beginner
Work style: Analytical
Goal: Fast hiring
Category preference: No preference
Extra text: I enjoy solving business problems using data and building useful web dashboards.
```

## Limitations

- This is an educational recommendation system, not a hiring system.
- Scores are based on curated profile data and transparent heuristics.
- Salary, demand, and difficulty scores are approximate demo values.
- It does not guarantee job offers, admissions, salaries, or career outcomes.

## Future Improvements

- Add more regional career data
- Add CSV upload for custom career profiles
- Add more visual explanations of each scoring factor
- Add printable PDF report generation
- Add more project templates for each role

## Disclaimer

CareerCompass AI is a free educational demo and presentation project. It does not collect payments, require authentication, call external APIs, or store personal data permanently.
