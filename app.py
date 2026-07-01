from __future__ import annotations

import html
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.recommender import load_career_data, recommend_careers
from utils.roadmap import (
    generate_interview_questions,
    generate_project_recommendations,
    generate_roadmap,
)


st.set_page_config(page_title="CareerCompass AI", page_icon=None, layout="wide")


EDUCATION_OPTIONS = ["High School", "Diploma", "Undergraduate", "Graduate", "Postgraduate", "Self-taught", "Other"]
EXPERIENCE_OPTIONS = ["Beginner", "Intermediate", "Advanced"]
WORK_STYLE_OPTIONS = ["Analytical", "Creative", "Technical", "Business-Oriented", "Research-Oriented"]
GOAL_OPTIONS = ["Fast hiring", "High salary", "Remote job", "Freelancing", "Startup role", "Research career"]
CATEGORY_OPTIONS = [
    "No preference", "Web Development", "Mobile Development", "Software Engineering", "Data & AI",
    "Cloud & DevOps", "Cybersecurity", "Design & Product", "QA & Testing", "Game & Creative Tech",
    "Emerging Tech", "Systems & Infrastructure", "Enterprise Software", "Technical Communication",
]

DEFAULT_PROFILE = {
    "name": "",
    "skills": "",
    "interests": "",
    "education": "Undergraduate",
    "experience": "Beginner",
    "work_style": "Analytical",
    "goal": "Fast hiring",
    "category_preference": "No preference",
    "extra_text": "",
}

DEMO_PROFILE = {
    "name": "Aisha",
    "skills": "Python, SQL, Excel, Statistics, Pandas, HTML, CSS, Git",
    "interests": "data analysis, dashboards, web apps, business insights, prediction",
    "education": "Undergraduate",
    "experience": "Beginner",
    "work_style": "Analytical",
    "goal": "Fast hiring",
    "category_preference": "No preference",
    "extra_text": "I enjoy solving business problems using data and building useful web dashboards.",
}


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def init_state() -> None:
    for key, value in DEFAULT_PROFILE.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("analysis_result", None)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #111417;
            --panel: #191e23;
            --panel-2: #20262d;
            --ink: #f5f0e8;
            --muted: #aeb8bd;
            --accent: #d8b46a;
            --accent-2: #8fb8a8;
            --green: #8fb8a8;
            --orange: #d39b61;
            --violet: #b5a0d8;
            --line: rgba(245,240,232,.13);
            --line-strong: rgba(216,180,106,.42);
        }
        .stApp { background: var(--bg); }
        .block-container { padding-top: 2rem; max-width: 1240px; }
        .hero {
            position: relative;
            border: 1px solid var(--line);
            border-top: 3px solid var(--accent);
            border-radius: 8px;
            padding: clamp(1.35rem, 3vw, 2.35rem);
            background: linear-gradient(135deg, #181d22 0%, #15191d 62%, #1d211d 100%);
            box-shadow: 0 16px 36px rgba(0,0,0,.22);
            margin-bottom: 1.15rem;
        }
        .hero .kicker {
            margin: 0 0 .65rem;
            color: var(--accent);
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
        }
        .hero h1 { font-size: clamp(2.05rem, 4.4vw, 3.8rem); margin: 0 0 .45rem; letter-spacing: 0; line-height: 1; }
        .hero p { color: #d9ddd7; max-width: 790px; font-size: 1.04rem; line-height: 1.6; }
        .hero small { color: var(--muted); }
        .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: 1rem 0; }
        .metric-card, .rec-card, .road-card, .tool-card {
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(245,240,232,.045), rgba(245,240,232,.02));
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(0,0,0,.18);
        }
        .metric-card { border-top-color: var(--line-strong); }
        .metric-card span { display: block; color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }
        .metric-card strong { display: block; color: var(--ink); font-size: clamp(1.25rem, 2.5vw, 1.9rem); margin-top: .2rem; overflow-wrap: anywhere; }
        .rec-card { margin-bottom: .9rem; }
        .rec-top { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }
        .rec-card h3 { margin: 0; font-size: 1.35rem; }
        .rec-card p { color: #d7ddd7; }
        .score-pill { color:#17130a; background: var(--accent); padding:.35rem .6rem; border-radius:999px; font-weight:800; white-space:nowrap; }
        .badge, .tag {
            display:inline-flex; align-items:center; margin:.16rem .22rem .16rem 0; padding:.25rem .48rem;
            border:1px solid var(--line); border-radius:999px; font-size:.78rem; line-height:1.2;
            background:rgba(245,240,232,.06); color:#f2eee7;
        }
        .badge.best { background: rgba(216,180,106,.16); border-color: rgba(216,180,106,.6); color:#f6d991; }
        .tag.green { background:rgba(143,184,168,.14); border-color:rgba(143,184,168,.46); }
        .tag.blue { background:rgba(134,160,184,.16); border-color:rgba(134,160,184,.45); }
        .tag.orange { background:rgba(211,155,97,.15); border-color:rgba(211,155,97,.48); }
        .tag.violet { background:rgba(181,160,216,.14); border-color:rgba(181,160,216,.45); }
        .road-grid, .tool-grid { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:.85rem; }
        .road-card h4, .tool-card h4 { margin:.1rem 0 .4rem; }
        .footer { color: var(--muted); border-top:1px solid var(--line); margin-top:2rem; padding-top:1rem; font-size:.9rem; }
        div[data-testid="stTabs"] button { font-size: .95rem; }
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            min-height: 44px;
            border: 1px solid var(--line);
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--line-strong);
            transform: translateY(-1px);
        }
        .stButton > button:focus, .stDownloadButton > button:focus {
            box-shadow: 0 0 0 2px rgba(216,180,106,.35);
        }
        div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="select"] {
            border-radius: 8px;
        }
        @media (prefers-reduced-motion: reduce) {
            .stButton > button, .stDownloadButton > button { transition: none; }
            .stButton > button:hover, .stDownloadButton > button:hover { transform: none; }
        }
        @media (max-width: 900px) {
            .metric-grid, .road-grid, .tool-grid { grid-template-columns: 1fr; }
            .rec-top { display:block; }
            .score-pill { display:inline-block; margin-top:.5rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_profile(profile: dict) -> None:
    for key, value in profile.items():
        st.session_state[key] = value
    st.session_state["analysis_result"] = None


def list_tags(items: object, color: str = "") -> str:
    values = items if isinstance(items, list) else []
    if not values:
        return "<span class='tag'>None yet</span>"
    return "".join(f"<span class='tag {color}'>{esc(item)}</span>" for item in values[:20])


def badges(row: pd.Series, rank: int) -> str:
    data = []
    if rank == 0:
        data.append(("Best Match", "best"))
    if row.get("beginner_friendly_score", 0) >= 80:
        data.append(("Beginner Friendly", ""))
    if row.get("salary_potential_score", 0) >= 85:
        data.append(("High Salary", ""))
    if row.get("remote_friendly_score", 0) >= 80:
        data.append(("Remote Friendly", ""))
    if row.get("market_demand_score", 0) >= 80:
        data.append(("High Demand", ""))
    if row.get("creativity_score", 0) >= 80:
        data.append(("Creative Path", ""))
    if row.get("technical_depth_score", 0) >= 85:
        data.append(("Deep Tech", ""))
    if row.get("final_score", 0) < 50:
        data.append(("Exploratory", ""))
    return "".join(f"<span class='badge {klass}'>{label}</span>" for label, klass in data)


def safe_join(value: object, sep: str = ", ") -> str:
    if isinstance(value, list):
        return sep.join(str(item) for item in value)
    return str(value or "")


def build_report(user_inputs: dict, top_rows: pd.DataFrame) -> str:
    try:
        top = top_rows.iloc[0] if not top_rows.empty else {}
        roadmap = generate_roadmap(top.get("career_name", "Career"), top.get("missing_required_skills", []), top.get("roadmap", []))
        projects = generate_project_recommendations(top.get("career_name", "Career"), top.get("mini_project_ideas", []), top.get("missing_required_skills", []))
        questions = generate_interview_questions(top.get("career_name", "Career"))
        buffer = StringIO()
        buffer.write("# CareerCompass AI Report\n\n")
        buffer.write(f"**Name:** {user_inputs.get('name') or 'Learner'}\n\n")
        buffer.write(f"**Skills:** {user_inputs.get('skills') or 'Not provided'}\n\n")
        buffer.write(f"**Interests:** {user_inputs.get('interests') or 'Not provided'}\n\n")
        buffer.write(f"**Education:** {user_inputs.get('education')}\n\n")
        buffer.write(f"**Experience:** {user_inputs.get('experience')}\n\n")
        buffer.write(f"**Goal:** {user_inputs.get('goal')}\n\n")
        buffer.write(f"**Category preference:** {user_inputs.get('category_preference')}\n\n")
        buffer.write("## Top Recommendations\n\n")
        for _, row in top_rows.head(3).iterrows():
            buffer.write(f"### {row.get('career_name')} - {row.get('final_score')}%\n")
            buffer.write(f"- Category: {row.get('category')}\n")
            buffer.write(f"- Why: {row.get('explanation')}\n")
            buffer.write(f"- Matched skills: {safe_join(row.get('matched_required_skills', [])) or 'None yet'}\n")
            buffer.write(f"- Missing skills: {safe_join(row.get('missing_required_skills', [])) or 'None'}\n")
            buffer.write(f"- Tools to learn: {safe_join(row.get('tools', []))}\n\n")
        buffer.write("## 4-Week Roadmap\n\n")
        for week in roadmap:
            buffer.write(f"### {week['week']}: {week['title']}\n")
            for task in week["tasks"]:
                buffer.write(f"- {task}\n")
            buffer.write("\n")
        buffer.write("## Project Ideas\n\n")
        for project in projects:
            buffer.write(f"- **{project['title']}** ({project['difficulty']}): {project['description']}\n")
        buffer.write("\n## Interview Questions\n\n")
        for question in questions:
            buffer.write(f"- {question}\n")
        buffer.write("\n## Disclaimer\n\nThis is an educational recommendation system for demo and learning purposes. It does not guarantee job offers, salaries, admissions, or hiring outcomes.\n")
        return buffer.getvalue()
    except Exception:
        return "# CareerCompass AI Report\n\nA partial report could be generated. Please run the analysis again for full details.\n"


@st.cache_data(show_spinner=False)
def cached_data() -> tuple[pd.DataFrame, list[str]]:
    return load_career_data()


init_state()
inject_css()
career_df, data_warnings = cached_data()

st.markdown(
    """
    <section class="hero">
        <p class="kicker">Software career path predictor</p>
        <h1>CareerCompass AI</h1>
        <p>Compare your skills, interests, and goals against curated software career profiles using transparent scoring and text similarity.</p>
        <small>Built with Python, Streamlit, scikit-learn, Pandas, NumPy, and Plotly. Free and open-source demo project.</small>
    </section>
    """,
    unsafe_allow_html=True,
)

if data_warnings:
    with st.expander("Data loading notes"):
        for warning in data_warnings:
            st.warning(warning)

with st.sidebar:
    st.header("Your Profile")
    st.text_input("Name", key="name")
    st.text_area("Skills", key="skills", placeholder="Python, JavaScript, SQL, React, Git", height=90)
    st.text_area("Interests", key="interests", placeholder="web apps, cybersecurity, AI, cloud, mobile apps", height=90)
    st.selectbox("Education level", EDUCATION_OPTIONS, key="education")
    st.selectbox("Experience level", EXPERIENCE_OPTIONS, key="experience")
    st.selectbox("Preferred work style", WORK_STYLE_OPTIONS, key="work_style")
    st.selectbox("Career goal", GOAL_OPTIONS, key="goal")
    st.selectbox("Career category preference", CATEGORY_OPTIONS, key="category_preference")
    st.text_area(
        "Extra profile text",
        key="extra_text",
        placeholder="Paste resume summary, project description, learning goals, or background.",
        height=130,
    )
    analyze = st.button("Analyze My Career Path", type="primary", use_container_width=True)
    st.button("Try Demo Profile", use_container_width=True, on_click=set_profile, args=(DEMO_PROFILE,))
    st.button("Reset", use_container_width=True, on_click=set_profile, args=(DEFAULT_PROFILE,))

current_inputs = {
    "name": st.session_state.name,
    "skills": st.session_state.skills,
    "interests": st.session_state.interests,
    "education": st.session_state.education,
    "experience": st.session_state.experience,
    "work_style": st.session_state.work_style,
    "goal": st.session_state.goal,
    "category_preference": st.session_state.category_preference,
    "extra_text": st.session_state.extra_text,
}

if analyze:
    with st.spinner("Analyzing your profile and comparing it with software career paths..."):
        st.session_state.analysis_result = recommend_careers(current_inputs, career_df)

result = st.session_state.analysis_result
if not result:
    left, right = st.columns([1.2, .8])
    with left:
        st.subheader("Start with any useful signal")
        st.write("Add skills, interests, tools, a resume summary, or learning goals. The app handles messy text and still explains how every score is produced.")
    with right:
        st.info("For a quick presentation run, click **Try Demo Profile** in the sidebar, then analyze.")
else:
    for warning in result.get("warnings", []):
        st.warning(warning)
    for error in result.get("errors", []):
        st.error(error)

    recs = result.get("recommendations", pd.DataFrame())
    if result.get("success") and isinstance(recs, pd.DataFrame) and not recs.empty:
        top = recs.iloc[0]
        missing_count = len(top.get("missing_required_skills", []))
        st.markdown(
            f"""
            <div class="metric-grid">
                <div class="metric-card"><span>Best Career Match</span><strong>{esc(top.get('career_name'))}</strong></div>
                <div class="metric-card"><span>Match Score</span><strong>{float(top.get('final_score', 0)):.1f}%</strong></div>
                <div class="metric-card"><span>Career Category</span><strong>{esc(top.get('category'))}</strong></div>
                <div class="metric-card"><span>Missing Skills</span><strong>{missing_count}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if top.get("final_score", 0) < 50:
            st.warning("Your profile currently has exploratory matches. Learning a few core skills can improve your fit.")

        report = build_report(current_inputs, recs)
        st.download_button("Download Markdown Report", report, "careercompass_report.md", "text/markdown")

        tab_recs, tab_gap, tab_roadmap, tab_explorer, tab_toolkit, tab_model = st.tabs(
            ["Recommendations", "Skill Gap", "Roadmap", "Career Explorer", "Free Toolkit", "Model Explanation"]
        )

        with tab_recs:
            for rank, (_, row) in enumerate(recs.head(3).iterrows()):
                st.markdown(
                    f"""
                    <article class="rec-card">
                        <div class="rec-top">
                            <div>
                                <h3>{esc(row.get('career_name'))}</h3>
                                <p>{esc(row.get('category'))}</p>
                            </div>
                            <span class="score-pill">{float(row.get('final_score', 0)):.1f}%</span>
                        </div>
                        <p>{esc(row.get('short_description'))}</p>
                        <p><strong>Why:</strong> {esc(row.get('explanation'))}</p>
                        <div>{badges(row, rank)}</div>
                    </article>
                    """,
                    unsafe_allow_html=True,
                )
            try:
                chart_data = recs.head(15).sort_values("final_score", ascending=True)
                fig = px.bar(chart_data, x="final_score", y="career_name", orientation="h", color="category", title="Top Career Match Scores")
                fig.update_layout(template="plotly_dark", xaxis_title="Match score", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.dataframe(recs[["career_name", "category", "final_score"]].head(15), use_container_width=True)

            for _, row in recs.head(5).iterrows():
                with st.expander(f"Score breakdown: {row.get('career_name')}"):
                    st.dataframe(
                        pd.DataFrame({
                            "Factor": ["Skill", "TF-IDF", "Interest", "Category", "Experience", "Work style", "Goal", "Market demand"],
                            "Score": [
                                row.get("skill_score"), row.get("tfidf_score"), row.get("interest_score"), row.get("category_score"),
                                row.get("experience_score"), row.get("work_style_score"), row.get("goal_score"), row.get("market_demand_score"),
                            ],
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

        with tab_gap:
            readiness = max(0, min(100, float(top.get("skill_score", 0))))
            st.metric("Skill readiness for top career", f"{readiness:.1f}%")
            st.metric("Missing required skills", missing_count)
            if not top.get("matched_required_skills"):
                st.info("No direct skill matches were found yet, but your interests and profile text may still suggest this path.")
            if not top.get("missing_required_skills"):
                st.success("You already cover the core skills. Focus on portfolio depth, real-world projects, and interview preparation.")
            st.markdown("**Matched required skills**")
            st.markdown(list_tags(top.get("matched_required_skills", []), "green"), unsafe_allow_html=True)
            st.markdown("**Matched nice-to-have skills**")
            st.markdown(list_tags(top.get("matched_nice_skills", []), "blue"), unsafe_allow_html=True)
            st.markdown("**Missing required skills**")
            st.markdown(list_tags(top.get("missing_required_skills", []), "orange"), unsafe_allow_html=True)
            st.markdown("**Tools to learn**")
            st.markdown(list_tags(top.get("tools", []), "violet"), unsafe_allow_html=True)
            next_skills = top.get("missing_required_skills", [])[:5]
            st.write("Top skills to learn next:", ", ".join(next_skills) if next_skills else "Portfolio depth, projects, and interviews.")

        with tab_roadmap:
            roadmap = generate_roadmap(top.get("career_name"), top.get("missing_required_skills", []), top.get("roadmap", []))
            projects = generate_project_recommendations(top.get("career_name"), top.get("mini_project_ideas", []), top.get("missing_required_skills", []))
            questions = generate_interview_questions(top.get("career_name"))
            st.markdown('<div class="road-grid">', unsafe_allow_html=True)
            for week in roadmap:
                tasks = "".join(f"<li>{esc(task)}</li>" for task in week["tasks"])
                st.markdown(f"<div class='road-card'><h4>{esc(week['week'])}: {esc(week['title'])}</h4><ul>{tasks}</ul></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.subheader("Mini Projects")
            st.markdown('<div class="tool-grid">', unsafe_allow_html=True)
            for project in projects:
                st.markdown(
                    f"<div class='tool-card'><h4>{esc(project['title'])}</h4><p>{esc(project['description'])}</p><span class='badge'>{esc(project['difficulty'])}</span></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.subheader("Interview Questions")
            for question in questions:
                st.write(f"- {question}")

        with tab_explorer:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                selected_category = st.selectbox("Filter category", ["All"] + sorted(career_df["category"].dropna().unique().tolist()))
            with col2:
                beginner_only = st.toggle("Beginner friendly")
            with col3:
                remote_only = st.toggle("Remote friendly")
            with col4:
                high_salary_only = st.toggle("High salary")
            search = st.text_input("Search by career name")
            filtered = career_df.copy()
            if selected_category != "All":
                filtered = filtered[filtered["category"] == selected_category]
            if beginner_only:
                filtered = filtered[filtered["beginner_friendly_score"] >= 80]
            if remote_only:
                filtered = filtered[filtered["remote_friendly_score"] >= 80]
            if high_salary_only:
                filtered = filtered[filtered["salary_potential_score"] >= 85]
            if search:
                filtered = filtered[filtered["career_name"].str.contains(search, case=False, na=False)]
            if filtered.empty:
                st.info("No careers match the selected filters. Try relaxing the filters.")
            else:
                st.dataframe(
                    filtered[["career_name", "category", "beginner_friendly_score", "salary_potential_score", "remote_friendly_score", "market_demand_score"]],
                    use_container_width=True,
                    hide_index=True,
                )
                try:
                    fig = px.scatter(
                        filtered,
                        x="learning_difficulty_score",
                        y="salary_potential_score",
                        size="market_demand_score",
                        color="category",
                        hover_name="career_name",
                        title="Career Explorer Map",
                    )
                    fig.update_layout(template="plotly_dark", xaxis_title="Learning difficulty", yaxis_title="Salary potential")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.dataframe(filtered, use_container_width=True)

        with tab_toolkit:
            st.subheader("Free Career Toolkit")
            st.info("Everything here is included for the demo. No payment, login, account, or API key is needed.")
            toolkit = [
                ("12-week advanced roadmap", "Expand the 4-week plan into foundations, guided projects, portfolio depth, and interview preparation."),
                ("Resume keyword optimizer", "Mirror the top role's skills in project bullets only when you can support them with real work."),
                ("Interview practice set", "Use the questions in the roadmap tab and add one story for each project you built."),
                ("Portfolio builder", "Publish three small projects: one fundamentals project, one practical workflow, and one polished case study."),
                ("Job search checklist", "Track applications, role requirements, missing skills, interview rounds, and follow-up dates."),
                ("Presentation script", "Explain the problem, inputs, scoring formula, top match, skill gap, and roadmap in five minutes."),
            ]
            st.markdown('<div class="tool-grid">', unsafe_allow_html=True)
            for title, body in toolkit:
                st.markdown(f"<div class='tool-card'><h4>{esc(title)}</h4><p>{esc(body)}</p></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_model:
            st.subheader("How the model works")
            st.write(
                "CareerCompass cleans text, normalizes common skill aliases, compares exact skill overlap, uses TF-IDF with cosine similarity for profile text, "
                "and combines interests, category preference, experience fit, work style, goals, and market demand into one explainable score."
            )
            st.code(
                "final_score =\n"
                "0.30 * skill_score\n"
                "+ 0.18 * tfidf_score\n"
                "+ 0.12 * interest_score\n"
                "+ 0.10 * category_score\n"
                "+ 0.09 * experience_score\n"
                "+ 0.08 * work_style_score\n"
                "+ 0.06 * goal_score\n"
                "+ 0.07 * market_demand_score"
            )
            st.write("Skill gap analysis compares your normalized skills against each role's required and nice-to-have skills.")
            st.warning("This is an educational recommendation system. It does not guarantee job offers, salaries, admissions, or hiring outcomes.")
    elif not result.get("errors"):
        st.info("No recommendations could be displayed yet. Try adding a few skills or a short profile description.")

st.markdown(
    "<div class='footer'>CareerCompass AI is a free, open-source educational demo. It stores no personal data permanently and makes no external API calls.</div>",
    unsafe_allow_html=True,
)
