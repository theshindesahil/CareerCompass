"""Safe text processing helpers for CareerCompass AI."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable


_SKILL_ALIASES = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "ts": "typescript",
    "reactjs": "react",
    "react.js": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "nextjs": "next.js",
    "vuejs": "vue.js",
    "angularjs": "angular",
    "tailwind": "tailwind css",
    "bootstrap5": "bootstrap",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "mysql db": "mysql",
    "ms excel": "excel",
    "powerbi": "power bi",
    "sklearn": "scikit-learn",
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "genai": "generative ai",
    "llm": "large language models",
    "aws cloud": "aws",
    "gcp": "google cloud",
    "azure cloud": "azure",
    "k8s": "kubernetes",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "oop": "object oriented programming",
    "dsa": "data structures and algorithms",
    "dbms": "database management systems",
    "os": "operating systems",
    "cn": "computer networks",
    "ui": "user interface",
    "ux": "user experience",
    "qa": "quality assurance",
    "sdet": "automation test engineer",
    "pentesting": "penetration testing",
    "web3": "web3",
    "smart contracts": "smart contract development",
}

_STOP_PHRASES = {
    "i", "know", "and", "a", "an", "the", "little", "some", "with", "using",
    "have", "experience", "in", "of", "for", "to", "am", "learning", "built",
}


def _is_nan(value: object) -> bool:
    try:
        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False


def clean_text(text: object) -> str:
    """Convert any input to a safe, normalized text string."""
    try:
        if text is None or _is_nan(text):
            return ""
        if isinstance(text, str):
            raw = text
        elif isinstance(text, Iterable) and not isinstance(text, (bytes, bytearray, dict)):
            raw = " ".join(str(item) for item in text if item is not None)
        else:
            raw = str(text)
        raw = raw.lower()
        raw = re.sub(r"[^a-z0-9\s\+\#\./,\-;_|]", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:10000]
    except Exception:
        return ""


def truncate_text(text: object, max_chars: int = 3000) -> str:
    try:
        value = clean_text(text)
        max_chars = max(0, int(max_chars))
        return value[:max_chars]
    except Exception:
        return ""


def normalize_skill(skill: object) -> str:
    value = clean_text(skill).strip(" ,;_|")
    value = re.sub(r"\s+", " ", value)
    if not value or value in _STOP_PHRASES:
        return ""
    return _SKILL_ALIASES.get(value, value)


def parse_list_input(text: object) -> list[str]:
    """Parse lists, delimiters, and sentence-like input into normalized terms."""
    value = truncate_text(text, 3000)
    if not value:
        return []
    normalized_sentence = re.sub(r"\band\b", ",", value)
    pieces = re.split(r"[,;\n|]+", normalized_sentence)
    if len(pieces) == 1:
        pieces = re.split(r"\s{2,}", normalized_sentence)
    if len(pieces) == 1 and len(pieces[0].split()) > 6:
        pieces = pieces[0].split()

    seen: set[str] = set()
    result: list[str] = []
    for piece in pieces:
        item = normalize_skill(piece)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) >= 100:
            break
    return result


def safe_split_pipe(value: object) -> list[str]:
    if isinstance(value, list):
        pieces = value
    else:
        text = clean_text(value)
        pieces = text.split("|") if text else []
    seen: set[str] = set()
    result: list[str] = []
    for piece in pieces:
        item = normalize_skill(piece)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def create_user_profile_text(
    skills: object,
    interests: object,
    education: object,
    experience: object,
    work_style: object,
    goal: object,
    category_preference: object,
    extra_text: object,
) -> str:
    parts = [
        " ".join(parse_list_input(skills) if not isinstance(skills, list) else skills),
        " ".join(parse_list_input(interests) if not isinstance(interests, list) else interests),
        clean_text(education),
        clean_text(experience),
        clean_text(work_style),
        clean_text(goal),
        clean_text(category_preference),
        truncate_text(extra_text, 3000),
    ]
    return clean_text(" ".join(part for part in parts if part))
