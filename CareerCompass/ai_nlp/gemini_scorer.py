"""
Gemini-powered scoring module – replaces entire old scorer
Functions: calculate_final_score(), generate_feedback()
"""
import os
import google.generativeai as genai
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

# Import the hard booster
from ai_nlp.hard_booster import boosted_hard_score

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------ helpers ------------------
model_embed = genai.GenerativeModel("models/embedding-001")
model_gen   = genai.GenerativeModel("gemini-1.5-flash")

TECH_PROMPT = """You are a tech recruiter.  
From the following job description, list ONLY concrete technical skills, tools, libraries, frameworks, cloud services, databases, programming languages.  
Return them as a single comma-separated list, nothing else.

Job Description:
{jd}

Skills:"""

FEEDBACK_PROMPT = """Resume:
{resume}

Job Description:
{jd}

Missing skills: {missing}

Give 3 short, actionable bullet points (less than or equal to 15 words each) to improve the resume for this job.
Bullet points:"""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _gemini_cosine(text1: str, text2: str) -> float:
    try:
        emb1 = genai.embed_content(model="models/embedding-001", content=text1)["embedding"]
        emb2 = genai.embed_content(model="models/embedding-001", content=text2)["embedding"]
        import numpy as np
        a, b = np.array(emb1), np.array(emb2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception as e:
        # Fallback to a simple similarity calculation if Gemini API fails
        # This could happen due to quota limits or network issues
        print(f"Warning: Gemini embedding failed with error: {e}")
        print("Falling back to simple text matching...")
        # Simple fallback: calculate ratio of common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        common = len(words1.intersection(words2))
        total = len(words1.union(words2))
        return common / total if total > 0 else 0.0

# ------------------ tech extractor ------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _extract_tech(jd: str) -> List[str]:
    try:
        prompt = TECH_PROMPT.format(jd=jd)
        reply = model_gen.generate_content(prompt, generation_config={"temperature": 0.0}).text
        return [s.strip().lower() for s in reply.split(",") if s.strip()]
    except Exception as e:
        # Fallback to simple regex-based extraction if Gemini fails
        print(f"Warning: Tech extraction failed with error: {e}")
        print("Falling back to basic regex extraction...")
        import re
        # Simple regex to extract potential tech terms (camelCase, PascalCase, or separated words)
        tech_terms = re.findall(r'\b(?:[A-Z][a-z]+|[a-z]+(?:[A-Z][a-z]*)+)\b', jd)
        # Filter out common non-tech words
        common_words = {'need', 'and', 'with', 'for', 'the', 'in', 'to', 'of', 'is', 'are', 'was', 'were'}
        return [term.lower() for term in tech_terms if term.lower() not in common_words and len(term) > 2]

# ------------------ public API ------------------
def calculate_final_score(resume_text: str, jd_text: str) -> Dict[str, any]:
    # 1. semantic score (0-100)
    semantic = round(_gemini_cosine(resume_text, jd_text) * 100, 2)

    # 2. tech skills & missing
    jd_tech  = _extract_tech(jd_text)
    missing  = [t for t in jd_tech if t not in resume_text.lower() and len(t) > 2][:12]

    # 3. hard proxy = keyword coverage of missing vs total tech
    base_hard = round(max(0, (len(jd_tech) - len(missing)) / (len(jd_tech) or 1) * 100), 2)
    hard      = boosted_hard_score(resume_text, jd_text, base_hard)

    # 4. final score & verdict
    total = round((hard + semantic) / 2, 2)
    verdict = "High" if total >= 80 else "Medium" if total >= 60 else "Low"

    return {
        "hard_score": hard,
        "semantic_score": semantic,
        "total_score": total,
        "verdict": verdict,
        "missing_skills": missing,
    }

def generate_feedback(resume_text: str, jd_text: str, score_data: Dict) -> str:
    missing = ", ".join(score_data["missing_skills"][:6])
    prompt = FEEDBACK_PROMPT.format(resume=resume_text, jd=jd_text, missing=missing)
    bullets = model_gen.generate_content(prompt, generation_config={"temperature": 0.0}).text
    return bullets.strip()