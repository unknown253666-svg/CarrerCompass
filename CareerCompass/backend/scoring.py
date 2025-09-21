"""
scoring.py  –  single-file, drop-in scorer
Hard + semantic + missing skills + short feedback
Works for ANY domain / any JD / any resume
"""
from typing import List, Dict
import re
import string
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

# ------------------ load once ------------------
nlp = spacy.load("en_core_web_sm")  # Enable all components including parser for noun_chunks
_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ text clean ------------------
def _clean(text: str) -> str:
    """Lower-case + insert space before camel-case + collapse whitespace."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

STOP_EN = set("the and or of to in for with on at by an a is was are be been").union(string.punctuation)

# ------------------ hard score ------------------
def calculate_hard_score(resume: str, jd: str) -> float:
    r, j = map(_clean, (resume, jd))
    # 1. keyword recall
    jd_kw = {tok.lemma_.lower() for tok in nlp(j) if tok.pos_ in {"NOUN", "PROPN"} and len(tok) > 2 and not tok.is_stop}
    if not jd_kw:
        kw_score = 0.0
    else:
        kw_score = len([w for w in jd_kw if w in r]) / len(jd_kw) * 100
    # 2. TF-IDF
    try:
        vect = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        tfidf = vect.fit_transform([r, j])
        tfidf_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
    except ValueError:
        tfidf_score = 0.0
    # 3. fuzzy
    fuzzy_score = fuzz.token_set_ratio(r, j)
    return round(0.4 * kw_score + 0.4 * tfidf_score + 0.2 * fuzzy_score, 2)

# ------------------ semantic score ------------------
def calculate_semantic_score(resume: str, jd: str) -> float:
    r, j = map(_clean, (resume, jd))
    emb = _sbert.encode([r, j], normalize_embeddings=True)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return round(sim * 100, 2)

# ------------------ missing skills ------------------
def extract_missing_skills(resume: str, jd: str) -> List[str]:
    resume_lower, jd_lower = resume.lower(), jd.lower()
    # Extract capitalized words as potential technologies (excluding common words)
    tech = {w.lower() for w in re.findall(r'\b[A-Z][a-z]+\b', jd)} - {"the","and","for","with","from","this","that","need"}
    # Process with spaCy
    doc = nlp(jd_lower)
    # Extract bigrams that contain relevant technical terms
    bigrams = set()
    for i in range(len(doc)-1):
        # Check if either token is a relevant technical term
        if any(token.text.lower() in {"spark","kafka","pyspark","databricks","nlp","vision","devops","python","pandas","numpy","ml","deep","learning","sql","tableau","docker","kubernetes"} for token in (doc[i], doc[i+1])):
            # Create bigram and ensure it doesn't have punctuation
            bigram = " ".join([doc[i].text, doc[i+1].text])
            # Filter out bigrams with common words like "and", "with", etc. at the beginning or end
            words = bigram.split()
            if len(words) == 2 and all(c.isalpha() or c.isspace() for c in bigram) and words[0] not in {"and","with","for","need"} and words[1] not in {"and","with","for","need"}:
                bigrams.add(bigram)
    
    # Combine candidates and filter out those present in resume
    missing = sorted([c for c in (tech | bigrams) if c not in resume_lower and len(c) > 2])[:12]
    return missing

# ------------------ verdict ------------------
def _verdict(total: float) -> str:
    return "High" if total >= 80 else "Medium" if total >= 60 else "Low"

# ------------------ public API ------------------
def calculate_final_score(resume_text: str, jd_text: str) -> Dict[str, any]:
    hard = calculate_hard_score(resume_text, jd_text)
    semantic = calculate_semantic_score(resume_text, jd_text)
    total = round((hard + semantic) / 2, 2)
    missing = extract_missing_skills(resume_text, jd_text)
    return {
        "hard_score": float(hard),  # Convert to Python native float
        "semantic_score": float(semantic),  # Convert to Python native float
        "total_score": float(total),  # Convert to Python native float
        "verdict": _verdict(total),
        "missing_skills": missing,
    }

def generate_feedback(resume_text: str, jd_text: str, score_data: Dict) -> str:
    missing = score_data["missing_skills"]
    if not missing:
        return "Good match! Minor tweaks may help."
    top = ", ".join(missing[:6])
    return f"Add / highlight: {top}."

