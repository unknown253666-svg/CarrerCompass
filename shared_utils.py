"""
Shared utilities module for CareerCompass application.
This module contains functions that are used by both the main Streamlit app and pages.
"""
import tempfile
import os
import json
import sqlite3
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pdfplumber
import docx2txt
import io
import csv

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        # Fallback to punkt if punkt_tab is not available
        nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Load models
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None
except ImportError:
    nlp = None

# Load SentenceTransformer model
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Database functions
def get_db_connection():
    """
    Create a database connection
    """
    db_path = os.getenv('DB_PATH', 'career_compass.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initialize the database with required tables
    """
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_text TEXT,
            jd_text TEXT,
            hard_score FLOAT,
            semantic_score FLOAT,
            final_score FLOAT,
            verdict TEXT,
            feedback TEXT,
            missing_skills TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_evaluation(resume_text, jd_text, score_data, feedback, missing_skills):
    """
    Save evaluation results to database
    
    Args:
        resume_text (str): Parsed resume content
        jd_text (str): Parsed job description content
        score_data (dict): Score calculation results
        feedback (str): Generated feedback
        missing_skills (list): List of missing skills
        
    Returns:
        int: ID of the saved evaluation
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Convert missing_skills list to JSON string
    missing_skills_json = json.dumps(missing_skills)
    
    cursor.execute('''
        INSERT INTO evaluations 
        (resume_text, jd_text, hard_score, semantic_score, final_score, verdict, feedback, missing_skills)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        resume_text,
        jd_text,
        score_data['hard_score'],
        score_data['semantic_score'],
        score_data['total_score'],
        score_data['verdict'],
        feedback,
        missing_skills_json
    ))
    
    evaluation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return evaluation_id

def get_all_evaluations():
    """
    Retrieve all evaluations from database
    
    Returns:
        list: List of evaluations
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM evaluations ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    
    evaluations = []
    for row in rows:
        # Convert missing_skills JSON string back to list
        missing_skills = json.loads(row['missing_skills']) if row['missing_skills'] else []
        
        evaluations.append({
            'id': row['id'],
            'hard_score': row['hard_score'],
            'semantic_score': row['semantic_score'],
            'final_score': row['final_score'],
            'verdict': row['verdict'],
            'missing_skills': missing_skills,
            'timestamp': row['timestamp']
        })
    
    conn.close()
    return evaluations

def get_evaluation_by_id(evaluation_id):
    """
    Retrieve a specific evaluation by ID
    
    Args:
        evaluation_id (int): ID of the evaluation to retrieve
        
    Returns:
        dict: Evaluation data or None if not found
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM evaluations WHERE id = ?', (evaluation_id,))
    row = cursor.fetchone()
    
    if row:
        # Convert missing_skills JSON string back to list
        missing_skills = json.loads(row['missing_skills']) if row['missing_skills'] else []
        
        evaluation = {
            'id': row['id'],
            'resume_text': row['resume_text'],
            'jd_text': row['jd_text'],
            'hard_score': row['hard_score'],
            'semantic_score': row['semantic_score'],
            'final_score': row['final_score'],
            'verdict': row['verdict'],
            'feedback': row['feedback'],
            'missing_skills': missing_skills,
            'timestamp': row['timestamp']
        }
        conn.close()
        return evaluation
    
    conn.close()
    return None

# Text processing functions
def extract_text_from_pdf(file_path):
    """
    Extract text from PDF file
    
    Args:
        file_path (str): Path to PDF file
        
    Returns:
        str: Extracted text
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        # If pdfplumber fails, return empty string
        pass
    return text

def extract_text_from_docx(file_path):
    """
    Extract text from DOCX file
    
    Args:
        file_path (str): Path to DOCX file
        
    Returns:
        str: Extracted text
    """
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        # If docx2txt fails, return empty string
        return ""

def normalize_text(text):
    """
    Normalize text using NLP techniques
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize with error handling
    try:
        tokens = word_tokenize(text)
    except LookupError:
        # If tokenization fails, split by whitespace
        tokens = text.split()
    
    # Remove stopwords with error handling
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    except LookupError:
        # If stopwords fail, skip this step
        pass
    
    # Lemmatize with error handling
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    except:
        # If lemmatization fails, skip this step
        pass
    
    # Join tokens back to text
    normalized_text = ' '.join(tokens)
    
    return normalized_text

def parse_resume(uploaded_file):
    """
    Parse resume file and extract text content
    
    Args:
        uploaded_file: Uploaded file object (PDF or DOCX)
        
    Returns:
        str: Extracted text content
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filename = tmp_file.name
    
    try:
        # Check file extension
        if uploaded_file.name.lower().endswith('.pdf'):
            text = extract_text_from_pdf(tmp_filename)
        elif uploaded_file.name.lower().endswith(('.docx', '.doc')):
            text = extract_text_from_docx(tmp_filename)
        else:
            # Try to read as text file
            with open(tmp_filename, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        # Normalize the text
        text = normalize_text(text)
        return text
    finally:
        # Clean up temporary file
        os.unlink(tmp_filename)

# Scoring functions
def _clean(text: str) -> str:
    """Lower-case + insert space before camel-case + collapse whitespace."""
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()

def calculate_hard_score(resume: str, jd: str) -> float:
    r, j = map(_clean, (resume, jd))
    # 1. keyword recall
    if nlp is not None:
        jd_kw = {tok.lemma_.lower() for tok in nlp(j) if tok.pos_ in {"NOUN", "PROPN"} and len(tok) > 2 and not tok.is_stop}
        if not jd_kw:
            kw_score = 0.0
        else:
            kw_score = len([w for w in jd_kw if w in r]) / len(jd_kw) * 100
    else:
        # Fallback if spaCy is not available
        kw_score = 0.0
        
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

def calculate_semantic_score(resume: str, jd: str) -> float:
    r, j = map(_clean, (resume, jd))
    emb = sbert.encode([r, j], normalize_embeddings=True)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return round(sim * 100, 2)

def extract_missing_skills(resume: str, jd: str):
    resume_lower, jd_lower = resume.lower(), jd.lower()
    # Extract capitalized words as potential technologies (excluding common words)
    tech = {w.lower() for w in re.findall(r'\b[A-Z][a-z]+\b', jd)} - {"the","and","for","with","from","this","that","need"}
    
    # Process with spaCy if available
    if nlp:
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
    else:
        # Fallback if spaCy is not available
        missing = sorted([c for c in tech if c not in resume_lower and len(c) > 2])[:12]
        return missing

def _verdict(total: float) -> str:
    return "High" if total >= 80 else "Medium" if total >= 60 else "Low"

def calculate_final_score(resume_text: str, jd_text: str):
    hard = calculate_hard_score(resume_text, jd_text)
    semantic = calculate_semantic_score(resume_text, jd_text)
    total = round((hard + semantic) / 2, 2)
    missing = extract_missing_skills(resume_text, jd_text)
    return {
        "hard_score": float(hard),
        "semantic_score": float(semantic),
        "total_score": float(total),
        "verdict": _verdict(total),
        "missing_skills": missing,
    }

def generate_feedback(resume_text: str, jd_text: str, score_data) -> str:
    missing = score_data["missing_skills"]
    if not missing:
        return "Good match! Minor tweaks may help."
    top = ", ".join(missing[:6])
    return f"Add / highlight: {top}."

def export_csv():
    """
    Export results as CSV
    """
    evaluations = get_all_evaluations()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Evaluation ID', 'Resume Text', 'JD Text', 'Hard Score', 'Semantic Score', 'Final Score', 'Verdict', 'Missing Skills', 'Feedback'])
    for evaluation in evaluations:
        # Handle case where feedback might not be present
        feedback = evaluation.get('feedback', '') if isinstance(evaluation, dict) else ''
        writer.writerow([
            evaluation['id'],
            evaluation['resume_text'],
            evaluation['jd_text'],
            evaluation['hard_score'],
            evaluation['semantic_score'],
            evaluation['final_score'],
            evaluation['verdict'],
            ', '.join(evaluation['missing_skills']),
            feedback
        ])
    output.seek(0)
    return output.getvalue()
