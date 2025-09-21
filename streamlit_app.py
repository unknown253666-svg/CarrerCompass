import streamlit as st
import pandas as pd
import os
from datetime import datetime
import base64
import json
import tempfile
import sqlite3
import re
import string
import numpy as np
import spacy
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
@st.cache_resource
def load_models():
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If the model is not installed, we'll use a basic approach
        nlp = None
    
    # Load SentenceTransformer model
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    
    return nlp, sbert

nlp, sbert = load_models()

# Add the missing get_nlp function
def get_nlp():
    """Lazy loading of spaCy model to avoid import-time errors"""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model is not found, try to download it
            import subprocess
            import sys
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                nlp = spacy.load("en_core_web_sm")
            except:
                # If downloading also fails, we'll continue with None
                pass
    return nlp

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

# Initialize database
init_db()

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

STOP_EN = set("the and or of to in for with on at by an a is was are be been").union(string.punctuation)

def calculate_hard_score(resume: str, jd: str) -> float:
    r, j = map(_clean, (resume, jd))
    # 1. keyword recall
    jd_kw = {tok.lemma_.lower() for tok in get_nlp()(j) if tok.pos_ in {"NOUN", "PROPN"} and len(tok) > 2 and not tok.is_stop}
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
    if nlp:  # This should use get_nlp() but nlp is already defined from load_models
        # Use the loaded nlp model
        doc = nlp(jd_lower) if nlp is not None else None
        # Extract bigrams that contain relevant technical terms
        bigrams = set()
        if doc is not None:
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

# Database functions
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

# Set page configuration
st.set_page_config(
    page_title="Career Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom HTML for header
st.markdown("""
<div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: #4a90e2;">🧭 Career Compass</h1>
    <p>Upload your resume and get AI-powered insights to optimize your job application success.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["Upload Resume", "Student Upload", "Placement Dashboard", "History"])
    
    st.header("About")
    st.write("Career Compass helps you optimize your resume for better job application success using AI analysis.")

# Main content based on selected page
if page == "Upload Resume":
    st.header("Resume Analysis")
    
    # File uploaders
    resume_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    jd_text = st.text_area("Paste the job description", height=200)
    
    if st.button("Analyze Resume"):
        if resume_file is not None and jd_text:
            with st.spinner("Analyzing your resume..."):
                try:
                    # Parse resume
                    resume_text = parse_resume(resume_file)
                    
                    # Calculate scores
                    score_data = calculate_final_score(resume_text, jd_text)
                    
                    # Generate feedback
                    feedback = generate_feedback(resume_text, jd_text, score_data)
                    
                    # Extract missing skills from score_data
                    missing_skills = score_data['missing_skills']
                    
                    # Save to database
                    evaluation_id = save_evaluation(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        score_data=score_data,
                        feedback=feedback,
                        missing_skills=missing_skills
                    )
                    
                    # Display results
                    st.success("Analysis completed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Hard Score", f"{score_data['hard_score']:.2f}%")
                    with col2:
                        st.metric("Semantic Score", f"{score_data['semantic_score']:.2f}%")
                    with col3:
                        st.metric("Final Score", f"{score_data['total_score']:.2f}%")
                    
                    st.subheader("Verdict")
                    st.write(score_data['verdict'])
                    
                    st.subheader("Feedback")
                    st.info(feedback)
                    
                    if missing_skills:
                        st.subheader("Missing Skills")
                        st.write(", ".join(missing_skills))
                    
                    # Add to history
                    if 'history' not in st.session_state:
                        st.session_state.history = []
                    
                    st.session_state.history.append({
                        'Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Hard Score': f"{score_data['hard_score']:.2f}%",
                        'Semantic Score': f"{score_data['semantic_score']:.2f}%",
                        'Final Score': f"{score_data['total_score']:.2f}%",
                        'Verdict': score_data['verdict']
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
        else:
            st.warning("Please upload a resume and enter a job description.")

elif page == "Placement Dashboard":
    st.header("Placement Dashboard")
    
    st.write("""
    View all resume evaluations, filter by score or student, and download results as CSV.
    """)
    
    # Fetch evaluations
    evaluations = get_all_evaluations()
    
    if evaluations:
        # Convert to DataFrame
        df = pd.DataFrame(evaluations)
        
        # Format timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Filters
        st.sidebar.subheader("Filters")
        
        # Score filter
        min_score, max_score = st.sidebar.slider(
            "Final Score Range",
            0.0, 100.0, (0.0, 100.0)
        )
        
        # Verdict filter
        verdict_options = ["All", "High", "Medium", "Low"]
        selected_verdict = st.sidebar.selectbox("Verdict", verdict_options)
        
        # Apply filters
        filtered_df = df[
            (df['final_score'] >= min_score) &
            (df['final_score'] <= max_score)
        ]
        
        if selected_verdict != "All":
            filtered_df = filtered_df[filtered_df['verdict'] == selected_verdict]
        
        # Show filtered results count
        st.write(f"Showing {len(filtered_df)} of {len(df)} evaluations")
        
        # Display table
        st.dataframe(filtered_df[['id', 'final_score', 'verdict', 'timestamp']].style.format({
            'final_score': '{:.2f}%'
        }))
        
        # Show details for a selected evaluation
        selected_id = st.selectbox("Select evaluation to view details", filtered_df['id'].tolist() if not filtered_df.empty else [0], 
                                  format_func=lambda x: f"Evaluation {x}" if x != 0 else "Select an evaluation")
        
        if selected_id != 0:
            # Get the selected evaluation
            selected_eval = get_evaluation_by_id(selected_id)
            
            if selected_eval:
                st.subheader(f"Evaluation Details (ID: {selected_eval['id']})")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Hard Score", f"{selected_eval['hard_score']:.2f}%")
                with col2:
                    st.metric("Semantic Score", f"{selected_eval['semantic_score']:.2f}%")
                with col3:
                    st.metric("Final Score", f"{selected_eval['final_score']:.2f}%")
                
                st.write(f"**Verdict:** {selected_eval['verdict']}")
                st.write(f"**Timestamp:** {selected_eval['timestamp']}")
                
                if selected_eval['missing_skills']:
                    st.write("**Missing Skills:**")
                    st.write(", ".join(selected_eval['missing_skills']))
                
                if selected_eval.get('feedback'):
                    st.write("**Feedback:**")
                    st.info(selected_eval['feedback'])
                
                with st.expander("View Resume Text"):
                    st.text(selected_eval['resume_text'])
                
                with st.expander("View Job Description Text"):
                    st.text(selected_eval['jd_text'])
    else:
        st.info("No evaluations found in the database.")

elif page == "Student Upload":
    st.header("Student Resume Upload")
    
    st.write("""
    Upload multiple resumes and job descriptions to get relevance scores and verdicts for each.
    The system will analyze how well each resume matches the job description.
    """)
    
    # File uploaders for multiple resumes
    resume_files = st.file_uploader("Upload your resumes (PDF or DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)
    jd_text = st.text_area("Paste the job description")
    
    if st.button("Evaluate All") and resume_files and jd_text:
        results = []
        
        with st.spinner("Evaluating your resumes..."):
            try:
                # Process each resume
                for resume_file in resume_files:
                    # Parse resume
                    resume_text = parse_resume(resume_file)
                    
                    # Calculate scores
                    score_data = calculate_final_score(resume_text, jd_text)
                    
                    # Generate feedback
                    feedback = generate_feedback(resume_text, jd_text, score_data)
                    
                    results.append({
                        'filename': resume_file.name,
                        'hard_score': score_data['hard_score'],
                        'semantic_score': score_data['semantic_score'],
                        'final_score': score_data['total_score'],
                        'verdict': score_data['verdict'],
                        'missing_skills': score_data['missing_skills'],
                        'feedback': feedback
                    })
                
                # Display results
                st.success(f"Successfully evaluated {len(results)} resumes!")
                
                # Convert to DataFrame for display
                results_df = pd.DataFrame(results)
                results_df_display = results_df[['filename', 'hard_score', 'semantic_score', 'final_score', 'verdict']].copy()
                results_df_display.style.format({
                    'hard_score': '{:.2f}%',
                    'semantic_score': '{:.2f}%',
                    'final_score': '{:.2f}%'
                })
                
                st.dataframe(results_df_display.style.format({
                    'hard_score': '{:.2f}%',
                    'semantic_score': '{:.2f}%',
                    'final_score': '{:.2f}%'
                }))
                
                # Show detailed view
                selected_file = st.selectbox("Select a file to view detailed results", 
                                           results_df['filename'].tolist() if not results_df.empty else ["No files evaluated"])
                
                if selected_file != "No files evaluated":
                    selected_result = results_df[results_df['filename'] == selected_file].iloc[0]
                    
                    st.subheader(f"Detailed Results for {selected_file}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Hard Score", f"{selected_result['hard_score']:.2f}%")
                    with col2:
                        st.metric("Semantic Score", f"{selected_result['semantic_score']:.2f}%")
                    with col3:
                        st.metric("Final Score", f"{selected_result['final_score']:.2f}%")
                    
                    st.write(f"**Verdict:** {selected_result['verdict']}")
                    
                    if selected_result['missing_skills']:
                        st.write("**Missing Skills:**")
                        st.write(", ".join(selected_result['missing_skills']))
                    
                    st.write("**Feedback:**")
                    st.info(selected_result['feedback'])
                
            except Exception as e:
                st.error(f"An error occurred during evaluation: {str(e)}")

elif page == "History":
    st.header("Analysis History")
    
    # Add export button
    if st.button("Export as CSV"):
        csv_data = export_csv()
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="evaluations.csv",
            mime="text/csv"
        )
    
    if 'history' in st.session_state and st.session_state.history:
        # Convert history to DataFrame for better display
        df = pd.DataFrame(st.session_state.history)
        st.table(df)
    else:
        st.info("No analysis history yet. Upload and analyze a resume to see history here.")

# Footer
st.markdown("""
<div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-top: 30px; text-align: center;">
    <p>© 2023 Career Compass | AI-Powered Resume Analysis</p>
</div>
""", unsafe_allow_html=True)