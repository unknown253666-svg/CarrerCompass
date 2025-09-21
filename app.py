import streamlit as st
import os
from dotenv import load_dotenv
from scoring import calculate_total_score

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="Resume Relevance Checker", layout="wide")

st.title("Automated Resume Relevance Check System")
st.write("Upload your resume and a job description to get a relevance score, missing skills, and verdict.")

# File upload
resume_file = st.file_uploader("Upload your Resume (PDF/DOCX)", type=["pdf","docx"])
jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf","docx"])

def extract_text_from_file(file):
    """Extract text from PDF or DOCX"""
    import pdfplumber
    import docx2txt
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
        return "\n".join(pages)
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return ""

if resume_file and jd_file:
    resume_text = extract_text_from_file(resume_file)
    jd_text = extract_text_from_file(jd_file)
    
    if not resume_text.strip() or not jd_text.strip():
        st.error("Failed to extract text from uploaded files.")
    else:
        st.info("Evaluating...")
        results = calculate_total_score(resume_text, jd_text, gemini_api_key=GEMINI_API_KEY)
        
        # Display results
        st.subheader("Evaluation Results")
        st.metric("Hard Match Score", f"{results['hard_score']}%")
        st.metric("Semantic Score", f"{results['semantic_score']}%")
        st.metric("Total Score", f"{results['total_score']}%")
        st.markdown(f"**Verdict:** {results['verdict']}")
        
        st.subheader("Missing Skills")
        if results['missing_skills']:
            st.write(", ".join(results['missing_skills']))
        else:
            st.write("No missing skills found. Good match!")
        
        st.subheader("Personalized Feedback")
        feedback = f"""
        Your resume has a {results['verdict']} match with the job description.
        Focus on aligning your skills and experiences more closely with the job requirements.
        Pay attention to missing skills: {', '.join(results['missing_skills']) if results['missing_skills'] else 'None'}.
        """
        st.write(feedback)