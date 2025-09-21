import pdfplumber
import docx2txt
import tempfile
import os
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not installed, we'll use a basic approach
    nlp = None

def parse_resume(file):
    """
    Parse resume file and extract text content
    
    Args:
        file: File object (PDF or DOCX)
        
    Returns:
        str: Extracted text content
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        file.save(tmp_file.name)
        tmp_filename = tmp_file.name
    
    try:
        # Check file extension
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(tmp_filename)
        elif file.filename.lower().endswith(('.docx', '.doc')):
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

def extract_skills(resume_text, jd_skills=None):
    """
    Extract skills from resume text
    
    Args:
        resume_text (str): Normalized resume text
        jd_skills (list): List of skills from job description (optional)
        
    Returns:
        list: Extracted skills
    """
    # Common skill keywords
    common_skills = [
        'python', 'java', 'javascript', 'c++', 'sql', 'html', 'css', 'react', 'angular', 'vue',
        'node.js', 'express', 'django', 'flask', 'spring', 'mongodb', 'postgresql', 'mysql',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'tensorflow', 'pytorch',
        'machine learning', 'deep learning', 'data analysis', 'data science', 'nlp',
        'computer vision', 'api', 'rest', 'graphql', 'testing', 'agile', 'scrum'
    ]
    
    # If we have skills from JD, prioritize those
    if jd_skills:
        skills_to_check = jd_skills + common_skills
    else:
        skills_to_check = common_skills
    
    found_skills = []
    for skill in skills_to_check:
        if skill.lower() in resume_text.lower():
            found_skills.append(skill)
    
    return list(set(found_skills))

def extract_education(resume_text):
    """
    Extract education information from resume text
    
    Args:
        resume_text (str): Normalized resume text
        
    Returns:
        list: Extracted education information
    """
    education_keywords = [
        'bachelor', 'master', 'phd', 'degree', 'university', 'college', 'institute',
        'bs', 'ms', 'ma', 'ba', 'b.tech', 'm.tech', 'b.e', 'm.e'
    ]
    
    sentences = re.split(r'[.\n]', resume_text)
    education_info = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in education_keywords):
            education_info.append(sentence.strip())
    
    return education_info

def extract_experience(resume_text):
    """
    Extract experience information from resume text
    
    Args:
        resume_text (str): Normalized resume text
        
    Returns:
        list: Extracted experience information
    """
    experience_keywords = [
        'experience', 'worked', 'developed', 'managed', 'led', 'created', 'implemented',
        'designed', 'built', 'improved', 'optimized', 'collaborated', 'contributed'
    ]
    
    sentences = re.split(r'[.\n]', resume_text)
    experience_info = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in experience_keywords):
            experience_info.append(sentence.strip())
    
    return experience_info