import sqlite3
import json
from datetime import datetime
import os

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