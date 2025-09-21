import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from database import init_db, save_evaluation, get_all_evaluations, get_evaluation_by_id
from scoring import calculate_final_score, generate_feedback
from resume_parser import parse_resume
import os
import io
import csv
import spacy

# Download spaCy model if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Initialize database and logging
init_db()

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Enable CORS
CORS(app)

@app.route('/')
def home():
    """
    Default route to prevent 404 errors
    """
    return jsonify({
        "message": "CareerCompass API is running",
        "endpoints": {
            "upload_resume": "/upload_resume (POST)",
            "upload_jd": "/upload_jd (POST)",
            "evaluate": "/evaluate (POST)",
            "results": "/results (GET)",
            "result_by_id": "/results/<id> (GET)",
            "export_csv": "/export_csv (GET)"
        }
    })

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    """
    Endpoint for uploading resumes (supporting multiple files)
    """
    if 'resumes' not in request.files:
        return jsonify({'error': 'Missing resume files'}), 400

    resume_files = request.files.getlist('resumes')
    
    if not resume_files or all(f.filename == '' for f in resume_files):
        return jsonify({'error': 'No selected files'}), 400

    parsed_resumes = []
    for resume_file in resume_files:
        if not resume_file.filename.lower().endswith(('.pdf', '.docx')):
            return jsonify({'error': f'Unsupported file format for {resume_file.filename}'}), 400

        # Parse file
        try:
            resume_text = parse_resume(resume_file)
            parsed_resumes.append({
                'filename': resume_file.filename,
                'text': resume_text
            })
        except Exception as e:
            app.logger.error(f"Resume parsing error for {resume_file.filename}: {e}")
            return jsonify({'error': f'Error parsing resume {resume_file.filename}'}), 500

    return jsonify({
        'message': f'{len(parsed_resumes)} resume(s) uploaded successfully',
        'resumes': parsed_resumes
    })

@app.route('/upload_jd', methods=['POST'])
def upload_jd():
    """
    Endpoint for uploading job descriptions
    """
    if 'jd' not in request.files:
        return jsonify({'error': 'Missing job description file'}), 400

    jd_file = request.files['jd']
    
    if jd_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not jd_file.filename.lower().endswith(('.pdf', '.docx', '.txt')):
        return jsonify({'error': 'Unsupported file format'}), 400

    # Parse file
    try:
        jd_text = parse_resume(jd_file)
    except Exception as e:
        app.logger.error(f"JD parsing error: {e}")
        return jsonify({'error': 'Error parsing job description'}), 500

    return jsonify({
        'message': 'Job description uploaded successfully',
        'jd_text': jd_text
    })

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Endpoint for evaluating resume against job description
    """
    try:
        data = request.get_json()
        
        if not data or 'resume_text' not in data or 'jd_text' not in data:
            return jsonify({'error': 'Missing resume_text or jd_text'}), 400
        
        resume_text = data['resume_text']
        jd_text = data['jd_text']
        
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
        
        return jsonify({
            'message': 'Evaluation completed successfully',
            'hard_score': score_data['hard_score'],
            'semantic_score': score_data['semantic_score'],
            'final_score': score_data['total_score'],  # Using the correct key name
            'verdict': score_data['verdict'],
            'missing_skills': missing_skills,
            'feedback': feedback,
            'evaluation_id': evaluation_id
        })
    except Exception as e:
        app.logger.error(f"Evaluation error: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/evaluate_multiple', methods=['POST'])
def evaluate_multiple():
    """
    Endpoint for evaluating multiple resumes against a job description
    """
    try:
        data = request.get_json()
        
        if not data or 'resumes' not in data or 'jd_text' not in data:
            return jsonify({'error': 'Missing resumes or jd_text'}), 400
        
        jd_text = data['jd_text']
        resumes = data['resumes']  # List of resume texts with filenames
        
        results = []
        for resume in resumes:
            resume_text = resume['text']
            filename = resume['filename']
            
            # Calculate scores
            score_data = calculate_final_score(resume_text, jd_text)
            
            # Generate feedback
            feedback = generate_feedback(resume_text, jd_text, score_data)
            
            # Extract missing skills from score_data
            missing_skills = score_data['missing_skills']
            
            results.append({
                'filename': filename,
                'hard_score': score_data['hard_score'],
                'semantic_score': score_data['semantic_score'],
                'final_score': score_data['total_score'],
                'verdict': score_data['verdict'],
                'missing_skills': missing_skills,
                'feedback': feedback
            })
        
        # Save to database (optional - could save all or just best one)
        return jsonify({
            'message': 'Evaluations completed successfully',
            'results': results
        })
    except Exception as e:
        app.logger.error(f"Evaluation error: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

@app.route('/results', methods=['GET'])
def get_results():
    """
    Endpoint to get all evaluations
    """
    evaluations = get_all_evaluations()
    return jsonify(evaluations)

@app.route('/results/<int:evaluation_id>', methods=['GET'])
def get_result(evaluation_id):
    """
    Endpoint to get a specific evaluation by ID
    """
    evaluation = get_evaluation_by_id(evaluation_id)
    if evaluation:
        return jsonify(evaluation)
    else:
        return jsonify({'error': 'Evaluation not found'}), 404

@app.route('/export_csv', methods=['GET'])
def export_csv():
    """
    Endpoint to export results as CSV
    """
    evaluations = get_all_evaluations()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Evaluation ID', 'Resume Text', 'JD Text', 'Hard Score', 'Semantic Score', 'Final Score', 'Verdict', 'Missing Skills', 'Feedback'])
    for evaluation in evaluations:
        writer.writerow([
            evaluation['id'],
            evaluation['resume_text'],
            evaluation['jd_text'],
            evaluation['score_data']['hard_score'],
            evaluation['score_data']['semantic_score'],
            evaluation['score_data']['total_score'],
            evaluation['score_data']['verdict'],
            ', '.join(evaluation['missing_skills']),
            evaluation['feedback']
        ])
    output.seek(0)
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='evaluations.csv')

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Listen on all available network interfaces for Render deployment
    app.run(debug=False, host='0.0.0.0', port=port)
