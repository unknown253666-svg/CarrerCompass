# CareerCompass

CareerCompass is an Automated Resume Relevance Check System for placement teams and students. It evaluates resumes against job descriptions, provides scores, highlights missing skills, and gives verdicts (High/Medium/Low) with a Streamlit frontend and Flask + SQLite backend.

## Features

- **Resume Analysis**: Upload resumes (PDF/DOCX) and job descriptions for analysis
- **Relevance Scoring**: Get hard keyword match scores and semantic scores using AI
- **Verdict System**: Receive High/Medium/Low verdicts based on overall scores
- **Personalized Feedback**: Get AI-generated personalized improvement feedback
- **Dashboard**: Placement teams can view all evaluations and filter results
- **CSV Export**: Download evaluation results as CSV files

## Project Structure

```
CareerCompass/
├── backend/
│   ├── app.py              # Flask backend API
│   ├── database.py         # SQLite database initialization and helpers
│   ├── resume_parser.py    # PDF/DOCX text extraction
│   ├── scoring.py          # Scoring logic implementation and feedback generation
│   └── requirements.txt    # Backend dependencies
├── frontend/
│   ├── streamlit_app.py    # Main Streamlit application
│   ├── pages/
│   │   ├── student_upload.py     # Student resume upload page
│   │   └── dashboard.py          # Placement team dashboard
│   └── assets/
│       └── logo.png        # Application logo
├── .env                    # Environment variables
├── README.md               # Project documentation
├── .gitignore              # Git ignore file
└── tests/
    ├── test_resume_parser.py   # Tests for resume parsing
    └── test_scoring.py         # Tests for scoring logic
```

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: Flask
- **Database**: SQLite
- **API Integration**: Gemini API for semantic matching and feedback generation
- **File Processing**: pdfplumber, docx2txt

## Workflow

1. Placement team uploads job description (JD)
2. Students upload resumes
3. System extracts and parses text
4. Perform hard match (keywords, skills, education)
5. Perform semantic match using Gemini API
6. Compute weighted final score
7. Generate verdict (High / Medium / Low)
8. Provide improvement feedback
9. Store results in database
10. Frontend displays results and dashboard search/filter

## Setup

1. Clone the repository
2. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`:
   - `GEMINI_API_KEY`: Your Gemini API key
   - `DB_PATH`: Path to SQLite database file
4. Run the backend:
   ```bash
   cd backend
   python app.py
   ```
5. In a new terminal, run the frontend:
   ```bash
   cd frontend
   streamlit run streamlit_app.py
   ```

## API Endpoints

- `POST /upload`: Upload resume and job description for evaluation
- `GET /evaluations`: Get all evaluations
- `GET /evaluation/<id>`: Get a specific evaluation by ID
- `POST /feedback`: Generate personalized feedback

## Environment Variables

- `GEMINI_API_KEY`: API key for Gemini AI
- `DB_PATH`: Path to SQLite database file (default: career_compass.db)

## Database Schema

The application uses a single table for storing evaluations:

```
evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_text TEXT,
    jd_text TEXT,
    hard_score REAL,
    semantic_score REAL,
    final_score REAL,
    verdict TEXT,
    feedback TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```