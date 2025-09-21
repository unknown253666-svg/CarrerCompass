# CareerCompass - AI-Powered Resume Analysis System

CareerCompass is an intelligent career planning assistant that analyzes the match between resumes and job descriptions using Natural Language Processing (NLP) techniques. It provides personalized career advice, skill gap analysis, and resume optimization suggestions.

## Features

- **Multi-format Resume Parsing**: Supports PDF and DOCX resume formats
- **Intelligent Matching**: Analyzes resumes against job descriptions using multiple NLP techniques
- **Skill Gap Analysis**: Identifies missing skills from job requirements
- **Scoring System**: Provides hard skills score, semantic similarity score, and overall matching score
- **Actionable Feedback**: Generates personalized suggestions for resume improvement
- **Batch Processing**: Process multiple resumes against a single job description
- **Historical Tracking**: View and filter past evaluations
- **Data Export**: Export evaluation results as CSV

## Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - For building interactive web interface
- **NLP Libraries**: 
  - [spaCy](https://spacy.io/) - For natural language processing
  - [NLTK](https://www.nltk.org/) - For text processing
  - [sentence-transformers](https://www.sbert.net/) - For semantic similarity calculations
  - [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) - For fuzzy string matching
- **Document Processing**: 
  - [pdfplumber](https://github.com/jsvine/pdfplumber) - For PDF text extraction
  - [docx2txt](https://github.com/ankushshah89/python-docx2txt) - For DOCX text extraction
- **Data Processing**: [pandas](https://pandas.pydata.org/) - For data manipulation
- **Database**: SQLite - For local data storage

## Application Structure

```
├── streamlit_app.py          # Main application file
├── shared_utils.py           # Shared functions used by main app and pages
├── pages/                    # Directory containing additional pages
│   ├── student_upload.py     # Batch processing page
│   └── dashboard.py          # Results dashboard
├── career_compass.db         # SQLite database file
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/unknown253666-svg/CarrerCompass.git
   cd CarrerCompass
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Running Locally

1. **Start the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the application**:
   Open your browser and go to `http://localhost:8501`

### Application Pages

1. **Resume Analysis**: Upload a single resume and paste a job description to get an analysis
2. **Student Upload**: Upload multiple resumes and a job description for batch processing
3. **Placement Dashboard**: View and filter historical evaluations
4. **History**: Track your analysis history

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Select the main branch
4. Streamlit will automatically detect and run `streamlit_app.py`

### Requirements for Deployment

- All functionality is self-contained with no external dependencies
- No backend services required
- All processing happens within the Streamlit application

## How It Works

1. **File Upload**: User uploads resume(s) and job description
2. **Text Extraction**: Application parses files using pdfplumber/docx2txt
3. **NLP Processing**: Text is processed with spaCy and sentence-transformers
4. **Matching Analysis**: Calculates multiple scores:
   - Hard skills matching (keyword analysis)
   - Semantic similarity (meaning-based matching)
   - Fuzzy string matching (flexible text comparison)
5. **Feedback Generation**: Provides actionable suggestions for resume improvement
6. **Data Storage**: Results are stored in SQLite database
7. **Results Display**: Scores and feedback are shown in the web interface

## Development

### Code Structure

- `streamlit_app.py`: Main application with routing logic
- `shared_utils.py`: Contains all core business logic and functions
- `pages/`: Contains additional pages for specific functionality
- `career_compass.db`: SQLite database for storing evaluation results

### Adding New Features

1. Modify the appropriate files in the codebase
2. Test locally with `streamlit run streamlit_app.py`
3. Commit and push changes to GitHub

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all the open-source libraries that made this project possible
- Special thanks to the NLP community for their excellent tools and models