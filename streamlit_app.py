import streamlit as st
import pandas as pd
import os
from datetime import datetime
import json
import io
import csv

# Import shared utilities
import shared_utils

# Initialize database
shared_utils.init_db()

# Set page configuration
st.set_page_config(
    page_title="Career Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom HTML for header
st.markdown("""
<div style="background-color: #000000; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: #ffffff;">🧭 Career Compass</h1>
    <p style="color: #ffffff;">Upload your resume and get AI-powered insights to optimize your job application success.</p>
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
                    resume_text = shared_utils.parse_resume(resume_file)
                    
                    # Calculate scores
                    score_data = shared_utils.calculate_final_score(resume_text, jd_text)
                    
                    # Generate feedback
                    feedback = shared_utils.generate_feedback(resume_text, jd_text, score_data)
                    
                    # Extract missing skills from score_data
                    missing_skills = score_data['missing_skills']
                    
                    # Save to database
                    evaluation_id = shared_utils.save_evaluation(
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

elif page == "Student Upload":
    st.header("Student Resume Upload")
    
    st.write("""
    Upload multiple resumes and job descriptions to get relevance scores and verdicts for each.
    The system will analyze how well each resume matches the job description.
    """)
    
    # File uploaders for multiple resumes
    resume_files = st.file_uploader("Upload your resumes (PDF or DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)
    jd_file = st.file_uploader("Upload the job description (PDF/DOCX)", type=['pdf', 'docx'])
    
    if st.button("Evaluate All") and resume_files and jd_file:
        results = []
        
        with st.spinner("Evaluating your resumes..."):
            try:
                # Parse job description
                jd_text = shared_utils.parse_resume(jd_file)
                
                # Process each resume
                for resume_file in resume_files:
                    # Parse resume
                    resume_text = shared_utils.parse_resume(resume_file)
                    
                    # Calculate scores
                    score_data = shared_utils.calculate_final_score(resume_text, jd_text)
                    
                    # Generate feedback
                    feedback = shared_utils.generate_feedback(resume_text, jd_text, score_data)
                    
                    # Save to database
                    evaluation_id = shared_utils.save_evaluation(
                        resume_text=resume_text,
                        jd_text=jd_text,
                        score_data=score_data,
                        feedback=feedback,
                        missing_skills=score_data['missing_skills']
                    )
                    
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

elif page == "Placement Dashboard":
    st.header("Placement Dashboard")
    
    st.write("""
    View all resume evaluations, filter by score or student, and download results as CSV.
    """)
    
    # Fetch evaluations
    evaluations = shared_utils.get_all_evaluations()
    
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
            selected_eval = shared_utils.get_evaluation_by_id(selected_id)
            
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

elif page == "History":
    st.header("Analysis History")
    
    # Add export button
    if st.button("Export as CSV"):
        csv_data = shared_utils.export_csv()
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
<div style="background-color: #000000; padding: 15px; border-radius: 10px; margin-top: 30px; text-align: center;">
    <p style="color: #ffffff;">© 2023 Career Compass | AI-Powered Resume Analysis</p>
</div>
""", unsafe_allow_html=True)