import streamlit as st
import pandas as pd
import os
import sys

# Add the parent directory to the path so we can import functions from shared_utils.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the shared utilities module
try:
    import shared_utils
except ImportError as e:
    st.error(f"Error importing functions from shared utilities: {e}")
    st.stop()

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

st.info("""
**Instructions:**
1. Upload multiple resumes (PDF or DOCX format)
2. Upload one job description (PDF or DOCX format)
3. Click "Evaluate All" to process all resumes against the job description
4. View results in the table and detailed views
5. Check the "Placement Dashboard" to see all evaluations
""")