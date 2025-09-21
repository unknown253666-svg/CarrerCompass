import streamlit as st
import requests
import os

st.header("Student Resume Upload")

st.write("""
Upload multiple resumes and job descriptions to get relevance scores and verdicts for each.
The system will analyze how well each resume matches the job description.
""")

# File uploaders for multiple resumes
resume_files = st.file_uploader("Upload your resumes (PDF or DOCX)", type=['pdf', 'docx'], accept_multiple_files=True)
jd_file = st.file_uploader("Upload the job description (PDF/DOCX)", type=['pdf', 'docx'])

# Backend API URL
backend_url = os.getenv('BACKEND_URL', 'http://localhost:5000')

if st.button("Evaluate All") and resume_files and jd_file:
    results = []
    
    with st.spinner("Evaluating your resumes..."):
        try:
            # Upload the job description
            jd_files = {'jd': (jd_file.name, jd_file.getvalue())}
            jd_response = requests.post(f"{backend_url}/upload_jd", files=jd_files)
            
            if jd_response.status_code != 200:
                st.error(f"Error uploading job description: {jd_response.status_code} - {jd_response.text}")
                st.stop()
            
            jd_data = jd_response.json()
            jd_text = jd_data['jd_text']
            
            # Upload all resumes in a single request
            resume_files_data = [('resumes', (resume_file.name, resume_file.getvalue())) for resume_file in resume_files]
            resume_response = requests.post(f"{backend_url}/upload_resume", files=resume_files_data)
            
            if resume_response.status_code != 200:
                st.error(f"Error uploading resumes: {resume_response.status_code} - {resume_response.text}")
                st.stop()
            
            resume_data = resume_response.json()
            parsed_resumes = resume_data['resumes']
            
            # Process each resume
            for i, resume_info in enumerate(parsed_resumes):
                resume_name = resume_info['filename']
                resume_text = resume_info['text']
                
                st.markdown(f"### Processing Resume {i+1}: {resume_name}")
                
                # Evaluate the resume against the job description
                evaluation_data = {
                    'resume_text': resume_text,
                    'jd_text': jd_text
                }
                
                evaluation_response = requests.post(f"{backend_url}/evaluate", json=evaluation_data)
                
                if evaluation_response.status_code == 200:
                    result = evaluation_response.json()
                    result['resume_name'] = resume_name
                    results.append(result)
                    st.success(f"Resume {resume_name} evaluated successfully!")
                else:
                    st.error(f"Error evaluating resume {resume_name}: {evaluation_response.status_code} - {evaluation_response.text}")
            
            # Display results for all resumes
            if results:
                st.header("Evaluation Results")
                
                # Summary table
                st.subheader("Summary")
                summary_data = []
                for result in results:
                    summary_data.append({
                        "Resume Name": result['resume_name'],
                        "Final Score": f"{result['final_score']:.1f}%",
                        "Verdict": result['verdict']
                    })
                
                st.table(summary_data)
                
                # Detailed results for each resume
                st.subheader("Detailed Results")
                for i, result in enumerate(results):
                    with st.expander(f"{result['resume_name']} - Final Score: {result['final_score']:.1f}%"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Final Score", f"{result['final_score']:.1f}%")
                        
                        with col2:
                            st.metric("Verdict", result['verdict'])
                        
                        with col3:
                            st.metric("Evaluation ID", result['evaluation_id'])
                        
                        # Detailed scores
                        st.subheader("Detailed Scores")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Hard Match Score:** {result['hard_score']:.1f}%")
                        
                        with col2:
                            st.write(f"**Semantic Score:** {result['semantic_score']:.1f}%")
                        
                        # Verdict with styling
                        st.subheader("Verdict")
                        if result['verdict'] == "High":
                            st.success("High Match - Your resume matches well with the job description. You have a good chance of being shortlisted.")
                        elif result['verdict'] == "Medium":
                            st.warning("Medium Match - Your resume has a moderate match with the job description. Consider tailoring it more specifically to the job.")
                        else:
                            st.error("Low Match - Your resume has a low match with the job description. Consider revising it to better align with the job requirements.")
                        
                        # Missing skills
                        if result.get('missing_skills'):
                            st.subheader("Missing Skills")
                            st.write("The following skills were found in the job description but are missing from your resume:")
                            for skill in result['missing_skills']:
                                st.write(f"• {skill}")
                        
                        # Feedback section
                        st.subheader("Personalized Feedback")
                        st.info(result['feedback'])
                    
            else:
                st.error("No resumes were successfully evaluated.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload at least one resume and one job description to get started.")

st.info("Note: Make sure the backend server is running at http://localhost:5000")