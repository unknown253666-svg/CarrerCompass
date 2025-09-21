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

# Backend API URL - Updated to handle both local and cloud deployments
# For Streamlit Cloud deployment, you need to set the BACKEND_URL in the Streamlit Cloud settings
backend_url = os.getenv('BACKEND_URL', 'http://localhost:5000')

# Show backend URL for debugging
st.info(f"Backend URL: {backend_url}")

if st.button("Evaluate All") and resume_files and jd_file:
    if backend_url == 'http://localhost:5000':
        st.warning("You are using the default localhost backend URL. For cloud deployment, please set the BACKEND_URL environment variable in Streamlit Cloud settings.")
    
    results = []
    
    with st.spinner("Evaluating your resumes..."):
        try:
            # Upload the job description
            jd_files = {'jd': (jd_file.name, jd_file.getvalue())}
            jd_response = requests.post(f"{backend_url}/upload_jd", files=jd_files, timeout=30)
            
            if jd_response.status_code != 200:
                st.error(f"Error uploading job description: {jd_response.status_code} - {jd_response.text}")
                st.stop()
            
            jd_data = jd_response.json()
            jd_text = jd_data['jd_text']
            
            # Upload all resumes in a single request
            resume_files_data = [('resumes', (resume_file.name, resume_file.getvalue())) for resume_file in resume_files]
            resume_response = requests.post(f"{backend_url}/upload_resume", files=resume_files_data, timeout=30)
            
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
                eval_data = {
                    'resume_text': resume_text,
                    'jd_text': jd_text
                }
                
                eval_response = requests.post(f"{backend_url}/evaluate", json=eval_data, timeout=30)
                
                if eval_response.status_code == 200:
                    eval_result = eval_response.json()
                    st.metric("Total Score", f"{eval_result['total_score']}%")
                    st.metric("Hard Match Score", f"{eval_result['hard_score']}%")
                    st.metric("Semantic Score", f"{eval_result['semantic_score']}%")
                    st.markdown(f"**Verdict:** {eval_result['verdict']}")
                    
                    if eval_result['missing_skills']:
                        st.subheader("Missing Skills")
                        st.write(", ".join(eval_result['missing_skills']))
                else:
                    st.error(f"Error evaluating resume {resume_name}: {eval_response.status_code} - {eval_response.text}")
        
        except requests.exceptions.ConnectionError as e:
            st.error(f"Error connecting to backend: {str(e)}")
            st.error("Please make sure your backend is running and accessible.")
        except requests.exceptions.Timeout as e:
            st.error(f"Request to backend timed out: {str(e)}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

st.markdown("---")
st.subheader("Instructions:")
st.write("""
1. Make sure your backend is running and accessible.
2. For local deployment, the backend should run on http://localhost:5000
3. For cloud deployment, you need to deploy your backend to a cloud service and set the BACKEND_URL environment variable.
""")