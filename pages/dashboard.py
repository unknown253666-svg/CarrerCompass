import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime

st.header("Placement Dashboard")

st.write("""
View all resume evaluations, filter by score or student, and download results as CSV.
""")

# Backend API URL - Updated to handle both local and cloud deployments
# For Streamlit Cloud deployment, you need to set the BACKEND_URL in the Streamlit Cloud settings
backend_url = os.getenv('BACKEND_URL', 'http://localhost:5000')

# Show backend URL for debugging in local development
if backend_url == 'http://localhost:5000':
    st.info(f"Backend URL: {backend_url}")

# Fetch evaluations
try:
    # Add timeout for better connection handling
    response = requests.get(f"{backend_url}/results", timeout=30)
    
    if response.status_code == 200:
        evaluations = response.json()
        
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
                # We need to fetch individual evaluations to get the verdict
                # Since the list endpoint doesn't return verdicts, we'll filter client-side
                filtered_evaluations = []
                for _, row in filtered_df.iterrows():
                    try:
                        eval_detail_response = requests.get(f"{backend_url}/results/{row['id']}", timeout=10)
                        if eval_detail_response.status_code == 200:
                            eval_detail = eval_detail_response.json()
                            if eval_detail.get('verdict') == selected_verdict:
                                filtered_evaluations.append(row)
                    except requests.exceptions.RequestException as e:
                        st.warning(f"Error fetching details for evaluation {row['id']}: {str(e)}")
                
                filtered_df = pd.DataFrame(filtered_evaluations) if filtered_evaluations else pd.DataFrame()
            
            # Display metrics
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Evaluations", len(df))
            col2.metric("Average Score", f"{df['final_score'].mean():.1f}%")
            col3.metric("Highest Score", f"{df['final_score'].max():.1f}%")
            
            # Display filtered results
            st.subheader("Evaluations")
            
            if not filtered_df.empty:
                # Show the dataframe with basic info
                display_df = filtered_df[['id', 'hard_score', 'semantic_score', 'final_score', 'timestamp']].copy()
                display_df.columns = ['ID', 'Hard Score', 'Semantic Score', 'Final Score', 'Timestamp']
                
                # Add a verdict column
                def get_verdict(score):
                    if score >= 80:
                        return "High"
                    elif score >= 60:
                        return "Medium"
                    else:
                        return "Low"
                
                display_df['Verdict'] = display_df['Final Score'].apply(get_verdict)
                
                st.dataframe(display_df)
                
                # Option to view details of a specific evaluation
                st.subheader("Evaluation Details")
                selected_id = st.selectbox("Select an evaluation to view details", filtered_df['id'].tolist())
                
                if selected_id:
                    detail_response = requests.get(f"{backend_url}/results/{selected_id}", timeout=10)
                    if detail_response.status_code == 200:
                        detail_data = detail_response.json()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Hard Score", f"{detail_data['hard_score']:.1f}%")
                        with col2:
                            st.metric("Semantic Score", f"{detail_data['semantic_score']:.1f}%")
                        with col3:
                            st.metric("Final Score", f"{detail_data['final_score']:.1f}%")
                        with col4:
                            st.metric("Verdict", detail_data['verdict'])
                        
                        # Missing skills
                        if detail_data.get('missing_skills'):
                            st.subheader("Missing Skills")
                            for skill in detail_data['missing_skills']:
                                st.write(f"• {skill}")
                        
                        st.subheader("Personalized Feedback")
                        st.info(detail_data['feedback'])
                        
                        st.subheader("Resume Text")
                        with st.expander("Click to view resume text"):
                            st.text_area("Resume", detail_data['resume_text'], height=200, label_visibility="collapsed")
                        
                        st.subheader("Job Description Text")
                        with st.expander("Click to view job description text"):
                            st.text_area("Job Description", detail_data['jd_text'], height=200, label_visibility="collapsed")
            else:
                st.info("No evaluations match the current filters.")
            
            # Download CSV
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="career_compass_evaluations.csv",
                mime="text/csv"
            )
        else:
            st.info("No evaluations found. Upload some resumes to get started.")
    else:
        st.error(f"Error fetching evaluations: {response.status_code} - {response.text}")
        
except requests.exceptions.ConnectionError as e:
    st.error(f"Error connecting to backend: {str(e)}")
    st.error("Please make sure your backend is running and accessible.")
    if backend_url == 'http://localhost:5000':
        st.warning("You are using the default localhost backend URL. For cloud deployment, please set the BACKEND_URL environment variable in Streamlit Cloud settings.")
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