import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime

st.header("Placement Dashboard")

st.write("""
View all resume evaluations, filter by score or student, and download results as CSV.
""")

# Backend API URL
backend_url = os.getenv('BACKEND_URL', 'http://localhost:5000')

# Fetch evaluations
try:
    response = requests.get(f"{backend_url}/results")
    
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
                    eval_detail_response = requests.get(f"{backend_url}/results/{row['id']}")
                    if eval_detail_response.status_code == 200:
                        eval_detail = eval_detail_response.json()
                        if eval_detail.get('verdict') == selected_verdict or selected_verdict == "All":
                            filtered_evaluations.append(row)
                
                filtered_df = pd.DataFrame(filtered_evaluations) if filtered_evaluations else pd.DataFrame()
            
            # Display metrics
            st.subheader("Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Evaluations", len(df))
            
            with col2:
                high_count = len(df[df['final_score'] >= 80])
                st.metric("High Matches", high_count)
            
            with col3:
                medium_count = len(df[(df['final_score'] >= 60) & (df['final_score'] < 80)])
                st.metric("Medium Matches", medium_count)
            
            with col4:
                low_count = len(df[df['final_score'] < 60])
                st.metric("Low Matches", low_count)
            
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
                    detail_response = requests.get(f"{backend_url}/results/{selected_id}")
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
        
except Exception as e:
    st.error(f"Error connecting to backend: {str(e)}")
    st.info("Make sure the backend server is running at http://localhost:5000")