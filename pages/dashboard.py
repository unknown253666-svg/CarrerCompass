import streamlit as st
import pandas as pd
import os
import sys

# Add the parent directory to the path so we can import functions from streamlit_app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the main app
try:
    from streamlit_app import (
        get_all_evaluations,
        get_evaluation_by_id
    )
except ImportError as e:
    st.error(f"Error importing functions from main app: {e}")
    st.stop()

st.header("Placement Dashboard")

st.write("""
View all resume evaluations, filter by score or student, and download results as CSV.
""")

# Fetch evaluations
evaluations = get_all_evaluations()

if evaluations:
    # Convert to DataFrame
    df = pd.DataFrame(evaluations)
    
    # Format timestamp if it exists
    if 'timestamp' in df.columns:
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
        selected_eval = get_evaluation_by_id(selected_id)
        
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
            if 'timestamp' in selected_eval:
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

st.sidebar.markdown("---")
st.sidebar.info("""
**Instructions:**
1. Upload resumes and job descriptions using the "Student Upload" page
2. Come back to this dashboard to view results
3. Filter results by score or verdict
""")