import streamlit as st
import pandas as pd
import re

# Constants
NUM_RESUMES = 20
JOBS_PER_RESUME = 10
TOTAL_EVALUATIONS = NUM_RESUMES * JOBS_PER_RESUME
CSV_PATH = "4096_LongFormer_Tokenizer_BM25_matches.csv"
OUTPUT_CSV = "TEst.csv"

# Load data into session state only once
if 'bert_df' not in st.session_state:
    df = pd.read_csv(CSV_PATH)

    if 'evaluation' not in df.columns:
        df['evaluation'] = -1
    st.session_state.bert_df = df

if 'current_job' not in st.session_state:
    st.session_state.current_job = 0

# make the layout wide
st.set_page_config(layout="wide")

bert_df = st.session_state.bert_df
job_idx = st.session_state.current_job
resume_idx = job_idx // JOBS_PER_RESUME

# Stop if evaluation is finished
if job_idx >= TOTAL_EVALUATIONS:
    st.success("🎉 All evaluations completed!")
    st.write("Saving results to CSV...")
    bert_df.to_csv(OUTPUT_CSV, index=False)
    st.download_button("Download CSV", data=bert_df.to_csv(index=False), file_name=OUTPUT_CSV, mime="text/csv")
    st.stop()

# UI
st.title("Resume–Job Relevance Evaluator")

# Progress Bar
st.progress(job_idx / TOTAL_EVALUATIONS, text=f"{job_idx + 1} / {TOTAL_EVALUATIONS} evaluated")

# Extract resume summary
resume_text = bert_df.loc[resume_idx * JOBS_PER_RESUME, 'Resume_str']
match = re.search(r'Summary(.*?)Highlights', resume_text, re.DOTALL)
summary_text = match.group(1).strip() if match else "No summary found between keywords."

# Show resume and job
st.subheader(f"Resume {resume_idx + 1} Summary")
st.markdown(f"<div style='white-space: pre-wrap; word-wrap: break-word; font-size: 16px'>{summary_text}</div>", unsafe_allow_html=True)

st.subheader(f"Job Match {job_idx % JOBS_PER_RESUME + 1}")
job_desc = bert_df.loc[job_idx, 'Job_posting']
st.markdown(f"<div style='white-space: pre-wrap; word-wrap: break-word; font-size: 16px'>{job_desc}</div>", unsafe_allow_html=True)

# Margin before buttons
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

# Centered buttons
col1, col2, col3 = st.columns([3, 1, 1])

with col2:
    if st.button("✅ Relevant"):
        bert_df.at[job_idx, 'evaluation'] = 1
        st.session_state.current_job += 1
        st.rerun()

with col3:
    if st.button("❌ Not Relevant"):
        bert_df.at[job_idx, 'evaluation'] = 0
        st.session_state.current_job += 1
        st.rerun()