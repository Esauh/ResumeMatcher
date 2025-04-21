resume_index = 0
job_index = 0
total_jobs = len(bert_df)
jobs_per_resume = 10
total_resumes = total_jobs // jobs_per_resume

progress_bar = widgets.IntProgress(value=0, min=0, max=total_jobs, description='Progress:', bar_style='info')

def get_df_idx():
    return resume_index * jobs_per_resume + job_index

def render():
    clear_output()

    if resume_index >= total_resumes:
        progress_bar.value = total_jobs
        display(progress_bar)
        display(HTML("<h3 style='color: green;'>✅ Evaluation complete. CSV saved as <code>manual_evaluation_done.csv</code>.</h3>"))
        return

    df_idx = get_df_idx()

    resume_text = bert_df.loc[df_idx, 'resume_text']
    job_text = bert_df.loc[df_idx, 'job_matching']
    bm25_score = bert_df.loc[df_idx, 'bm25_score']

    progress_bar.value = df_idx
    display(progress_bar)

    display(HTML(f"""
    <div style="white-space: pre-wrap; font-size: 18px; border: 2px solid #ccc; padding: 10px; margin-top: 10px; margin-bottom: 10px;">
        <b>Resume {resume_index + 1}:</b><br><br>{resume_text}
    </div>
    """))

    display(HTML(f"""
    <div style="white-space: pre-wrap; font-size: 16px; border: 1px solid #aaa; padding: 8px; margin-bottom: 10px;">
        <b>Job Match {job_index + 1} of 10 (BM25 Score: {bm25_score:.2f}):</b><br><br>{job_text}
    </div>
    """))

    relevant_button.disabled = False
    not_relevant_button.disabled = False
    display(widgets.HBox([relevant_button, not_relevant_button]))

def save_eval_and_advance(score):
    global job_index, resume_index

    df_idx = get_df_idx()
    bert_df.at[df_idx, 'eval_score'] = score

    job_index += 1
    if job_index >= jobs_per_resume:
        job_index = 0
        resume_index += 1

    if resume_index >= total_resumes:
        bert_df.to_csv("manual_evaluation_done.csv", index=False)

    render()

def on_relevant_clicked(b):
    relevant_button.disabled = True
    not_relevant_button.disabled = True
    save_eval_and_advance(1)

def on_not_relevant_clicked(b):
    relevant_button.disabled = True
    not_relevant_button.disabled = True
    save_eval_and_advance(0)

relevant_button = widgets.Button(description="👍 Relevant", button_style='success')
not_relevant_button = widgets.Button(description="👎 Not Relevant", button_style='danger')

relevant_button.on_click(on_relevant_clicked)
not_relevant_button.on_click(on_not_relevant_clicked)

render()
