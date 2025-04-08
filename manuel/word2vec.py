import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec
import numpy as np

# Load the resume CSV file
resume_df = pd.read_csv("resume_queries.csv")  # change path if needed

def simple_preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return tokens

# Apply preprocessing to each resume
resume_df['tokens'] = resume_df['Resume_str'].apply(simple_preprocess)

# Train a Word2Vec model on the resume tokens
token_lists = resume_df['tokens'].tolist()
w2v_model = Word2Vec(sentences=token_lists, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Create embeddings by averaging word vectors in each resume
def embed_resume(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Generate and store the embeddings
resume_df['embedding'] = resume_df['tokens'].apply(lambda tokens: embed_resume(tokens, w2v_model))

resume_df = resume_df.drop(['Resume_html', 'Resume_str', 'Category'], axis=1)


resume_df.to_csv('output.csv', index=None)
# (Optional) Save embeddings for later use
np.save("resume_embeddings.npy", np.vstack(resume_df['embedding'].values))
