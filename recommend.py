import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("shl_data.csv")

# Create combined text
df["combined"] = df["name"] + " " + df["description"] + " " + df["category"] + " " + df["job_level"]

# TF-IDF vectorizer (much lighter than sentence-transformers)
vectorizer = TfidfVectorizer()
corpus_matrix = vectorizer.fit_transform(df["combined"])

def recommend(query: str, top_k: int = 5):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, corpus_matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "name": row["name"],
            "description": row["description"],
            "category": row["category"],
            "job_level": row["job_level"],
            "duration_minutes": int(row["duration_minutes"]),
            "remote_testing": row["remote_testing"],
            "adaptive": row["adaptive"],
            "score": round(float(scores[idx]), 4)
        })
    return resultsgit