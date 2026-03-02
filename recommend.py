import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

df = pd.read_csv("shl_data.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

df["combined"] = df["name"] + " " + df["description"] + " " + df["category"] + " " + df["job_level"]
corpus_embeddings = model.encode(df["combined"].tolist(), convert_to_tensor=True)

def recommend(query: str, top_k: int = 5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = scores.topk(k=min(top_k, len(df)))

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        row = df.iloc[idx.item()]
        results.append({
            "name": row["name"],
            "description": row["description"],
            "category": row["category"],
            "job_level": row["job_level"],
            "duration_minutes": int(row["duration_minutes"]),
            "remote_testing": row["remote_testing"],
            "adaptive": row["adaptive"],
            "score": round(score.item(), 4)
        })
    return results