from fastapi import FastAPI
from pydantic import BaseModel
from recommend import recommend

app = FastAPI(title="SHL Assessment Recommendation Engine")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def home():
    return {"message": "SHL Assessment Recommendation Engine is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def get_recommendations(request: QueryRequest):
    results = recommend(request.query, request.top_k)
    return {
        "query": request.query,
        "recommendations": results
    }

