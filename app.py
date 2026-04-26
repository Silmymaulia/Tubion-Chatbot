from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# INIT
# =========================

app = FastAPI(title="RAG TBC API")

class Query(BaseModel):
    question: str

# =========================
# LOAD MODEL & DATA
# =========================

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("vector_store/tbc_index.faiss")

print("Loading chunks...")
with open("vector_store/tbc_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("API READY 🚀")

# =========================
# ENDPOINT
# =========================

@app.get("/")
def root():
    return {"message": "API running"}

# ✅ GET (untuk browser test)
@app.get("/chat")
def chat_get(q: str):
    return {"answer": f"(GET) Kamu bertanya: {q}"}

# ✅ POST (INI YANG DIPAKAI n8n + RAG)
@app.post("/chat")
def chat_post(data: Query):
    q = data.question

    # =========================
    # EMBEDDING
    # =========================
    query_vector = model.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)

    # =========================
    # RETRIEVAL
    # =========================
    distances, indices = index.search(query_vector, 3)

    results = [chunks[idx] for idx in indices[0]]

    # =========================
    # RESPONSE
    # =========================
    return {
        "question": q,
        "retrieved_context": results
    }