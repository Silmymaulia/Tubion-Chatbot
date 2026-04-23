import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ===== CONFIG =====
INDEX_PATH = "vector_store/tbc_index.faiss"
CHUNKS_PATH = "vector_store/tbc_chunks.pkl"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CROSS_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

BI_THRESHOLD = 0.3
CROSS_THRESHOLD = 0.4
TOP_K = 10

# ===== LOAD MODEL =====
bi_encoder = SentenceTransformer(EMBED_MODEL)
cross_encoder = CrossEncoder(CROSS_MODEL)

# ===== LOAD INDEX =====
index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

def retrieve(query, verbose=False):
    query_emb = bi_encoder.encode([query])
    query_emb = np.array(query_emb).astype("float32")
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, TOP_K)

    candidates = []

    for score, idx in zip(scores[0], indices[0]):
        if score >= BI_THRESHOLD:
            candidates.append({
                "idx": idx,
                "text": chunks[idx],
                "bi_score": float(score)
            })

    if not candidates:
        return []

    # Re-ranking dengan cross encoder
    pairs = [(query, c["text"]) for c in candidates]
    cross_scores = cross_encoder.predict(pairs)

    for c, cs in zip(candidates, cross_scores):
        c["cross_score"] = float(cs)

    filtered = [c for c in candidates if c["cross_score"] >= CROSS_THRESHOLD]
    filtered = sorted(filtered, key=lambda x: x["cross_score"], reverse=True)

    results = []
    for rank, item in enumerate(filtered, start=1):
        results.append({
            "rank": rank,
            "idx": int(item["idx"]),
            "text": item["text"],
            "bi_score": item["bi_score"],
            "cross_score": item["cross_score"]
        })

    return results

def build_context(results, max_chunks=3):
    context_chunks = [r["text"] for r in results[:max_chunks]]
    return "\n\n".join(context_chunks)

def answer_query(query, verbose=False):
    results = retrieve(query)

    if not results:
        return "Maaf, tidak ditemukan informasi yang relevan."

    context = build_context(results)

    # Versi sederhana: langsung return konteks teratas
    # Nanti bisa diganti OpenAI / Ollama
    return context

if __name__ == "__main__":
    print("=== TEST RETRIEVE ===")
    results = retrieve("Berapa lama pengobatan TBC?")
    print(results)

    print("\n=== TEST ANSWER ===")
    answer = answer_query("Apa gejala TBC?")
    print(answer)