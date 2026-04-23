import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from dataset_eval import EVAL_DATA

# =========================
# CONFIG
# =========================

INDEX_PATH = "vector_store/tbc_index.faiss"
CHUNKS_PATH = "vector_store/tbc_chunks.pkl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FAISS_TOP_K = 10
FINAL_TOP_K = 5

# =========================
# LOAD MODELS
# =========================

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)

print("Loading cross encoder...")
cross_encoder = CrossEncoder(CROSS_MODEL)

# =========================
# LOAD DATA
# =========================

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# =========================
# METRICS STORAGE
# =========================

baseline_mrr = []
rerank_mrr = []

baseline_recall = []
rerank_recall = []

# =========================
# EVALUATION LOOP
# =========================

for item in EVAL_DATA:

    query = item["query"]
    relevant_ids = item["relevant_chunk_ids"]

    # ======================
    # BASELINE (FAISS only)
    # ======================

    q_vec = embed_model.encode([query])
    q_vec = np.array(q_vec).astype("float32")

    _, indices = index.search(q_vec, FAISS_TOP_K)
    retrieved = indices[0][:FINAL_TOP_K]

    # Recall
    hit = any(idx in relevant_ids for idx in retrieved)
    baseline_recall.append(1 if hit else 0)

    # MRR
    rank = None
    for i, idx in enumerate(retrieved):
        if idx in relevant_ids:
            rank = i + 1
            break

    baseline_mrr.append(1 / rank if rank else 0)

    # ======================
    # RERANKING
    # ======================

    # Ambil kandidat top 10 dulu
    candidate_ids = indices[0]
    candidate_texts = [chunks[i] for i in candidate_ids]

    # Buat pasangan (query, chunk)
    pairs = [(query, text) for text in candidate_texts]

    scores = cross_encoder.predict(pairs)

    # Urutkan berdasarkan skor
    reranked = sorted(
        zip(candidate_ids, scores),
        key=lambda x: x[1],
        reverse=True
    )

    reranked_ids = [x[0] for x in reranked[:FINAL_TOP_K]]

    # Recall
    hit = any(idx in relevant_ids for idx in reranked_ids)
    rerank_recall.append(1 if hit else 0)

    # MRR
    rank = None
    for i, idx in enumerate(reranked_ids):
        if idx in relevant_ids:
            rank = i + 1
            break

    rerank_mrr.append(1 / rank if rank else 0)

# =========================
# RESULT
# =========================

print("\n===== HASIL EVALUASI =====\n")

print("Baseline FAISS")
print(f"Recall@5 : {np.mean(baseline_recall):.4f}")
print(f"MRR@5    : {np.mean(baseline_mrr):.4f}")

print("\nFAISS + CrossEncoder")
print(f"Recall@5 : {np.mean(rerank_recall):.4f}")
print(f"MRR@5    : {np.mean(rerank_mrr):.4f}")