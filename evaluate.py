import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dataset_eval import dataset

# =========================
# CONFIG
# =========================

INDEX_PATH = "vector_store/tbc_index.faiss"
CHUNKS_PATH = "vector_store/tbc_chunks.pkl"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5

# =========================
# LOAD MODEL
# =========================

print("🔄 Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)

# =========================
# LOAD INDEX & CHUNKS
# =========================

print("📦 Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("📄 Loading chunks...")
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Total chunks in DB: {len(chunks)}")

# =========================
# METRICS STORAGE
# =========================

recall_scores = []
mrr_scores = []

# =========================
# EVALUATION LOOP
# =========================

print("\n🚀 Starting evaluation...\n")

for i, item in enumerate(dataset):

    query = item["question"]

    # Support single or multiple relevant_idx
    if isinstance(item["relevant_idx"], list):
        relevant_ids = item["relevant_idx"]
    else:
        relevant_ids = [item["relevant_idx"]]

    # Safety check (hindari IDX salah)
    relevant_ids = [idx for idx in relevant_ids if idx < len(chunks)]

    # Encode query
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)

    # Search FAISS
    distances, indices = index.search(query_vector, TOP_K)
    retrieved_ids = indices[0]

    # ---------------------
    # Recall@K
    # ---------------------

    hit = any(idx in relevant_ids for idx in retrieved_ids)
    recall_scores.append(1 if hit else 0)

    # ---------------------
    # MRR
    # ---------------------

    rank = 0
    for r, idx in enumerate(retrieved_ids):
        if idx in relevant_ids:
            rank = r + 1
            break

    if rank > 0:
        mrr_scores.append(1 / rank)
    else:
        mrr_scores.append(0)

    # ---------------------
    # Debug Output
    # ---------------------

    print(f"[{i+1}] Query: {query}")
    print(f"Retrieved IDs: {retrieved_ids}")
    print(f"Relevant IDs : {relevant_ids}")
    print(f"Hit: {hit}")
    print("-" * 60)

# =========================
# FINAL RESULTS
# =========================

avg_recall = np.mean(recall_scores)
avg_mrr = np.mean(mrr_scores)

print("\n==========================")
print("📊 FINAL EVALUATION RESULT")
print("==========================")
print(f"Recall@{TOP_K}: {avg_recall:.4f}")
print(f"MRR@{TOP_K}:    {avg_mrr:.4f}")
print("==========================")