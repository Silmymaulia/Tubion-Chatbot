import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vector_store/tbc_index.faiss"
CHUNKS_PATH = "vector_store/tbc_chunks.pkl"

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

while True:
    query = input("\nMasukkan pertanyaan: ")

    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, 5)

    print("\nTop 5 hasil:\n")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {chunks[idx][:200]}")
        print("-" * 50)