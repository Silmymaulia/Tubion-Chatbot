# inspect_chunks.py

import pickle

with open("vector_store/tbc_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

for i, chunk in enumerate(chunks):
    print(f"\n===== IDX {i} =====")
    print(chunk)