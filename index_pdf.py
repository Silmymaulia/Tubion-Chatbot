import pdfplumber
import re
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

# ===============================
# KONFIGURASI
# ===============================

BASE_PATH = r"F:\COOLYAHH\Skripsi\project_rag"
PDF_PATH = os.path.join(BASE_PATH, "Buku-Panduan-Tenaga-Medis-dan-Kesehatan-Tuberkulosis.pdf")
VECTOR_FOLDER = os.path.join(BASE_PATH, "vector_store")

# Hyperparameter chunking
CHUNK_SIZE = 512        # karakter per chunk
CHUNK_OVERLAP = 64      # overlap antar chunk untuk konteks

# Model — ganti dengan multilingual karena dokumen berbahasa Indonesia
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ===============================
# 1. LOAD PDF & EXTRACT TEXT
# ===============================

print("📄 Membaca PDF...")

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF tidak ditemukan: {PDF_PATH}")

pages_text = []
with pdfplumber.open(PDF_PATH) as pdf:
    for i, page in enumerate(tqdm(pdf.pages, desc="Ekstraksi halaman")):
        text = page.extract_text()
        if text:
            pages_text.append(text)

all_text = "\n".join(pages_text)
print(f"✅ Ekstraksi selesai. Total halaman: {len(pages_text)}")

# ===============================
# 2. CLEANING
# ===============================

print("🧹 Cleaning teks...")

# Hapus nomor halaman: — 45 —
all_text = re.sub(r'—\s*\d+\s*—', '', all_text)

# Hapus header/footer umum
all_text = re.sub(r'Kementerian Kesehatan Republik Indonesia', '', all_text, flags=re.IGNORECASE)

# Hapus baris yang hanya berisi angka (biasanya nomor halaman)
all_text = re.sub(r'(?m)^\s*\d+\s*$', '', all_text)

# Normalisasi whitespace
all_text = re.sub(r'\n{3,}', '\n\n', all_text)   # max 2 newline berturut-turut
all_text = re.sub(r'[ \t]+', ' ', all_text)        # spasi berlebih dalam baris

clean_text = all_text.strip()
print("✅ Cleaning selesai.")

# ===============================
# 3. CHUNKING DENGAN OVERLAP
# ===============================

def chunk_with_overlap(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """
    Split teks menjadi chunk dengan overlap untuk menjaga konteks
    antar chunk yang berdekatan.
    """
    # Pisah per kalimat dulu untuk tidak memotong di tengah kalimat
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sen_len = len(sentence)

        # Jika satu kalimat sudah melebihi chunk_size, potong paksa
        if sen_len > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            # Potong kalimat panjang
            for i in range(0, sen_len, chunk_size - overlap):
                chunks.append(sentence[i:i + chunk_size])
            continue

        if current_len + sen_len + 1 > chunk_size:
            chunks.append(" ".join(current_chunk))
            
            # Ambil kalimat terakhir sebagai overlap ke chunk berikutnya
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) < overlap:
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_sentences
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += sen_len + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Buang chunk terlalu pendek (noise)
    chunks = [c for c in chunks if len(c) > 30]
    return chunks


print("✂️  Membagi teks menjadi chunk...")
chunks = chunk_with_overlap(clean_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
print(f"✅ Total chunk: {len(chunks)}")

# ===============================
# 4. EMBEDDING
# ===============================

print(f"🔢 Membuat embedding dengan model: {MODEL_NAME}")

# paraphrase-multilingual lebih cocok untuk teks Indonesia
model = SentenceTransformer(MODEL_NAME)

# batch_size disesuaikan dengan RAM, show_progress_bar untuk monitoring
embeddings = model.encode(
    chunks,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)
embeddings = embeddings.astype("float32")

print(f"✅ Embedding selesai. Shape: {embeddings.shape}")

# ===============================
# 5. BUILD FAISS INDEX
# ===============================

print("🗃️  Membangun FAISS index...")

# Normalisasi untuk cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# IndexFlatIP = Inner Product → setelah normalisasi setara cosine similarity
# Untuk dataset besar (>100k chunk), ganti ke IndexIVFFlat atau HNSW
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"✅ Total vektor dalam index: {index.ntotal}")

# ===============================
# 6. SIMPAN VECTOR STORE
# ===============================

print("💾 Menyimpan Vector Database...")

os.makedirs(VECTOR_FOLDER, exist_ok=True)

faiss_path = os.path.join(VECTOR_FOLDER, "tbc_index.faiss")
chunks_path = os.path.join(VECTOR_FOLDER, "tbc_chunks.pkl")

faiss.write_index(index, faiss_path)

with open(chunks_path, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ SELESAI! Tersimpan di: {VECTOR_FOLDER}")
print(f"   - Index  : {faiss_path}")
print(f"   - Chunks : {chunks_path}")

# ===============================
# 7. QUICK SANITY CHECK
# ===============================

print("\n🔍 Sanity check — contoh retrieval:")

query = "gejala tuberkulosis"
query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
faiss.normalize_L2(query_vec)

scores, indices = index.search(query_vec, k=3)
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    print(f"\n[Top {rank}] Score: {score:.4f}")
    print(f"  {chunks[idx][:200]}...")