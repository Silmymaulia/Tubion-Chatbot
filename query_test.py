import os
import sys
import faiss
import pickle
import numpy as np
from enum import Enum
from sentence_transformers import SentenceTransformer, CrossEncoder

# ===============================
# KONFIGURASI
# ===============================

BASE_PATH = r"F:\COOLYAHH\Skripsi\project_rag"
VECTOR_FOLDER = os.path.join(BASE_PATH, "vector_store")

MODEL_NAME            = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_NAME         = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
TOP_K                 = 10    # kandidat awal dari FAISS
TOP_N_RERANK          = 3     # hasil final setelah re-ranking
BI_SCORE_THRESHOLD    = 0.30  # minimum cosine similarity (bi-encoder)
CROSS_SCORE_THRESHOLD = 0.40  # minimum skor cross-encoder setelah re-ranking
CHUNK_MAX_CHARS       = 200   # batas karakter per chunk

# ===============================
# INTENT DETECTION
# ===============================

class QueryIntent(Enum):
    DURASI      = "durasi"
    GEJALA      = "gejala"
    PENGOBATAN  = "pengobatan"
    PENULARAN   = "penularan"
    MDR_TB      = "mdr_tb"
    DIAGNOSIS   = "diagnosis"
    UMUM        = "umum"

INTENT_KEYWORDS: dict[QueryIntent, list[str]] = {
    QueryIntent.DURASI      : ["lama", "berapa lama", "durasi", "minggu", "bulan", "hari", "berapa hari", "berapa minggu", "berapa bulan"],
    QueryIntent.GEJALA      : ["gejala", "tanda", "ciri", "keluhan", "symptom", "tanda-tanda"],
    QueryIntent.PENGOBATAN  : ["obat", "pengobatan", "terapi", "minum obat", "regimen", "sembuh", "penyembuhan"],
    QueryIntent.PENULARAN   : ["menular", "penularan", "menyebar", "transmisi", "kontak", "tertular"],
    QueryIntent.MDR_TB      : ["mdr", "resistan", "kebal", "resistensi", "mdr-tb", "xdr"],
    QueryIntent.DIAGNOSIS   : ["diagnosis", "diagnosa", "deteksi", "tes", "uji", "pemeriksaan", "foto", "rontgen", "dahak"],
}

# Keyword wajib hadir dalam chunk untuk setiap intent.
# Chunk yang tidak mengandung SATUPUN keyword ini akan dibuang
# sebelum masuk ke cross-encoder.
INTENT_FILTER_KEYWORDS: dict[QueryIntent, list[str]] = {
    QueryIntent.DURASI      : ["bulan", "minggu", "hari", "lama", "durasi"],
    QueryIntent.GEJALA      : ["gejala", "tanda", "batuk", "demam", "keringat", "keluhan"],
    QueryIntent.PENGOBATAN  : ["obat", "terapi", "pengobatan", "regimen", "dosis"],
    QueryIntent.PENULARAN   : ["menular", "penularan", "droplet", "transmisi", "udara"],
    QueryIntent.MDR_TB      : ["mdr", "resistan", "xdr", "kebal", "resistensi"],
    QueryIntent.DIAGNOSIS   : ["diagnosis", "dahak", "rontgen", "thorax", "tuberkulin", "pemeriksaan"],
    QueryIntent.UMUM        : [],  # tidak ada filter → semua chunk lolos
}


def detect_intent(query: str) -> QueryIntent:
    """
    Deteksi intent dari query berdasarkan keyword matching.
    Fallback ke UMUM jika tidak ada keyword yang cocok.
    """
    query_lower = query.lower()
    scores: dict[QueryIntent, int] = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw in query_lower)
        if hit > 0:
            scores[intent] = hit

    if not scores:
        return QueryIntent.UMUM

    return max(scores, key=lambda i: scores[i])


def filter_by_intent(chunk: str, intent: QueryIntent) -> bool:
    """
    Kembalikan True jika chunk LOLOS filter intent.
    Chunk lolos jika mengandung setidaknya SATU keyword dari
    INTENT_FILTER_KEYWORDS[intent].
    Untuk intent UMUM, semua chunk selalu lolos.
    """
    required_keywords = INTENT_FILTER_KEYWORDS[intent]

    # Intent UMUM tidak punya filter — semua lolos
    if not required_keywords:
        return True

    chunk_lower = chunk.lower()
    return any(kw in chunk_lower for kw in required_keywords)

# ===============================
# LOAD VECTOR DB
# ===============================

print("🔎 Loading Vector DB...")

faiss_path  = os.path.join(VECTOR_FOLDER, "tbc_index.faiss")
chunks_path = os.path.join(VECTOR_FOLDER, "tbc_chunks.pkl")

if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
    raise FileNotFoundError(
        f"Vector store tidak ditemukan di: {VECTOR_FOLDER}\n"
        "Jalankan build_vectordb.py terlebih dahulu."
    )

index = faiss.read_index(faiss_path)
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

print(f"✅ Index loaded — {index.ntotal} vektor | {len(chunks)} chunks")

bi_encoder    = SentenceTransformer(MODEL_NAME)

print("🔁 Loading cross-encoder untuk re-ranking...")
cross_encoder = CrossEncoder(RERANKER_NAME)
print("✅ Cross-encoder siap.")

# ===============================
# CHUNK TRIMMER
# ===============================

def trim_chunk(chunk: str, max_chars: int = CHUNK_MAX_CHARS) -> str:
    """
    Perkecil chunk ke batas karakter tanpa memotong di tengah kata.
    """
    if len(chunk) <= max_chars:
        return chunk
    trimmed = chunk[:max_chars].rsplit(" ", 1)[0]
    return trimmed + "..."

# ===============================
# FUNGSI RETRIEVAL + RE-RANKING
# ===============================

def retrieve(
    query: str,
    top_k: int = TOP_K,
    top_n: int = TOP_N_RERANK,
    bi_threshold: float = BI_SCORE_THRESHOLD,
    cross_threshold: float = CROSS_SCORE_THRESHOLD,
    verbose: bool = True,
) -> list[dict]:
    """
    Pipeline tiga tahap:
      1. Bi-encoder + FAISS        → retrieval cepat (top_k kandidat)
      2. Intent filter             → buang chunk tidak relevan dengan topik query
      3. Cross-encoder re-ranking  → skor akurat + filter cross_threshold

    Returns:
        List of dicts: {rank, intent, bi_score, cross_score, chunk, chunk_full}
    """

    # ── Tahap 1: Deteksi intent ──────────────────────────────────────────────
    intent = detect_intent(query)
    if verbose:
        print(f"🎯 Intent terdeteksi : {intent.value}")

    # ── Tahap 2: Bi-encoder retrieval ────────────────────────────────────────
    query_vec = bi_encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    bi_scores, indices = index.search(query_vec, top_k)

    candidates = []
    for bi_score, idx in zip(bi_scores[0], indices[0]):
        if idx == -1:
            continue
        if float(bi_score) < bi_threshold:
            continue
        candidates.append({
            "idx"      : int(idx),
            "bi_score" : float(bi_score),
            "chunk"    : chunks[idx],
        })

    if verbose:
        print(f"📥 Lolos bi-threshold  : {len(candidates)} / {top_k}")

    if not candidates:
        if verbose:
            print("⚠️  Tidak ada kandidat yang melewati bi-threshold.")
        return []

    # ── Tahap 3: Filter berdasarkan intent ───────────────────────────────────
    before_filter = len(candidates)
    candidates = [c for c in candidates if filter_by_intent(c["chunk"], intent)]

    if verbose:
        dropped = before_filter - len(candidates)
        print(f"🔍 Lolos intent filter : {len(candidates)} / {before_filter}"
              f"  ({dropped} chunk dibuang karena tidak relevan dengan intent '{intent.value}')")

    if not candidates:
        if verbose:
            print("⚠️  Semua kandidat dibuang oleh intent filter. "
                  "Coba turunkan filter atau gunakan query yang lebih spesifik.")
        return []

    # ── Tahap 4: Cross-encoder re-ranking ────────────────────────────────────
    pairs        = [(query, c["chunk"]) for c in candidates]
    cross_scores = cross_encoder.predict(pairs)

    for c, cs in zip(candidates, cross_scores):
        c["cross_score"] = float(cs)

    # Urutkan berdasarkan cross-encoder score
    candidates.sort(key=lambda x: x["cross_score"], reverse=True)

    # Filter cross-score minimum
    before_cross = len(candidates)
    candidates   = [c for c in candidates if c["cross_score"] >= cross_threshold]

    if verbose:
        dropped = before_cross - len(candidates)
        print(f"✂️  Lolos cross-threshold: {len(candidates)} / {before_cross}"
              f"  ({dropped} chunk dibuang karena cross_score < {cross_threshold})")

    if not candidates:
        if verbose:
            print("⚠️  Semua kandidat dibuang oleh cross-threshold. "
                  "Coba turunkan CROSS_SCORE_THRESHOLD.")
        return []

    # Ambil top_n hasil final
    final   = candidates[:top_n]
    results = []
    for rank, c in enumerate(final, 1):
        results.append({
            "rank"        : rank,
            "intent"      : intent.value,
            "bi_score"    : round(c["bi_score"], 4),
            "cross_score" : round(c["cross_score"], 4),
            "chunk"       : trim_chunk(c["chunk"]),
            "chunk_full"  : c["chunk"],  # versi lengkap untuk dikirim ke LLM
        })

    return results

# ===============================
# RAG GENERATION (APPLIED RAG)
# ===============================

def build_context(results: list[dict]) -> str:
    """Gabungkan chunk_full menjadi satu blok konteks untuk LLM."""
    return "\n\n".join(
        f"[Sumber {r['rank']}]\n{r['chunk_full']}" for r in results
    )


def answer_query(query: str, verbose: bool = True) -> str:
    """
    Pipeline RAG lengkap:
      retrieve → build context → format prompt untuk LLM.

    Kembalikan prompt siap kirim atau pesan fallback jika tidak ada hasil.
    Ganti bagian LLM_CALL dengan model pilihanmu (OpenAI, Ollama, dsb).
    """
    results = retrieve(query, verbose=verbose)

    if not results:
        return (
            "Maaf, tidak ditemukan informasi yang relevan untuk pertanyaan tersebut. "
            "Silakan coba pertanyaan lain atau hubungi tenaga kesehatan."
        )

    context = build_context(results)
    intent  = results[0]["intent"]

    prompt = f"""Kamu adalah asisten medis yang membantu tenaga kesehatan menjawab
pertanyaan tentang Tuberkulosis (TBC) berdasarkan buku panduan resmi.

Jawab pertanyaan berikut HANYA berdasarkan konteks yang diberikan.
Jika informasi tidak tersedia dalam konteks, katakan "Informasi tidak tersedia."

Intent Query : {intent}
Pertanyaan   : {query}

Konteks:
{context}

Jawaban:"""

    # ── Ganti blok ini dengan pemanggilan LLM pilihanmu ──────────────────────
    # Contoh OpenAI:
    #   from openai import OpenAI
    #   client = OpenAI()
    #   response = client.chat.completions.create(
    #       model="gpt-4o-mini",
    #       messages=[{"role": "user", "content": prompt}]
    #   )
    #   return response.choices[0].message.content
    #
    # Contoh Ollama (lokal):
    #   import ollama
    #   response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    #   return response["message"]["content"]
    # ─────────────────────────────────────────────────────────────────────────

    # Sementara kembalikan prompt agar bisa ditest tanpa LLM
    return prompt

# ===============================
# PRINT RESULTS
# ===============================

def print_results(results: list[dict], query: str) -> None:
    print(f"\n{'='*60}")
    print(f"📋 Query  : {query}")
    print(f"📦 Hasil  : {len(results)} chunk relevan")
    if results:
        print(f"🎯 Intent : {results[0]['intent']}")
    print(f"{'='*60}")

    if not results:
        print("Tidak ada hasil yang ditemukan.")
        return

    for r in results:
        print(f"\n🔹 Rank {r['rank']}")
        print(f"   Bi-encoder   : {r['bi_score']:.4f}")
        print(f"   Cross-encoder: {r['cross_score']:.4f}")
        print(f"   {r['chunk']}")

    print(f"\n{'='*60}\n")

# ===============================
# TEST QUERIES
# ===============================

TEST_QUERIES = [
    "Berapa lama pengobatan TBC paru-paru?",
    "Apa saja gejala utama tuberkulosis?",
    "Obat apa yang digunakan untuk mengobati TBC?",
    "Bagaimana cara penularan tuberkulosis?",
    "Apa itu TBC resistan obat (MDR-TB)?",
]


def run_tests() -> None:
    print("\n🧪 Menjalankan test queries...\n")
    for query in TEST_QUERIES:
        results = retrieve(query, verbose=True)
        print_results(results, query)


# ===============================
# MAIN — INTERACTIVE + TEST
# ===============================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        print("\n💬 Mode Interaktif  (ketik 'exit' untuk keluar, 'test' untuk test queries)\n")
        while True:
            try:
                user_input = input("Pertanyaan: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nKeluar.")
                break

            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("Keluar.")
                break
            if user_input.lower() == "test":
                run_tests()
                continue

            # Applied RAG: retrieve + format prompt untuk LLM
            print("\n" + answer_query(user_input))
