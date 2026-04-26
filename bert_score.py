import pandas as pd
from bert_score import score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data_eval.csv")

ground_truth = df["ground_truth"].tolist()
response_a = df["response_a"].tolist()
response_b = df["response_b"].tolist()

# =========================
# HITUNG BERTSCORE
# =========================

print("Menghitung BERTScore untuk Response A...")
P_a, R_a, F1_a = score(response_a, ground_truth, lang="id")

print("Menghitung BERTScore untuk Response B...")
P_b, R_b, F1_b = score(response_b, ground_truth, lang="id")

# =========================
# MASUKKAN KE DATAFRAME
# =========================

df["precision_A"] = P_a.numpy()
df["recall_A"] = R_a.numpy()
df["f1_A"] = F1_a.numpy()

df["precision_B"] = P_b.numpy()
df["recall_B"] = R_b.numpy()
df["f1_B"] = F1_b.numpy()

# =========================
# RATA-RATA
# =========================

print("\n=== HASIL RATA-RATA ===")
print("Response A")
print("Precision:", df["precision_A"].mean())
print("Recall   :", df["recall_A"].mean())
print("F1 Score :", df["f1_A"].mean())

print("\nResponse B")
print("Precision:", df["precision_B"].mean())
print("Recall   :", df["recall_B"].mean())
print("F1 Score :", df["f1_B"].mean())

# =========================
# SIMPAN HASIL
# =========================

df.to_csv("hasil_bertscore.csv", index=False)

print("\nHasil disimpan ke hasil_bertscore.csv")