# hybrid_predictor.py

import re
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ---------- CONFIG ----------
BERT_MODEL = "all-MiniLM-L6-v2"
XGB_MODEL_PATH = "xgb_unspsc_model.pkl"
ENCODER_PATH = "xgb_label_encoder.pkl"
REFERENCE_EMBEDDINGS_PATH = "unspsc_reference_embeddings.pkl"
TOP_N = 3
# ----------------------------

def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# Get input
input_text = input("📝 Enter procurement line: ")
cleaned_text = clean(input_text)

# Load models
print("🔍 Loading models...")
bert = SentenceTransformer(BERT_MODEL)
clf = joblib.load(XGB_MODEL_PATH)
le = joblib.load(ENCODER_PATH)
ref_df = joblib.load(REFERENCE_EMBEDDINGS_PATH)
ref_embeddings = np.stack(ref_df["embedding"].values)

# Embed input
print("🔄 Encoding input...")
embedding = bert.encode([cleaned_text])

# --- Semantic Search (cosine similarity) ---
cos_scores = util.cos_sim(embedding, ref_embeddings)[0]
sem_top_idx = cos_scores.argsort(descending=True)[:TOP_N]
semantic_results = [
    {
        "source": "semantic",
        "unspsc_code": ref_df.iloc[int(i)]["unspsc_code"],
        "description": ref_df.iloc[int(i)]["description"],
        "score": float(cos_scores[int(i)])
    }
    for i in sem_top_idx
]

# --- Supervised XGBoost Prediction ---
probs = clf.predict_proba(embedding)[0]
xgb_top_idx = probs.argsort()[::-1][:TOP_N]
xgb_results = [
    {
        "source": "xgboost",
        "unspsc_code": le.inverse_transform([i])[0],
        "description": ref_df[ref_df["unspsc_code"] == le.inverse_transform([i])[0]].description.values[0],
        "score": float(probs[i])
    }
    for i in xgb_top_idx if le.inverse_transform([i])[0] in ref_df["unspsc_code"].values
]

# --- Merge and Deduplicate by Code ---
all_results = semantic_results + xgb_results
seen = set()
merged = []
for r in sorted(all_results, key=lambda x: -x["score"]):
    if r["unspsc_code"] not in seen:
        seen.add(r["unspsc_code"])
        merged.append(r)
    if len(merged) == TOP_N:
        break

# --- Output ---
print(f"\n🔍 Top {TOP_N} UNSPSC predictions for:\n📦 '{input_text}'")
for i, r in enumerate(merged, 1):
    print(f"\n{i}. {r['unspsc_code']} → {r['description']}")
    print(f"   Source     : {r['source']}")
    print(f"   Confidence : {r['score']:.2%}")
