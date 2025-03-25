# predict_csv_xgb.py

import pandas as pd
import joblib
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
INPUT_CSV = "new_procurements.csv"
OUTPUT_CSV = "predicted_unspsc.csv"
MODEL_PATH = "xgb_unspsc_model.pkl"
ENCODER_PATH = "xgb_label_encoder.pkl"
BERT_MODEL = "all-MiniLM-L6-v2"
# ----------------------------

# Clean text
def clean(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# Load and clean data
df = pd.read_csv(INPUT_CSV)
df["cleaned"] = df["description"].apply(clean)

# Load BERT model
print("🔄 Encoding descriptions...")
bert = SentenceTransformer(BERT_MODEL)
X = bert.encode(df["cleaned"].tolist())

# Load model and encoder
print("🔍 Loading model...")
clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Predict
print("🚀 Predicting...")
probs = clf.predict_proba(X)
pred_indices = np.argmax(probs, axis=1)
df["predicted_code"] = le.inverse_transform(pred_indices)
df["confidence"] = np.max(probs, axis=1)

# Save results
df[["description", "predicted_code", "confidence"]].to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Predictions saved to {OUTPUT_CSV}")
