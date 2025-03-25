# predict_with_xgb.py

import joblib
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# ------------ CONFIG -------------
MODEL_PATH = "xgb_unspsc_model.pkl"
ENCODER_PATH = "xgb_label_encoder.pkl"
BERT_MODEL = "all-MiniLM-L6-v2"
# ----------------------------------

# Clean up procurement text
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# Input: your procurement description
input_text = input("📝 Enter procurement line: ")
cleaned = clean(input_text)

# Load BERT model and embed
print("🔄 Encoding with BERT...")
bert = SentenceTransformer(BERT_MODEL)
X = bert.encode([cleaned])

# Load model and label encoder
print("🔍 Loading model...")
clf = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Predict
probabilities = clf.predict_proba(X)[0]
pred_index = np.argmax(probabilities)
pred_class = le.inverse_transform([pred_index])[0]
confidence = probabilities[pred_index]

# Output
print("\n✅ Prediction:")
print(f"Description     : {input_text}")
print(f"UNSPSC Code     : {pred_class}")
print(f"Confidence      : {confidence:.2%}")
