import joblib
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# ---------- CONFIG ----------
TOP_N = 3
REFERENCE_FILE = "unspsc_reference_embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
# ----------------------------

# Clean incoming text
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# Load the reference embeddings
df = joblib.load(REFERENCE_FILE)

# Make sure the embeddings are stacked for performance
reference_embeddings = np.stack(df["embedding"].values)

# Load the BERT model
model = SentenceTransformer(MODEL_NAME)

# Your procurement-style test inputs
test_lines = [
    "Steel rods for concrete foundation",
    "Laptop with 16GB RAM for the office",
    "Order of eggs for kitchen",
    "Wireless router for new building",
    "Wall paint matte white 5L"
]

# Clean and embed inputs
cleaned_inputs = [clean(line) for line in test_lines]
input_embeddings = model.encode(cleaned_inputs)

# Match each input to closest UNSPSC codes
for i, input_emb in enumerate(input_embeddings):
    cosine_scores = util.cos_sim(input_emb, reference_embeddings)[0]
    top_indices = cosine_scores.argsort(descending=True)[:TOP_N]

    print(f"\n🔍 Input: {test_lines[i]}")
    print("Top UNSPSC code matches:")

    for rank, idx_tensor in enumerate(top_indices, start=1):
        try:
            idx = int(idx_tensor)
            code = df.iloc[idx]["unspsc_code"]
            desc = df.iloc[idx]["description"]
            score = cosine_scores[idx].item()
            print(f"  {rank}. {code} → {desc}  (Score: {score:.4f})")
        except Exception as e:
            print(f"  ⚠️  Error at rank {rank}: {e}")