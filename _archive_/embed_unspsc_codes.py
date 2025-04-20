from sentence_transformers import SentenceTransformer
import pandas as pd
import joblib

# Load the UNSPSC code reference CSV
df = pd.read_csv("UNSPSC_classification.csv")

# Load the BERT-based embedding model (very good balance of speed/quality)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for each UNSPSC description
print("Generating embeddings...")
df["embedding"] = df["description"].apply(lambda x: model.encode(x))

# Save the entire DataFrame (code, description, embedding) for reuse
joblib.dump(df, "unspsc_reference_embeddings.pkl")
print("✅ Saved embeddings to 'unspsc_reference_embeddings.pkl'")
