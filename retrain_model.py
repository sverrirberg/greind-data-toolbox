import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

# Load data (versioned training set with feedback)
df = pd.read_csv("data/training_data_v2.csv")  # Ensure this includes `description` and `unspsc_code`

# Step 1: Encode descriptions
bert = SentenceTransformer("all-MiniLM-L6-v2")
X = bert.encode(df["description"].astype(str).tolist())

# Step 2: Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["unspsc_code"].astype(str).str.zfill(6))

# Step 3: Train model
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
model.fit(X, y)

# Step 4: Save model and label encoder
joblib.dump(model, "xgb_unspsc_model.pkl")
joblib.dump(encoder, "xgb_label_encoder.pkl")

print("✅ Model retrained and saved.")
