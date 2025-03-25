import os
os.environ["PYTORCH_NO_CUSTOM_CLASS_WARNING"] = "1"


# train_model_6digit.py

import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import xgboost as xgb

# --- Config ---
CSV_PATH = "training_data_6digit_clean.csv"
MODEL_OUT = "xgb_unspsc_model.pkl"
ENCODER_OUT = "xgb_label_encoder.pkl"
BERT_MODEL = "all-MiniLM-L6-v2"

# --- Load data ---
print("📥 Loading training data...")
df = pd.read_csv(CSV_PATH)
descriptions = df["description"].astype(str).tolist()
labels = df["unspsc_code"].astype(str).str.zfill(6)

# --- Encode labels ---
print("🔠 Encoding UNSPSC codes...")
le = LabelEncoder()
y = le.fit_transform(labels)

# --- Embed descriptions with BERT ---
print("🔄 Embedding descriptions...")
bert = SentenceTransformer(BERT_MODEL)
X = bert.encode(descriptions, show_progress_bar=True)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train classifier ---
print("🚀 Training XGBoost model...")
clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    use_label_encoder=False
)
clf.fit(X_train, y_train)

# --- Evaluate ---
print("📊 Evaluating model...")
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=le.inverse_transform(np.unique(y_test)), zero_division=0)

print(report)

# --- Save model and encoder ---
print("💾 Saving model and label encoder...")
joblib.dump(clf, MODEL_OUT)
joblib.dump(le, ENCODER_OUT)
print("✅ Done. Model saved.")
