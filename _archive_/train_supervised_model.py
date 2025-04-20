import pandas as pd
import numpy as np
import re
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# -------- CONFIG --------
CSV_PATH = "training_feedback.csv"
BERT_MODEL = "all-MiniLM-L6-v2"
XGB_MODEL_PATH = "xgb_unspsc_model.pkl"
ENCODER_PATH = "xgb_label_encoder.pkl"
# ------------------------

# Text cleaning
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

# Load labeled training data
df = pd.read_csv(CSV_PATH)
df["clean_text"] = df["description"].apply(clean)

# Load sentence-transformers model
bert = SentenceTransformer(BERT_MODEL)

# Generate sentence embeddings
print("🔄 Encoding descriptions with BERT...")
X = bert.encode(df["clean_text"].tolist())

# Prepare target variable
y = df["unspsc_code"]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split train/test for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the XGBoost classifier
print("🚀 Training XGBoost classifier...")
clf = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(le.classes_),
    use_label_encoder=False
)
clf.fit(X_train, y_train)

# Evaluate performance
print("\n📊 Model Evaluation:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=[str(cls) for cls in le.classes_]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save model and encoder
joblib.dump(clf, XGB_MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"\n✅ Model saved to {XGB_MODEL_PATH}")
print(f"✅ Label encoder saved to {ENCODER_PATH}")
