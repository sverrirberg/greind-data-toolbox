import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load UNSPSC code description dataset
df = pd.read_csv("unspsc_classification.csv")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df["clean_description"] = df["description"].apply(clean_text)

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_description"], df["unspsc_code"], test_size=0.2, random_state=42
)

# Create model pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nModel Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "unspsc_code_model.pkl")
print("Model saved to 'unspsc_code_model.pkl'")
