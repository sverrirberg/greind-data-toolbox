import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load your dataset
df = pd.read_csv("generated_unspsc_training_data_10000.csv")
print(df.head())

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df["clean_description"] = df["description"].apply(clean_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_description"], df["unspsc_code"], test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, "unspsc_model.pkl")
print("\nModel saved as 'unspsc_model.pkl'")
