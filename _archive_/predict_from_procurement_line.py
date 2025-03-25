import joblib
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Load model
model = joblib.load("unspsc_code_model.pkl")

# Your test description
procurement_line = "Steel rods for foundation"

cleaned = clean_text(procurement_line)
predicted_code = model.predict([cleaned])[0]
print(f"Input: {procurement_line}")
print(f"Predicted UNSPSC code: {predicted_code}")
