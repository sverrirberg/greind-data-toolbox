import joblib
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Load model
model = joblib.load("unspsc_model.pkl")

# Your input
text_input = input("AALBORG CEMENT BASIS 25KG - T/BETON  42 PS/PAL")

# Clean and predict
cleaned = clean_text(text_input)
prediction = model.predict([cleaned])[0]

print(f"Predicted UNSPSC code: {prediction}")
