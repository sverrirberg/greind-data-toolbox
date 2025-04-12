import joblib
import re

# Preprocessing function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Load saved model
model = joblib.load("unspsc_model.pkl")

# New procurement descriptions to classify
new_lines = [
    "Invoice for 3 cement bags 25kg",
    "Line 12: 2 wireless routers",
    "Bottled water for office",
    "Order: 10 cheddar cheese blocks"
]

# Clean and predict
cleaned = [clean_text(line) for line in new_lines]
predictions = model.predict(cleaned)

for line, pred in zip(new_lines, predictions):
    print(f"{line} → UNSPSC: {pred}")
