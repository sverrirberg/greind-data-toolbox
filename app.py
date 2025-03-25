import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
import datetime

st.set_page_config(page_title="UNSPSC Toolkit", layout="wide")
st.title("🛠️ Klappir Data Toolbox")

# --- Caching ---
@st.cache_resource
def load_model():
    model = joblib.load("xgb_unspsc_model.pkl")
    encoder = joblib.load("xgb_label_encoder.pkl")
    return model, encoder

@st.cache_resource
def load_bert():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_unspsc_mapping():
    df = pd.read_csv("UNSPSC_classification.csv")
    df["unspsc_code"] = df["unspsc_code"].astype(str).str.zfill(6)
    return df.set_index("unspsc_code")["description"].to_dict()

model, encoder = load_model()
bert = load_bert()
unspsc_map = load_unspsc_mapping()

# --- Prediction Function ---
def predict(description):
    emb = bert.encode([description])
    probs = model.predict_proba(emb)[0]
    top_index = np.argmax(probs)
    top_code_raw = encoder.inverse_transform([top_index])[0]
    top_code = str(top_code_raw).zfill(6)
    top_desc = unspsc_map.get(top_code, "❌ Not found")
    confidence = probs[top_index]
    return top_code, top_desc, confidence

# --- Save Feedback ---
def log_feedback(description, predicted_code, correct_code):
    log_file = "feedback_log.csv"
    timestamp = datetime.datetime.now().isoformat()
    entry = pd.DataFrame([[timestamp, description, predicted_code, correct_code]],
                         columns=["timestamp", "description", "predicted_code", "correct_code"])
    if os.path.exists(log_file):
        entry.to_csv(log_file, mode="a", header=False, index=False)
    else:
        entry.to_csv(log_file, index=False)

# --- Sidebar Menu ---
mode = st.sidebar.radio("Choose a Tool:", ["🔍 UNSPSC Prediction", "🧪 CSV Profiling (YData)"])

if mode == "🔍 UNSPSC Prediction":
    st.header("🔍 Predict UNSPSC Code")
    uploaded_file = st.file_uploader("Upload a CSV with a 'description' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "description" not in df.columns:
            st.error("CSV must contain a 'description' column.")
        else:
            st.info(f"⏳ Predicting {len(df)} items...")
            results = []
            for desc in df["description"]:
                code, desc_text, conf = predict(str(desc))
                results.append({
                    "input_description": desc,
                    "predicted_code": code,
                    "predicted_description": desc_text,
                    "confidence": f"{conf:.2%}",
                    "correct_code": ""
                })
elif tool == "CSV Profiler":
    st.header("📊 CSV Profiler")

    profiling_file = st.file_uploader("Upload a CSV file for profiling", type=["csv"])

    if profiling_file:
        df = pd.read_csv(profiling_file)
        st.write("✅ File loaded. Generating profiling report...")

        profile = ProfileReport(df, title="📊 Data Profiling Report", explorative=True)
        st_profile_report(profile)

        # Save the HTML report to a temporary file and offer download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            profile.to_file(tmp_file.name)
            tmp_file.seek(0)
            st.download_button(
                label="📥 Download HTML Report",
                data=tmp_file.read(),
                file_name="data_profile_report.html",
                mime="text/html"
            )

            result_df = pd.DataFrame(results)
            st.success("✅ Predictions complete!")
            st.dataframe(result_df, use_container_width=True)

            # Download
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

    else:
        st.subheader("Or enter a single description below")
        input_text = st.text_area("Procurement Description:")
        if input_text:
            code, desc, conf = predict(input_text)
            st.write(f"**Predicted Code:** `{code}`")
            st.write(f"**Description:** {desc}")
            st.write(f"**Confidence:** {conf:.2%}")

            feedback = st.text_input("If incorrect, enter the correct code:")
            if feedback:
                log_feedback(input_text, code, feedback)
                st.success("📩 Feedback saved. Thank you!")

elif mode == "🧪 CSV Profiling (YData)":
    st.header("🧪 Automatic CSV Data Profiler")
    uploaded_file = st.file_uploader("Upload a CSV file to profile", type="csv", key="profile")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Profile Report")
        profile = ProfileReport(df, title="CSV Data Report", explorative=True)
        st_profile_report(profile)

import subprocess

with st.expander("⚙️ Admin Tools"):
    if st.button("🔄 Retrain Model"):
        with st.spinner("Retraining model..."):
            result = subprocess.run(["python", "retrain_model.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("✅ Model retrained successfully!")
            else:
                st.error(f"❌ Retraining failed:\n{result.stderr}")

import tempfile
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# After reading the uploaded CSV into `df`:
if profiling_file:
    df = pd.read_csv(profiling_file)
    profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
    st_profile_report(profile)

    # Save to a temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        profile.to_file(tmp_file.name)
        tmp_file.seek(0)
        st.download_button(
            label="📥 Download HTML Report",
            data=tmp_file.read(),
            file_name="data_profile_report.html",
            mime="text/html"
        )
