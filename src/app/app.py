import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
import datetime
import tempfile
import subprocess

st.set_page_config(
    page_title="Greind Data Toolbox",
    layout="wide",
    page_icon="src/app/assets/favicon-32x32.png"
)
st.title("")

from PIL import Image

# Load the logo
logo_path = "src/app/assets/Logo_Greind_Horizontal.png"
logo = Image.open(logo_path)

# Show the logo in the sidebar
with st.sidebar:
    st.image(logo, width=250)  # Adjust width as needed
    st.markdown("**Version:** 1.0.5", unsafe_allow_html=True)

# --- Caching ---
@st.cache_resource
def load_model():
    model = joblib.load("src/models/xgb_unspsc_model.pkl")
    encoder = joblib.load("src/models/xgb_label_encoder.pkl")
    return model, encoder

@st.cache_resource
def load_bert():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_unspsc_mapping():
    df = pd.read_csv("data/raw/UNSPSC_classification.csv")
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
mode = st.sidebar.radio("Choose a Tool:", ["🧪 CSV Profiling (YData)", "🔍 UNSPSC LLM Training", "📊 NAICS Project"])

# === YData Profiling ===
if mode == "🧪 CSV Profiling (YData)":
    st.header("🧪 CSV Data Profiler")
    
    # Upload section
    profiling_file = st.file_uploader("Upload CSV file", type="csv", key="profile")
    
    if profiling_file:
        df = pd.read_csv(profiling_file)
        
        # Configuration options
        with st.expander("⚙️ Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                show_correlation = st.checkbox("Show correlations", value=True)
                show_missing = st.checkbox("Show missing values", value=True)
            with col2:
                show_duplicates = st.checkbox("Show duplicates", value=True)
                show_samples = st.checkbox("Show samples", value=True)
        
        # Generate and show report
        profile = ProfileReport(
            df,
            title="CSV Data Report",
            explorative=True,
            correlations={"auto": {"calculate": show_correlation}},
            missing_diagrams={"matrix": show_missing},
            duplicates={"head": show_duplicates},
            samples={"head": show_samples}
        )
        st_profile_report(profile)

        # Download button
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            profile.to_file(tmp_file.name)
            tmp_file.seek(0)
            st.download_button(
                label="📥 Download Report",
                data=tmp_file.read(),
                file_name="data_profile_report.html",
                mime="text/html"
            )

# === UNSPSC Prediction ===
elif mode == "🔍 UNSPSC LLM Training":
    st.header("🔍 UNSPSC LLM Training")
    uploaded_file = st.file_uploader("Upload a CSV with a 'description' column", type=["csv"], key="unspsc")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "description" not in df.columns:
            st.error("❌ CSV file must contain a 'description' column")
        else:
            # Show sample of the data
            st.subheader("📋 Sample of your data")
            st.dataframe(df.head())

            # Process button
            if st.button("Process Data"):
                with st.spinner("Processing..."):
                    # Load models
                    model, encoder = load_model()
                    bert = load_bert()
                    unspsc_map = load_unspsc_mapping()

                    # Make predictions
                    predictions = []
                    for desc in df["description"]:
                        pred_code = predict(desc)
                        predictions.append(pred_code)

                    # Add predictions to dataframe
                    df["predicted_unspsc"] = predictions
                    df["predicted_unspsc_description"] = df["predicted_unspsc"].map(unspsc_map)

                    # Show results
                    st.subheader("🎯 Prediction Results")
                    st.dataframe(df)

                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="predicted_unspsc.csv",
                        mime="text/csv"
                    )

                    # Feedback section
                    st.subheader("💬 Feedback")
                    st.write("Help improve the model by providing feedback on predictions:")
                    feedback_desc = st.text_area("Description")
                    feedback_pred = st.text_input("Predicted UNSPSC Code")
                    feedback_correct = st.text_input("Correct UNSPSC Code (if different)")

                    if st.button("Submit Feedback"):
                        if feedback_desc and feedback_pred:
                            log_feedback(feedback_desc, feedback_pred, feedback_correct)
                            st.success("📩 Feedback saved. Thank you!")

    # Admin Tools
    with st.expander("⚙️ Admin Tools"):
        st.subheader("Retrain Model")
        training_file = st.file_uploader("Upload training data (CSV with 'description' and 'unspsc_code' columns)", type=["csv"], key="train")

        if training_file:
            train_df = pd.read_csv(training_file)
            if "description" not in train_df.columns or "unspsc_code" not in train_df.columns:
                st.error("❌ Training file must contain 'description' and 'unspsc_code' columns")
            else:
                if st.button("Retrain Model"):
                    with st.spinner("Training new model..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                            train_df.to_csv(tmp_file.name, index=False)
                        
                        # Run training script
                        try:
                            subprocess.run(["python", "src/models/train_model_6digit.py", tmp_file.name], check=True)
                            st.success("✅ Model retrained successfully!")
                        except subprocess.CalledProcessError as e:
                            st.error(f"❌ Error retraining model: {str(e)}")

# === NAICS Project ===
elif mode == "📊 NAICS Project":
    st.header("📊 NAICS Project")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Show the data
        st.subheader("📋 Data Preview")
        st.dataframe(df)
        
        # Add your NAICS Project functionality here
        # For example:
        # - Data processing
        # - Model predictions
        # - Results visualization
