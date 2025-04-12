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
    st.markdown("""
    <style>
    .logo-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    </style>
    <div class="logo-container">
    """, unsafe_allow_html=True)
    st.image(logo, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("**Version:** 1.1.2", unsafe_allow_html=True)

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
mode = st.sidebar.radio("Choose a Tool:", ["🧪 CSV Profiling (YData)", "🔍 UNSPSC LLM Training"])

# === YData Profiling ===
if mode == "🧪 CSV Profiling (YData)":
    st.markdown("<h1 style='text-align: center;'>🧪 CSV Data Profiler</h1>", unsafe_allow_html=True)
    
    # Information text
    st.markdown("""
    ### 📝 Important Information
    - The CSV file should be in UTF-8 encoding
    - The first row should contain column headers
    - Date columns should be in a standard format (YYYY-MM-DD or similar)
    - Numeric columns should use dot (.) as decimal separator
    - Empty cells will be treated as missing values
    """)
    
    # Upload section
    profiling_file = st.file_uploader("Upload CSV file", type="csv", key="profile")
    
    if profiling_file:
        df = pd.read_csv(profiling_file)
        
        # Calculate data quality metrics
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * total_cols)) * 100
        
        # Calculate data quality score (0-100)
        quality_score = 100 - (missing_percentage * 2)  # Simple scoring based on missing values
        
        # Show data quality score
        st.subheader("📊 Data Quality Score")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style='text-align: center;'>
                <h2 style='color: {'#4CAF50' if quality_score >= 80 else '#FFC107' if quality_score >= 60 else '#F44336'};'>
                    {quality_score:.1f}%
                </h2>
                <p>Overall Data Quality</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            ### Quality Indicators
            - 🟢 Excellent (80-100%)
            - 🟡 Good (60-79%)
            - 🔴 Needs Attention (0-59%)
            """)
        
        # Show file information in a table
        st.subheader("📊 File Information")
        file_info = pd.DataFrame({
            'Metric': ['Number of rows', 'Number of columns', 'Total missing values', 'Missing value percentage'],
            'Value': [len(df), len(df.columns), missing_values, f"{missing_percentage:.2f}%"]
        }).set_index('Metric')
        st.table(file_info)
        
        # Show column names in a table
        st.subheader("📋 Column Names")
        columns_df = pd.DataFrame({
            'Data Type': df.dtypes.astype(str),
            'Missing Values': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df) * 100).round(2).astype(str) + '%',
            'Quality Alert': [
                '🟢' if (df[col].isnull().sum() / len(df) * 100) < 5
                else '🟡' if (df[col].isnull().sum() / len(df) * 100) < 20
                else '🔴'
                for col in df.columns
            ]
        }, index=df.columns)
        st.table(columns_df)
        
        # Missing values download section
        st.subheader("📥 Download Missing Values")
        st.write("Select columns to include in the missing values report:")
        
        # Create checkboxes for each column
        selected_columns = []
        col1, col2, col3 = st.columns(3)
        columns_per_col = (len(df.columns) + 2) // 3  # Divide columns into 3 columns
        
        for i, col in enumerate(df.columns):
            if i < columns_per_col:
                with col1:
                    if st.checkbox(col, key=f"col_{i}"):
                        selected_columns.append(col)
            elif i < columns_per_col * 2:
                with col2:
                    if st.checkbox(col, key=f"col_{i}"):
                        selected_columns.append(col)
            else:
                with col3:
                    if st.checkbox(col, key=f"col_{i}"):
                        selected_columns.append(col)
        
        if selected_columns:
            # Create DataFrame with only rows that have missing values in selected columns
            missing_df = df[df[selected_columns].isnull().any(axis=1)][selected_columns]
            
            if not missing_df.empty:
                # Add a column showing which columns are missing for each row
                missing_df['Missing In'] = missing_df.apply(
                    lambda row: ', '.join([col for col in selected_columns if pd.isnull(row[col])]),
                    axis=1
                )
                
                # Download button
                csv = missing_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Missing Values Report",
                    data=csv,
                    file_name="missing_values_report.csv",
                    mime="text/csv"
                )
            else:
                st.info("No missing values found in selected columns.")
        
        # Show data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df)
        
        # Configuration options
        st.subheader("⚙️ Configuration")
        col1, col2 = st.columns(2)
        with col1:
            show_correlation = st.checkbox("Show correlations", value=True)
            show_missing = st.checkbox("Show missing values", value=True)
        with col2:
            show_duplicates = st.checkbox("Show duplicates", value=True)
            show_samples = st.checkbox("Show samples", value=True)
        
        # Generate report button
        if st.button("📊 PROFILE CSV", key="profile_button"):
            with st.spinner("Generating report..."):
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
