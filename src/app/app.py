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
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
import pdfkit

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Greind Data Toolbox",
    layout="wide",
    page_icon="src/app/assets/favicon-32x32.png"
)
st.title("")

# Load the logo
logo_path = "src/app/assets/Logo_Greind_Horizontal.png"
logo = Image.open(logo_path)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 1.2em;
        background-color: #808080;
        color: white;
        border: none;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #666666;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTable {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .section {
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .info-text {
        color: #666;
        font-size: 0.9em;
        margin-top: 0.5rem;
    }
    .score-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
        min-height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .big-number {
        font-size: 10.5vw;
        font-weight: bold;
        line-height: 1;
        margin: 0;
        padding: 0;
        text-align: center;
        color: #333;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    @media (min-width: 1200px) {
        .big-number {
            font-size: 126px;
        }
    }
    .quality-indicators {
        padding-left: 2rem;
    }
</style>
""", unsafe_allow_html=True)

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
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🧪 CSV Data Profiler</h1>", unsafe_allow_html=True)
    
    # Information text
    with st.container():
        st.markdown("""
        <div class="section">
            <h3>📝 Important Information</h3>
            <ul>
                <li>The CSV file should be in UTF-8 encoding</li>
                <li>The first row should contain column headers</li>
                <li>Date columns should be in a standard format (YYYY-MM-DD or similar)</li>
                <li>Numeric columns should use dot (.) as decimal separator</li>
                <li>Empty cells will be treated as missing values</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload section
    with st.container():
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        profiling_file = st.file_uploader("Upload CSV file", type="csv", key="profile")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if profiling_file:
        df = pd.read_csv(profiling_file)
        
        # Calculate data quality metrics
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * total_cols)) * 100
        
        # Calculate data quality score (0-100)
        quality_score = 100 - (missing_percentage * 2)
        
        # Show data quality score
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.subheader("📊 Data Quality Score")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class='score-card'>
                    <div class='big-number' style='color: {'#4CAF50' if quality_score >= 80 else '#FFC107' if quality_score >= 60 else '#F44336'};'>
                        {quality_score:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class='quality-indicators'>
                    <h4>Quality Indicators</h4>
                    <ul>
                        <li>🟢 Excellent (80-100%)</li>
                        <li>🟡 Needs Attention (60-79%)</li>
                        <li>🔴 Poor (0-59%)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Add download and email options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download PDF Report"):
                    pdf = create_pdf_report(df, quality_score, missing_values, file_info)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        pdf.build(tmp_file.name)
                        with open(tmp_file.name, "rb") as f:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=f,
                                file_name="data_quality_report.pdf",
                                mime="application/pdf"
                            )
            
            with col2:
                st.markdown("""
                <div style='margin-top: 1rem;'>
                    <h4>📧 Email Report</h4>
                </div>
                """, unsafe_allow_html=True)
                email = st.text_input("Enter email address")
                if st.button("Send Report"):
                    if email:
                        pdf = create_pdf_report(df, quality_score, missing_values, file_info)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            pdf.build(tmp_file.name)
                            if send_email(email, "Data Quality Report", "Please find attached the data quality report.", tmp_file.name):
                                st.success("Report sent successfully!")
                            os.unlink(tmp_file.name)
                    else:
                        st.warning("Please enter an email address")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show file information in a table
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.subheader("💡 File Information")
            st.markdown("<p class='info-text'>Basic information about your dataset</p>", unsafe_allow_html=True)
            file_info = pd.DataFrame({
                'Metric': ['Number of rows', 'Number of columns', 'Total missing values', 'Missing value percentage'],
                'Value': [len(df), len(df.columns), missing_values, f"{missing_percentage:.2f}%"]
            }).set_index('Metric')
            st.table(file_info)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show column names in a table
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.subheader("📋 Column Names")
            st.markdown("<p class='info-text'>Detailed information about each column</p>", unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Missing values download section
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.subheader("📥 Download Missing Values File")
            st.markdown("<p class='info-text'>Select columns to include in the missing values report</p>", unsafe_allow_html=True)
            
            # Create checkboxes for each column
            selected_columns = []
            col1, col2, col3 = st.columns(3)
            columns_per_col = (len(df.columns) + 2) // 3
            
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
                missing_df = df[df[selected_columns].isnull().any(axis=1)][selected_columns]
                
                if not missing_df.empty:
                    missing_df['Missing In'] = missing_df.apply(
                        lambda row: ', '.join([col for col in selected_columns if pd.isnull(row[col])]),
                        axis=1
                    )
                    
                    csv = missing_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Missing Values Report",
                        data=csv,
                        file_name="missing_values_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No missing values found in selected columns.")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show data preview
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.subheader("📋 Data Preview")
            st.markdown("<p class='info-text'>First few rows of your dataset</p>", unsafe_allow_html=True)
            st.dataframe(df)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Configuration options
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.subheader("⚙️ Detailed Analysis")
            st.markdown("<p class='info-text'>Customize your analysis</p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                show_correlation = st.checkbox("Show correlations", value=True)
                show_missing = st.checkbox("Show missing values", value=True)
            with col2:
                show_duplicates = st.checkbox("Show duplicates", value=True)
                show_samples = st.checkbox("Show samples", value=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Generate report button
        if st.button("⚙️ Detailed Analysis", key="profile_button"):
            with st.spinner("Generating detailed analysis..."):
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

def create_pdf_report(df, quality_score, missing_values, file_info):
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ text-align: center; color: #333; }}
            .score {{ font-size: 48px; color: blue; text-align: center; margin: 20px 0; }}
            h2 {{ color: #444; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Data Quality Report</h1>
        <div class="score">{quality_score:.1f}%</div>
        
        <h2>File Information</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
    """
    
    for metric, value in file_info.items():
        html_content += f"""
            <tr>
                <td>{metric}</td>
                <td>{value}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Missing Values Summary</h2>
        <table>
            <tr>
                <th>Column</th>
                <th>Missing Count</th>
                <th>Missing %</th>
            </tr>
    """
    
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            html_content += f"""
                <tr>
                    <td>{col}</td>
                    <td>{missing}</td>
                    <td>{missing/len(df)*100:.1f}%</td>
                </tr>
            """
    
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    options = {
        'page-size': 'Letter',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None
    }
    
    pdfkit.from_string(html_content, "data_quality_report.pdf", options=options)
    return "data_quality_report.pdf"

def send_email(receiver_email, subject, body, attachment_path=None):
    sender_email = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    
    msg.attach(MIMEText(body, "plain"))
    
    if attachment_path:
        with open(attachment_path, "rb") as f:
            attach = MIMEApplication(f.read(), _subtype="pdf")
            attach.add_header("Content-Disposition", "attachment", filename="data_quality_report.pdf")
            msg.attach(attach)
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False
