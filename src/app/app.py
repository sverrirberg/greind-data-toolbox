import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
import datetime
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64
from scipy import stats

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
    .info-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box h3 {
        color: #333;
        margin-top: 0;
    }
    .info-box ul {
        margin: 0;
        padding-left: 1rem;
    }
    .info-box li {
        margin: 0.5rem 0;
        color: #666;
        font-size: 0.9em;
    }
    </style>
    <div class="logo-container">
    """, unsafe_allow_html=True)
    st.image(logo, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("**Version:** 1.1.2", unsafe_allow_html=True)
    
    # Important Information
    st.markdown("""
    <div class="info-box">
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

def find_duplicates(df):
    """Find duplicate rows in the dataframe."""
    duplicates = df.duplicated(keep='first')
    duplicate_count = duplicates.sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100
    return duplicate_count, duplicate_percentage

def find_outliers(df):
    """Find outliers in numeric columns using z-score method."""
    outliers_info = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        # Count outliers (z-score > 3)
        outliers_count = (z_scores > 3).sum()
        if outliers_count > 0:
            outliers_percentage = (outliers_count / len(df[column].dropna())) * 100
            outliers_info[column] = {
                'count': outliers_count,
                'percentage': outliers_percentage
            }
    return outliers_info

def create_html_report(df, quality_score, missing_values, file_info, filename):
    # Get current date and time
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate duplicates
    duplicate_count, duplicate_percentage = find_duplicates(df)
    
    # Calculate outliers
    outliers_info = find_outliers(df)
    
    # Convert logo to base64
    with open("src/app/assets/Logo_Greind_Horizontal.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Quality Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: white;
                padding: 20px 40px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }}
            .logo {{
                height: 60px;
            }}
            .title {{
                color: #333;
                font-size: 24px;
                margin: 0;
            }}
            .info-box {{
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .info-box p {{
                margin: 5px 0;
                color: #666;
                font-size: 14px;
            }}
            .info-box .generated-by {{
                color: #999;
                font-size: 12px;
                margin-top: 10px;
            }}
            .score-container {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .score-circle {{
                width: 200px;
                height: 200px;
                border-radius: 50%;
                margin: 0 auto;
                position: relative;
                background: {'#4CAF50' if quality_score >= 80 else '#FFC107' if quality_score >= 60 else '#F44336'};
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            .score-number {{
                font-size: 48px;
                font-weight: bold;
                color: white;
            }}
            .score-label {{
                margin-top: 15px;
                font-size: 18px;
                color: #666;
            }}
            .score-description {{
                margin-top: 10px;
                color: #666;
                font-size: 14px;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }}
            .section {{
                background-color: white;
                border-radius: 10px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section-title {{
                color: #333;
                font-size: 20px;
                margin: 0 0 20px 0;
                display: flex;
                align-items: center;
            }}
            .section-title i {{
                margin-right: 10px;
                color: #666;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background-color: white;
            }}
            th, td {{
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #eee;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
                color: #333;
            }}
            tr:hover {{
                background-color: #f8f9fa;
            }}
            .metric-icon {{
                color: #666;
                margin-right: 10px;
            }}
            .indicator {{
                display: inline-block;
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: 500;
                font-size: 14px;
            }}
            .excellent {{
                background-color: #4CAF50;
                color: white;
            }}
            .attention {{
                background-color: #FFC107;
                color: black;
            }}
            .poor {{
                background-color: #F44336;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <img src="data:image/png;base64,{encoded_string}" alt="Greind Logo" class="logo">
            <h1 class="title">Data Quality Report</h1>
        </div>
        
        <div class="container">
            <div class="info-box">
                <p><strong>Report generated on:</strong> {date_time}</p>
                <p><strong>Source file:</strong> {filename}</p>
                <p class="generated-by">Generated by Greind Data Toolbox v1.1.2</p>
            </div>
            
            <div class="score-container">
                <div class="score-circle">
                    <div class="score-number">{quality_score:.1f}%</div>
                </div>
                <div class="score-label">Data Quality Score</div>
                <div class="score-description">
                    {'Your data quality is excellent! Only minor improvements needed.' if quality_score >= 80
                    else 'Your data quality needs some attention. Consider addressing the issues highlighted below.' if quality_score >= 60
                    else 'Your data quality needs significant improvement. Please review the issues detailed below.'}
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">
                    <i>📊</i> File Information
                </h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td><span class="metric-icon">📝</span>Number of rows</td>
                        <td>{len(df)}</td>
                    </tr>
                    <tr>
                        <td><span class="metric-icon">📊</span>Number of columns</td>
                        <td>{len(df.columns)}</td>
                    </tr>
                    <tr>
                        <td><span class="metric-icon">❓</span>Total missing values</td>
                        <td>{missing_values}</td>
                    </tr>
                    <tr>
                        <td><span class="metric-icon">📉</span>Missing value percentage</td>
                        <td>{(missing_values / (len(df) * len(df.columns)) * 100):.2f}%</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">
                    <i>🔍</i> Data Quality Issues
                </h2>
                <table>
                    <tr>
                        <th>Issue Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                        <th>Status</th>
                    </tr>
                    <tr>
                        <td><span class="metric-icon">🔄</span>Duplicate Rows</td>
                        <td>{duplicate_count}</td>
                        <td>{duplicate_percentage:.2f}%</td>
                        <td>{'🟢 Low' if duplicate_percentage < 5 else '🟡 Medium' if duplicate_percentage < 20 else '🔴 High'}</td>
                    </tr>
                    <tr>
                        <td><span class="metric-icon">❓</span>Missing Values</td>
                        <td>{missing_values}</td>
                        <td>{(missing_values / (len(df) * len(df.columns)) * 100):.2f}%</td>
                        <td>{'🟢 Low' if (missing_values / (len(df) * len(df.columns)) * 100) < 5 else '🟡 Medium' if (missing_values / (len(df) * len(df.columns)) * 100) < 20 else '🔴 High'}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2 class="section-title">
                    <i>📈</i> Outliers Analysis
                </h2>
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Outliers Count</th>
                        <th>Percentage</th>
                        <th>Status</th>
                    </tr>
    """
    
    for column, info in outliers_info.items():
        html_content += f"""
                    <tr>
                        <td><span class="metric-icon">📈</span>{column}</td>
                        <td>{info['count']}</td>
                        <td>{info['percentage']:.2f}%</td>
                        <td>{'🟢 Low' if info['percentage'] < 5 else '🟡 Medium' if info['percentage'] < 20 else '🔴 High'}</td>
                    </tr>
        """
    
    html_content += """
                </table>
                <p class="note">Note: Outliers are identified using the z-score method (|z| > 3)</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def get_status_indicator(percentage):
    if percentage < 5:
        return '<span class="indicator excellent">🟢 Low Impact</span>'
    elif percentage < 20:
        return '<span class="indicator attention">🟡 Medium Impact</span>'
    else:
        return '<span class="indicator poor">🔴 High Impact</span>'

# Main app content
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🧪 CSV Data Profiler</h1>", unsafe_allow_html=True)

# Upload section
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    profiling_file = st.file_uploader("Upload CSV file", type="csv", key="profile")
    st.markdown("</div>", unsafe_allow_html=True)

if profiling_file:
    df = pd.read_csv(profiling_file)
    
    # Calculate basic metrics
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_values = df.isnull().sum().sum()
    missing_percentage = (missing_values / (total_rows * total_cols)) * 100
    
    # Calculate duplicates
    duplicate_count, duplicate_percentage = find_duplicates(df)
    
    # Calculate outliers
    outliers_info = find_outliers(df)
    
    # Calculate quality score
    duplicate_impact = min(duplicate_percentage * 2, 20)  # Max 20 points deduction
    outlier_impact = min(sum(info['percentage'] for info in outliers_info.values()) / len(outliers_info) if outliers_info else 0, 20)  # Max 20 points deduction
    quality_score = 100 - (missing_percentage * 1.5) - duplicate_impact - outlier_impact
    quality_score = max(0, min(100, quality_score))  # Ensure score is between 0 and 100
    
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
            
            # Add custom CSS for the download button
            st.markdown("""
            <style>
            div[data-testid="stDownloadButton"] {
                margin-top: 35px;
                padding-top: 20px;
                margin-left: 15%;
                width: 85%;
            }
            </style>
            """, unsafe_allow_html=True)
            
            file_info = pd.DataFrame({
                'Value': [str(len(df)), str(len(df.columns)), str(missing_values), f"{missing_percentage:.2f}%"]
            }, index=['Number of rows', 'Number of columns', 'Total missing values', 'Missing value percentage'])
            
            html_content = create_html_report(df, quality_score, missing_values, file_info, profiling_file.name)
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_content,
                file_name="data_quality_report.html",
                mime="text/html"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Show file information in a table
    with st.container():
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("💡 File Information")
        st.markdown("<p class='info-text'>Basic information about your dataset</p>", unsafe_allow_html=True)
        file_info = pd.DataFrame({
            'Value': [str(len(df)), str(len(df.columns)), str(missing_values), f"{missing_percentage:.2f}%"]
        }, index=['Number of rows', 'Number of columns', 'Total missing values', 'Missing value percentage'])
        st.table(file_info)
        
        # Show data quality issues in a table
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.subheader("🔍 Data Quality Issues")
            
            # Create a DataFrame for quality issues without index
            quality_issues = pd.DataFrame({
                'Issue Type': ['Duplicate Rows', 'Missing Values'],
                'Count': [duplicate_count, missing_values],
                'Percentage': [
                    f"{duplicate_percentage:.2f}%",
                    f"{missing_percentage:.2f}%"
                ],
                'Impact': [
                    '🟢 Low' if duplicate_percentage < 5 else '🟡 Medium' if duplicate_percentage < 20 else '🔴 High',
                    '🟢 Low' if missing_percentage < 5 else '🟡 Medium' if missing_percentage < 20 else '🔴 High'
                ]
            })
            st.table(quality_issues.set_index('Issue Type', drop=True))
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show outliers information in a table if any found
        if outliers_info:
            with st.container():
                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.subheader("📈 Outliers Analysis")
                
                # Create a DataFrame for outliers without index
                outliers_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Outliers Count': info['count'],
                        'Percentage': f"{info['percentage']:.2f}%",
                        'Impact': '🟢 Low' if info['percentage'] < 5 else '🟡 Medium' if info['percentage'] < 20 else '🔴 High'
                    }
                    for col, info in outliers_info.items()
                ])
                st.table(outliers_df.set_index('Column', drop=True))
                st.markdown("<p class='info-text'>Outliers are identified using the z-score method (|z| > 3)</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown("<div class='section'>", unsafe_allow_html=True)
                st.subheader("📈 Outliers Analysis")
                st.info("No outliers found in numeric columns using z-score method (|z| > 3)")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Show column names in a table
        with st.container():
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.subheader("📋 Column Names")
            st.markdown("<p class='info-text'>Detailed information about each column</p>", unsafe_allow_html=True)
            columns_df = pd.DataFrame({
                'Data Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(1).astype(str) + '%',
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
            
            # Add filtering options
            st.markdown("### 🔍 Filter Options")
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                # Date range filter
                if 'date' in df.columns or any('date' in col.lower() for col in df.columns):
                    date_col = st.selectbox(
                        "Select date column for filtering",
                        [col for col in df.columns if 'date' in col.lower() or 'date' in col],
                        index=0
                    )
                    if date_col:
                        min_date = pd.to_datetime(df[date_col].min())
                        max_date = pd.to_datetime(df[date_col].max())
                        date_range = st.date_input(
                            "Select date range",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date
                        )
            
            with filter_col2:
                # Missing value threshold filter
                threshold = st.slider(
                    "Minimum missing values percentage to include",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                    format="%d%%"
                )
            
            # Create checkboxes for each column with color coding
            selected_columns = []
            col1, col2, col3 = st.columns(3)
            columns_per_col = (len(df.columns) + 2) // 3
            
            # Calculate missing percentages for color coding
            missing_percentages = {
                col: (df[col].isnull().sum() / len(df) * 100)
                for col in df.columns
            }
            
            for i, col in enumerate(df.columns):
                # Determine color based on missing percentage
                missing_pct = missing_percentages[col]
                color = '#4CAF50' if missing_pct < 5 else '#FFC107' if missing_pct < 20 else '#F44336'
                
                checkbox_style = f"""
                <style>
                div[data-testid="stCheckbox"] label[data-testid="stMarkdownContainer"] p {{
                    color: {color};
                    font-weight: bold;
                }}
                </style>
                """
                
                if i < columns_per_col:
                    with col1:
                        st.markdown(checkbox_style, unsafe_allow_html=True)
                        if st.checkbox(f"{col} ({missing_pct:.1f}% missing)", key=f"col_{i}"):
                            selected_columns.append(col)
                elif i < columns_per_col * 2:
                    with col2:
                        st.markdown(checkbox_style, unsafe_allow_html=True)
                        if st.checkbox(f"{col} ({missing_pct:.1f}% missing)", key=f"col_{i}"):
                            selected_columns.append(col)
                else:
                    with col3:
                        st.markdown(checkbox_style, unsafe_allow_html=True)
                        if st.checkbox(f"{col} ({missing_pct:.1f}% missing)", key=f"col_{i}"):
                            selected_columns.append(col)
            
            if selected_columns:
                # Apply date filter if selected
                if 'date_range' in locals() and date_range and len(date_range) == 2:
                    start_date, end_date = date_range
                    mask = (pd.to_datetime(df[date_col]) >= pd.to_datetime(start_date)) & \
                           (pd.to_datetime(df[date_col]) <= pd.to_datetime(end_date))
                    filtered_df = df[mask]
                else:
                    filtered_df = df
                
                # Apply missing value threshold filter
                missing_df = filtered_df[filtered_df[selected_columns].isnull().any(axis=1)][selected_columns]
                missing_df = missing_df[
                    missing_df.apply(
                        lambda row: any(
                            missing_percentages[col] >= threshold
                            for col in selected_columns
                            if pd.isnull(row[col])
                        ),
                        axis=1
                    )
                ]
                
                if not missing_df.empty:
                    missing_df['Missing In'] = missing_df.apply(
                        lambda row: ', '.join([col for col in selected_columns if pd.isnull(row[col])]),
                        axis=1
                    )
                    
                    # Add missing percentages to the report
                    for col in selected_columns:
                        missing_df[f'{col} Missing %'] = missing_percentages[col]
                    
                    csv = missing_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Missing Values Report",
                        data=csv,
                        file_name="missing_values_report.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No missing values found matching the selected criteria.")
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
