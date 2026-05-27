import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- Page Configuration ---
st.set_page_config(
    page_title="A-DAA v1 - AI Data Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #2C3E50;
    font-weight: 700;
}
h2, h3 {color: #34495E;}
.stTabs [data-baseweb="tab-list"] {justify-content: center;}
.stMetric {
    background-color: #e8f5e9;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.main {background-color: #f8f9fa;}
.css-1cpxqw2 {
    border: 2px dashed #4CAF50 !important;
    border-radius: 10px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>📊 A‑DAA v1 — AI Data Analyst</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Upload your dataset and start exploring insights.</p>", unsafe_allow_html=True)

# --- Recent Activity Log ---
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

def log_activity(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.append(f"{message} — {timestamp}")

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head(), use_container_width=True)
    log_activity(f"📁 File uploaded: {uploaded_file.name}")

    # --- Tabs for Navigation ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview", "EDA", "Visualizations", "Dashboard", "ML Mode", "Column Insights", "Recent Activity"
    ])

    # --- Overview ---
    with tab1:
        st.subheader("📌 Dataset Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Entries", len(df))
            st.metric("Features", df.shape[1])
        with col2:
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Duplicates", df.duplicated().sum())
        st.write(df.describe())

    # --- EDA ---
    with tab2:
        st.subheader("🔍 Exploratory Data Analysis")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, ax=ax)
            st.pyplot(fig)
            log_activity("🔍 Correlation heatmap generated")
        else:
            st.warning("No numeric columns found for correlation heatmap")

    # --- Visualizations ---
    with tab3:
        st.subheader("📈 Visualizations")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Line Plot", "Scatter Plot", "Box Plot"])
            if chart_type == "Histogram":
                col = st.selectbox("Choose Column", numeric_cols)
                fig = px.histogram(df, x=col, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                log_activity(f"📈 Histogram plotted for {col}")
            elif chart_type == "Line Plot":
                col = st.selectbox("Choose Column", numeric_cols)
                fig = px.line(df, y=col, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                log_activity(f"📈 Line plot generated for {col}")
            elif chart_type == "Scatter Plot":
                x = st.selectbox("X-axis", numeric_cols)
                y = st.selectbox("Y-axis", numeric_cols)
                fig = px.scatter(df, x=x, y=y, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                log_activity(f"📈 Scatter plot generated for {x} vs {y}")
            elif chart_type == "Box Plot":
                col = st.selectbox("Choose Column", numeric_cols)
                fig = px.box(df, y=col, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                log_activity(f"📈 Box plot generated for {col}")
        else:
            st.warning("No numeric columns found to visualize")

    # --- Dashboard Mode ---
    with tab4:
        st.subheader("📊 Dashboard Mode")
        col_filter = st.selectbox("Select Column to Filter", df.columns)
        unique_vals = df[col_filter].unique()
        selected_val = st.selectbox("Filter Value", unique_vals)
        filtered_df = df[df[col_filter] == selected_val]
        st.dataframe(filtered_df, use_container_width=True)

        numeric_cols = filtered_df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = st.selectbox("Select Column for KPIs & Charts", numeric_cols)

            # KPIs
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Mean", round(filtered_df[col].mean(), 2))
            kpi2.metric("Max", round(filtered_df[col].max(), 2))
            kpi3.metric("Min", round(filtered_df[col].min(), 2))

            st.subheader("📈 Charts")
            st.write("Line Plot")
            st.plotly_chart(px.line(filtered_df, y=col, template="plotly_white"), use_container_width=True)
            st.write("Histogram")
            st.plotly_chart(px.histogram(filtered_df, x=col, template="plotly_white"), use_container_width=True)
            st.write("Box Plot")
            st.plotly_chart(px.box(filtered_df, y=col, template="plotly_white"), use_container_width=True)

            if len(numeric_cols) > 1:
                st.write("Scatter Plot")
                other_col = st.selectbox("Select another column for Scatter", [c for c in numeric_cols if c != col])
                st.plotly_chart(px.scatter(filtered_df, x=col, y=other_col, template="plotly_white"), use_container_width=True)

            st.subheader("🔍 Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(filtered_df[numeric_cols].corr(), cmap="coolwarm", annot=True, ax=ax)
            st.pyplot(fig)
            log_activity("📊 Dashboard charts and heatmap generated")
        else:
            st.warning("No numeric columns to show KPIs or charts")

    # --- ML Mode ---
    with tab5:
        st.subheader("🤖 Machine Learning Mode (Regression)")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for ML mode")
        else:
            target = st.selectbox("Select Target Column", numeric_cols)
            features = st.multiselect("Select Feature Columns", [c for c in numeric_cols if c != target],
                                      default=[c for c in numeric_cols if c != target][:3])
            if st.button("Train Model"):
                clean_df = df[[target] + features].dropna()
                X = clean_df[features]
                y = clean_df[target]
                if len(X) < 10:
                    st.error("Not enough data after dropping NaNs. Need 10+ rows")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    st.success("Model Trained Successfully!")
                    st.write("R² Score:", round(r2_score(y_test, preds), 3))
                    st.write("MSE:", round(mean_squared_error(y_test, preds), 3))
                    log_activity("🤖 Regression model trained")

    # --- Column Insights ---
    with tab6:
        st.subheader("🧠 Column-wise Recommendations")
        for col in df.columns:
            st.write(f"### 📌 {col}")
            if df[col].dtype == "object":
                st.write("Recommendation: Encode this column (Label/One-Hot Encoding).")
            elif df[col].dtype in ["int64", "float64"]:
                st.write("Recommendation: Use for statistics, visualizations, and ML.")
            if df[col].isnull().sum() > 0:
                st.warning(f"Missing values detected: {df[col].isnull().sum()} — Impute or remove.")
else:
    st.info("Upload a CSV file to start analyzing.")
