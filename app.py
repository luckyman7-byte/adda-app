import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="A-DAA v1 - AI Data Analyst",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("📊 A-DAA v1 — AI Data Analyst App")
st.write("Upload a dataset and choose what you want to do.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    action = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Overview", "EDA", "Visualizations", "Dashboard Mode", "ML Mode", "Column Insights"]
    )

    if action == "Overview":
        st.subheader("📌 Dataset Info")
        st.write(df.describe())
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Duplicates:", df.duplicated().sum())

    if action == "EDA":
        st.subheader("🔍 Exploratory Data Analysis")
        st.write("Basic Statistics")
        st.write(df.describe())

        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            st.write("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for correlation heatmap")

    if action == "Visualizations":
        st.subheader("📈 Visualizations")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns found to visualize")
        else:
            chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Line Plot", "Scatter Plot", "Box Plot"])

            if chart_type == "Histogram":
                col = st.selectbox("Choose Column", numeric_cols)
                fig = px.histogram(df, x=col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Line Plot":
                col = st.selectbox("Choose Column", numeric_cols)
                fig = px.line(df, y=col)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Scatter Plot":
                x = st.selectbox("X-axis", numeric_cols)
                y = st.selectbox("Y-axis", numeric_cols)
                fig = px.scatter(df, x=x, y=y)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Box Plot":
                col = st.selectbox("Choose Column", numeric_cols)
                fig = px.box(df, y=col)
                st.plotly_chart(fig, use_container_width=True)

    if action == "Dashboard Mode":
        st.subheader("📊 Dashboard Mode (Auto KPIs + Filters)")

        col_filter = st.selectbox("Select Column to Filter", df.columns)
        unique_vals = df[col_filter].unique()
        selected_val = st.selectbox("Filter Value", unique_vals)
        filtered_df = df[df[col_filter] == selected_val]

        st.write("Filtered Data:")
        st.dataframe(filtered_df, use_container_width=True)

        numeric_cols = filtered_df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:
            st.subheader("📌 KPIs")
            col = st.selectbox("Select Column for KPIs", numeric_cols)

            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Mean", round(filtered_df[col].mean(), 2))
            kpi2.metric("Max", round(filtered_df[col].max(), 2))
            kpi3.metric("Min", round(filtered_df[col].min(), 2))

            st.subheader("📈 Chart")
            fig = px.line(filtered_df, y=col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns to show KPIs")

    if action == "ML Mode":
        st.subheader("🤖 Machine Learning Mode (Regression)")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for ML mode")
        else:
            target = st.selectbox("Select Target Column", numeric_cols)
            features = st.multiselect("Select Feature Columns", [c for c in numeric_cols if c!= target],
                                    default=[c for c in numeric_cols if c!= target][:3])

            if st.button("Train Model"):
                if not features:
                    st.error("Select at least 1 feature column")
                else:
                    X = df[features].dropna()
                    y = df.loc[X.index, target]

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

    if action == "Column Insights":
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
    st.info("Upload a CSV file to start analyzing")
