import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page Config
st.set_page_config(page_title="💲 AstroFinance: TrendSeer", layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: bold;
        text-shadow: 1px 1px 2px #000;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(45deg, #ff6ec4, #7873f5);
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.4em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(45deg, #ff9a8b, #ff6ec4);
    }
    .stSelectbox>div>div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stSlider>div>div {
        color: white;
    }
    .stRadio>div>label {
        color: white !important;
    }
    .stMetric>div {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        padding: 1em;
        border-radius: 15px;
        color: black;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    .css-1kyxreq, .css-ffhzg2, .css-1y4p8pa, .css-1x8cf1d {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💲 AstroFinance: TrendSeer")
st.markdown("""
Welcome to **AstroFinance: TrendSeer** — a colorful platform to explore, analyze, and predict financial time series data.
Upload your CSV, choose features, train a model, and enjoy bright visualizations with galactic flair.
""")

# --- Upload Data ---
st.sidebar.header("📥 Upload Financial Dataset")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Data uploaded successfully!")

    # Clean columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    if len(numeric_cols) < 2:
        st.error("Not enough numeric columns to train model.")
    else:
        # Feature Selection
        st.subheader("🎯 Feature Selection")
        features = st.multiselect("Choose input features", numeric_cols[:-1], default=numeric_cols[:-2])
        target = st.selectbox("Select prediction target", numeric_cols)

        if features and target:
            X = df[features].dropna()
            y = df[target].loc[X.index]

            # Train-test split
            test_size = st.slider("Test size (%)", 10, 50, 20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            # Model
            st.subheader("🧠 Model Training")
            model_type = st.radio("Choose Model", ["Linear Regression", "Random Forest"])
            if st.button("🚀 Train Model"):
                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # Metrics
                st.subheader("📈 Evaluation")
                st.metric("R²", f"{r2_score(y_test, preds):.3f}")
                st.metric("MSE", f"{mean_squared_error(y_test, preds):.3f}")

                # Plot
                st.subheader("🌠 Actual vs Predicted")
                plot_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
                fig = px.line(plot_df, labels={'value': 'Price'}, title="Actual vs Predicted Trend")
                fig.update_layout(template="plotly")
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv = plot_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results CSV", csv, "predictions.csv")
else:
    st.info("Upload a CSV to begin.")
