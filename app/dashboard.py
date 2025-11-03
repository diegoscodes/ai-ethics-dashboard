# app/dashboard.py
# AI Monitoring & Ethics Dashboard – Fairness + SHAP
# Run with: streamlit run app/dashboard.py

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="AI Monitoring & Ethics Dashboard", layout="wide")

# -----------------------------------------------------------
# Load dataset (absolute path fix)
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "..", "data", "raw", "adult.csv")
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        st.error(f"❌ Dataset not found at: {file_path}")
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["sex", "race", "income"])
    return df

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def encode_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include="object"):
        if col != "income":  # don't re-encode target
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def metricframe_table(y_true, y_pred, sensitive, index_map={0: "Female", 1: "Male"}):
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "recall": recall_score, "selection_rate": selection_rate},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    return mf.by_group.rename(index=index_map)

def fairness_gaps(y_true, y_pred, sensitive):
    dp = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    eo = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)
    return dp, eo

def plot_group_bars(title, before_series: pd.Series, after_series: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=before_series.index.tolist(), y=before_series.values.tolist()))
    fig.add_trace(go.Bar(name="After", x=after_series.index.tolist(), y=after_series.values.tolist()))
    fig.update_layout(title=title, barmode="group", xaxis_title="", yaxis_title="Score")
    return fig

# -----------------------------------------------------------
# Data Preparation & Model Training
# -----------------------------------------------------------
df_raw = load_data()
st.sidebar.title("Settings")
st.sidebar.write("Dataset: Adult Income (Census)")

# Ensure correct target encoding
st.sidebar.subheader("Unique income values (raw):")
unique_vals = list(df_raw["income"].unique())
st.sidebar.text(", ".join(unique_vals))

df_raw["income"] = df_raw["income"].apply(lambda x: 1 if ">50K" in x else 0)

st.sidebar.subheader("Target distribution (after encoding):")
dist = df_raw["income"].value_counts().to_dict()
st.sidebar.text(f"0 (<=50K): {dist.get(0,0)}")
st.sidebar.text(f"1 (>50K): {dist.get(1,0)}")


# Encode & split
df_enc, encoders = encode_dataframe(df_raw)
X = df_enc.drop("income", axis=1)
y = df_enc["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# -----------------------------
# Sensitive feature selection
# -----------------------------
sensitive_option = st.sidebar.selectbox(
    "Select sensitive attribute for fairness analysis:",
    options=["sex", "race"],
    index=0
)
if sensitive_option == "sex":
    label_name = "Gender"
else:
    label_name = "Race"


sensitive = df_enc.loc[X_test.index, sensitive_option]

# Map names for display
if sensitive_option == "sex":
    sensitive_map = {0: "Female", 1: "Male"}
else:
    sensitive_map = {i: f"Group_{i}" for i in sorted(sensitive.unique())}


# Scale & train
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

model = LogisticRegression(solver="saga", max_iter=5000)
model.fit(X_train_sc, y_train)

# Predictions before mitigation
y_pred_before = model.predict(X_test_sc)

# Mitigation (Equal Opportunity)
postproc = ThresholdOptimizer(
    estimator=model,
    constraints="true_positive_rate_parity",
    predict_method="predict_proba"
)
postproc.fit(X_test_sc, y_test, sensitive_features=sensitive)
y_pred_after = postproc.predict(X_test_sc, sensitive_features=sensitive)

# MetricFrames
mf_before = metricframe_table(y_test, y_pred_before, sensitive, index_map=sensitive_map)
mf_after  = metricframe_table(y_test, y_pred_after, sensitive, index_map=sensitive_map)


dp_before, eo_before = fairness_gaps(y_test, y_pred_before, sensitive)
dp_after,  eo_after  = fairness_gaps(y_test, y_pred_after, sensitive)

# -----------------------------------------------------------
# Dashboard Layout
# -----------------------------------------------------------
st.title("AI Monitoring & Ethics Dashboard")
st.caption("Responsible AI • Fairness • Explainability (SHAP) • Adult Income Dataset")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (before)", f"{accuracy_score(y_test, y_pred_before):.3f}")
col2.metric("Accuracy (after)",  f"{accuracy_score(y_test, y_pred_after):.3f}")
col3.metric("Demographic Parity Diff (↓)", f"{dp_before:.3f}", delta=f"{(dp_after - dp_before):.3f}")
col4.metric("Equalized Odds Diff (↓)",     f"{eo_before:.3f}", delta=f"{(eo_after - eo_before):.3f}")

st.markdown("---")

# Tables
c1, c2 = st.columns(2)
with c1:
    st.subheader(f"Before Mitigation – Metrics by {label_name}")
    st.dataframe(mf_before.style.format("{:.3f}"))
with c2:
    st.subheader(f"After Mitigation (Equal Opportunity) – Metrics by {label_name}")
    st.dataframe(mf_after.style.format("{:.3f}"))


# Charts
st.subheader("Group Comparison – Before vs After")
st.plotly_chart(plot_group_bars("Recall (Equal Opportunity target)", mf_before["recall"], mf_after["recall"]), use_container_width=True)
st.plotly_chart(plot_group_bars("Selection Rate (Demographic Parity proxy)", mf_before["selection_rate"], mf_after["selection_rate"]), use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------
# SHAP Explainability
# -----------------------------------------------------------
st.header("Model Explainability (SHAP)")

max_rows = st.sidebar.slider("SHAP sample size", 200, min(3000, len(X_test_sc)), 800, 100)
X_shap = X_test_sc[:max_rows]
feature_names = X.columns.tolist()

explainer = shap.LinearExplainer(model, X_train_sc, feature_names=feature_names)
shap_values = explainer.shap_values(X_shap)

st.subheader("Global Feature Importance (SHAP Summary)")
fig, ax = plt.subplots(figsize=(8, 5))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
st.pyplot(fig, clear_figure=True)

# Local explanation
st.subheader("Local Explanation (Single Prediction)")
idx = st.number_input("Select row index from test set", 0, len(X_shap)-1, 0)
row = X_shap[idx:idx+1]
pred_proba = float(model.predict_proba(row)[0, 1])
pred_label = int(pred_proba >= 0.5)
st.write(f"Predicted probability (>50K): **{pred_proba:.3f}** | Predicted label: **{pred_label}**")

try:
    wf = shap.plots._waterfall.waterfall_legacy(
        shap.Explanation(values=shap_values[idx],
                         base_values=explainer.expected_value,
                         data=row,
                         feature_names=feature_names)
    )
    st.pyplot(wf.figure, clear_figure=True)
except Exception:
    contrib = pd.Series(shap_values[idx], index=feature_names).sort_values(key=np.abs, ascending=False)[:10]
    fig_bar, ax_bar = plt.subplots(figsize=(7,4))
    contrib.plot(kind="bar", ax=ax_bar)
    ax_bar.set_title("Top SHAP Contributions (|value|)")
    st.pyplot(fig_bar, clear_figure=True)

st.markdown("---")
st.caption("Trains a Logistic Regression and applies Fairlearn post-processing (Equal Opportunity) to align recall across gender groups, then uses SHAP for global and local interpretability.")
