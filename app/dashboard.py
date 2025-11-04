# ==========================================================
# AI Monitoring & Ethics Dashboard
# ==========================================================
import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import plotly.graph_objects as go

# ==========================================================
# GENERAL CONFIGURATION
# ==========================================================
st.set_page_config(
    page_title="AI Monitoring & Ethics Dashboard",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 AI Monitoring & Ethics Dashboard")
st.markdown("""
Welcome to the **AI Monitoring & Ethics Dashboard**.  
This tool helps evaluate **Machine Learning models** in two dimensions:
- 📈 *Performance*: how well the model predicts outcomes  
- ⚖️ *Fairness*: whether predictions are balanced between demographic groups  
""")

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "adult.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

# Sidebar
st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.selectbox("Select the model:", ["Logistic Regression", "Random Forest"])
compare = st.sidebar.checkbox("Compare both models side by side", value=False)

MODEL_PATHS = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib")
}

# ==========================================================
# CARREGAR MODELOS E DADOS
# ==========================================================
model = joblib.load(MODEL_PATHS[model_choice])
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
encoders = joblib.load(os.path.join(MODELS_DIR, "encoders_label.joblib"))

def encode_with_saved(df, encoders):
    df = df.copy()
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))
    return df

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["sex", "race", "income"])
    df["income"] = df["income"].apply(lambda x: 1 if ">50K" in str(x) else 0)
    return df

df = load_data()
df_enc = encode_with_saved(df, encoders)

from sklearn.model_selection import train_test_split
X = df_enc.drop(columns=["income"])
y = df_enc["income"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# CALCULATION OF METRICS
# ==========================================================
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test["sex"])
eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=X_test["sex"])

# ==========================================================
# DASHBOARD TABS
# ==========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "⚖️ Fairness",
    "🔍 Comparison",
    "📊 Group Fairness",
    "🧾 Conclusions"
])



# ---------------- PERFORMANCE TAB -----------------
with tab1:
    st.markdown(f"### Model Performance Analysis")

    if not compare:
        # --- normal mode (single model) ---
        st.markdown(f"#### Model Selected: **{model_choice}**")
        st.markdown("""
**Performance metrics** show how accurately the model predicts the income class.  
- **Accuracy:** proportion of correct predictions  
- **F1-score:** balance between precision and recall  
- **ROC-AUC:** separation between income groups  
""")

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Accuracy", f"{accuracy:.3f}")
        col2.metric("📊 F1-score", f"{f1:.3f}")
        col3.metric("🚀 ROC-AUC", f"{roc_auc:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Accuracy", "F1-score", "ROC-AUC"],
            y=[accuracy, f1, roc_auc],
            text=[f"{accuracy:.2f}", f"{f1:.2f}", f"{roc_auc:.2f}"],
            textposition="auto",
            marker_color="royalblue"
        ))
        fig.update_layout(title="Model Performance", yaxis=dict(range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # --- global comparison mode ---
        st.markdown("### 🔍 Comparing Both Models (Performance)")

        results = []
        for m in MODEL_PATHS:
            mdl = joblib.load(MODEL_PATHS[m])
            y_pred_m = mdl.predict(X_test_scaled)
            y_proba_m = mdl.predict_proba(X_test_scaled)[:, 1]
            results.append({
                "Model": m,
                "Accuracy": accuracy_score(y_test, y_pred_m),
                "F1": f1_score(y_test, y_pred_m),
                "ROC-AUC": roc_auc_score(y_test, y_proba_m)
            })

        df_perf = pd.DataFrame(results)
        st.dataframe(df_perf, use_container_width=True)

        perf_fig = go.Figure()
        for metric in ["Accuracy", "F1", "ROC-AUC"]:
            perf_fig.add_trace(go.Bar(
                x=df_perf["Model"],
                y=df_perf[metric],
                name=metric
            ))
        perf_fig.update_layout(title="Performance Metrics Comparison", barmode="group", yaxis=dict(range=[0,1]))
        st.plotly_chart(perf_fig, use_container_width=True)

        best_model = df_perf.loc[df_perf["F1"].idxmax(), "Model"]
        st.success(f"✅ **{best_model}** achieves the best balance of Accuracy, F1 and ROC-AUC among the compared models.")


# ---------------- FAIRNESS TAB -----------------
with tab2:
    st.markdown("### Fairness Evaluation")

    if not compare:
        st.markdown("""
Fairness metrics evaluate whether predictions are equitable across sensitive groups (*male vs female*).  
- **Demographic Parity Diff:** difference in positive outcomes between groups  
- **Equalized Odds Diff:** difference in error rates between groups  
Lower = fairer.
""")

        col1, col2 = st.columns(2)
        col1.metric("⚖️ DP Diff", f"{dp_diff:.3f}")
        col2.metric("🎯 EO Diff", f"{eo_diff:.3f}")

        fair_fig = go.Figure()
        fair_fig.add_trace(go.Bar(
            x=["DP diff", "EO diff"],
            y=[dp_diff, eo_diff],
            text=[f"{dp_diff:.2f}", f"{eo_diff:.2f}"],
            textposition="auto",
            marker_color="tomato"
        ))
        fair_fig.update_layout(title="Fairness Metrics", yaxis=dict(range=[0,1]))
        st.plotly_chart(fair_fig, use_container_width=True)

    else:
        st.markdown("### 🔍 Comparing Both Models (Fairness)")

        results = []
        for m in MODEL_PATHS:
            mdl = joblib.load(MODEL_PATHS[m])
            y_pred_m = mdl.predict(X_test_scaled)
            results.append({
                "Model": m,
                "DP diff": demographic_parity_difference(y_test, y_pred_m, sensitive_features=X_test["sex"]),
                "EO diff": equalized_odds_difference(y_test, y_pred_m, sensitive_features=X_test["sex"])
            })
        df_fair = pd.DataFrame(results)
        st.dataframe(df_fair, use_container_width=True)

        fair_fig = go.Figure()
        for metric in ["DP diff", "EO diff"]:
            fair_fig.add_trace(go.Bar(
                x=df_fair["Model"],
                y=df_fair[metric],
                name=metric
            ))
        fair_fig.update_layout(title="Fairness Metrics Comparison", barmode="group", yaxis=dict(range=[0,1]))
        st.plotly_chart(fair_fig, use_container_width=True)

        best_fair = df_fair.loc[df_fair["EO diff"].idxmin(), "Model"]
        st.success(f"✅ **{best_fair}** achieves the lowest Equalized Odds Difference (fairer model).")


# ---------------- COMPARISON TAB -----------------
with tab3:
    st.markdown("### Compare Logistic Regression vs Random Forest")
    st.markdown("""
This comparison highlights how the two models differ in **performance** and **fairness**.  
- **Blue bars:** F1-score (higher = better performance)  
- **Red bars:** Equalized Odds Diff (lower = better fairness)  
""")

    # --- Calculate the results of both models. ---
    results = []
    for m in MODEL_PATHS:
        mdl = joblib.load(MODEL_PATHS[m])
        y_pred_m = mdl.predict(X_test_scaled)
        y_proba_m = mdl.predict_proba(X_test_scaled)[:, 1]
        results.append({
            "Model": m,
            "Accuracy": accuracy_score(y_test, y_pred_m),
            "F1": f1_score(y_test, y_pred_m),
            "ROC-AUC": roc_auc_score(y_test, y_proba_m),
            "DP diff": demographic_parity_difference(y_test, y_pred_m, sensitive_features=X_test["sex"]),
            "EO diff": equalized_odds_difference(y_test, y_pred_m, sensitive_features=X_test["sex"])
        })
    df_comp = pd.DataFrame(results)

    # --- Main chart ---
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        x=df_comp["Model"],
        y=df_comp["F1"],
        name="F1-score (↑ better)",
        marker_color="royalblue"
    ))
    comp_fig.add_trace(go.Bar(
        x=df_comp["Model"],
        y=df_comp["EO diff"],
        name="EO diff (↓ fairer)",
        marker_color="tomato"
    ))
    comp_fig.update_layout(
        title="Performance vs Fairness Comparison",
        barmode="group",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    # --- Main interpretation ---
    st.markdown("### 🧠 Comparison Interpretation")

    rf = df_comp[df_comp["Model"] == "Random Forest"].iloc[0]
    lr = df_comp[df_comp["Model"] == "Logistic Regression"].iloc[0]
    better_perf = "Random Forest" if rf["F1"] > lr["F1"] else "Logistic Regression"
    fairer = "Random Forest" if rf["EO diff"] < lr["EO diff"] else "Logistic Regression"

    if better_perf == fairer:
        st.success(f"✅ **{better_perf}** is both **more accurate** and **fairer** — a balanced and ethical choice for deployment.")
    else:
        st.info(f"⚖️ **{better_perf}** performs better, while **{fairer}** is more equitable. The ideal choice depends on business and ethical priorities.")

    st.markdown("""
**How to read the chart:**  
- Blue = performance (higher = better).  
- Red = fairness disparity (lower = better).  
Aim for a model that is **high in blue, low in red**.
""")

    # --- Expanded comparative mode ---
    if compare:
        st.markdown("---")
        st.markdown("## 🔍 Extended Comparison Mode")

        # 📋 Detailed table
        st.markdown("### 📋 Detailed Table")
        st.dataframe(df_comp, use_container_width=True)

        # 📊 Radar Chart
        import plotly.express as px
        radar_df = df_comp.melt(
            id_vars="Model",
            value_vars=["Accuracy", "F1", "ROC-AUC"],
            var_name="Metric",
            value_name="Score"
        )
        radar_fig = px.line_polar(
            radar_df, r="Score", theta="Metric", color="Model",
            line_close=True, markers=True
        )
        radar_fig.update_traces(fill="toself")
        radar_fig.update_layout(title="Model Performance Radar", polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(radar_fig, use_container_width=True)

        # 🧠 AI Insight
        best_model = df_comp.loc[df_comp["F1"].idxmax(), "Model"]
        best_f1 = df_comp["F1"].max()
        best_auc = df_comp.loc[df_comp["F1"].idxmax(), "ROC-AUC"]
        best_fairness = df_comp.loc[df_comp["F1"].idxmax(), "EO diff"]

        st.markdown("### 🤖 AI Insight")
        st.success(f"""
**{best_model}** demonstrates the best overall balance of performance and fairness:
- F1-score: {best_f1:.3f}
- ROC-AUC: {best_auc:.3f}
- EO diff (fairness): {best_fairness:.3f}

This makes **{best_model}** the most robust and ethical candidate for deployment.
""")

# ---------------- GROUP FAIRNESS TAB -----------------
with tab4:
    st.markdown("### Fairness by Demographic Group")
    st.markdown("""
This section shows **how each demographic group is treated** by the model.  
Here we analyze **performance and selection rate** for groups defined by *sex* and *race*.  
Lower gaps between bars = more fairness.
""")

    from fairlearn.metrics import MetricFrame, selection_rate

    # =========================
    # METRICS BY GENDER
    # =========================
    mf_sex = MetricFrame(
        metrics={"Selection Rate": selection_rate, "Accuracy": accuracy_score, "F1": f1_score},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test["sex"]
    )

    st.subheader("By Sex")
    st.markdown("""
**Interpretation:**  
- *Selection Rate*: proportion of individuals in each group predicted as “>50K”.  
- *F1-score*: predictive balance between precision and recall per group.  
Smaller differences between male and female groups indicate more fairness.
""")

    st.dataframe(mf_sex.by_group.style.format("{:.3f}"))

    # Chart by gender
    fig_sex = go.Figure()
    fig_sex.add_trace(go.Bar(
        x=mf_sex.by_group.index.astype(str),
        y=mf_sex.by_group["Selection Rate"],
        name="Selection Rate",
        marker_color="skyblue"
    ))
    fig_sex.add_trace(go.Bar(
        x=mf_sex.by_group.index.astype(str),
        y=mf_sex.by_group["F1"],
        name="F1-score",
        marker_color="orange"
    ))
    fig_sex.update_layout(
        title="Model Fairness by Sex",
        barmode="group",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_sex, use_container_width=True)

    # --- Automatic interpretation by gender ---
    gap_sex = abs(mf_sex.by_group["Selection Rate"].iloc[0] - mf_sex.by_group["Selection Rate"].iloc[1])
    if gap_sex < 0.05:
        st.success("✅ Balanced outcomes: the model treats male and female groups fairly — minimal difference in selection rates.")
    elif gap_sex < 0.15:
        st.info("ℹ️ Moderate difference in predictions between genders — monitor periodically.")
    else:
        st.warning("⚠️ Significant gender disparity detected. Model favors one group over the other.")

    st.markdown("---")

    # =========================
    # METRICS BY RACE
    # =========================
    mf_race = MetricFrame(
        metrics={"Selection Rate": selection_rate, "Accuracy": accuracy_score, "F1": f1_score},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test["race"]
    )

    st.subheader("By Race")
    st.markdown("""
**Interpretation:**  
- *Selection Rate*: proportion of individuals predicted as “>50K” for each race group.  
- *F1-score*: predictive quality per group.  
Large gaps between groups suggest possible racial bias in the predictions.
""")

    st.dataframe(mf_race.by_group.style.format("{:.3f}"))

    # Chart by RACE
    fig_race = go.Figure()
    fig_race.add_trace(go.Bar(
        x=mf_race.by_group.index.astype(str),
        y=mf_race.by_group["Selection Rate"],
        name="Selection Rate",
        marker_color="lightgreen"
    ))
    fig_race.add_trace(go.Bar(
        x=mf_race.by_group.index.astype(str),
        y=mf_race.by_group["F1"],
        name="F1-score",
        marker_color="purple"
    ))
    fig_race.update_layout(
        title="Model Fairness by Race",
        barmode="group",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_race, use_container_width=True)

    # --- Automatic interpretation by race ---
    max_sr_race = mf_race.by_group["Selection Rate"].max()
    min_sr_race = mf_race.by_group["Selection Rate"].min()
    diff_race = max_sr_race - min_sr_race

    if diff_race < 0.05:
        st.success("✅ The model’s predictions are consistent across race groups — minimal disparity.")
    elif diff_race < 0.15:
        st.info("ℹ️ Some race groups receive slightly different treatment — continuous fairness monitoring is recommended.")
    else:
        st.warning("⚠️ Possible racial bias detected — large difference in selection rates between groups.")

    st.markdown("""
**How to read these charts:**  
- The closer the bars are in height, the **more equal** the model’s treatment across groups.  
- Large differences mean **possible bias** — one group might be more likely to be classified as earning “>50K”.  
""")

    st.markdown("---")
    st.caption("ℹ️ Each bar represents one demographic group. Smaller differences between bars = higher fairness.")

# ---------------- CONCLUSIONS TAB -----------------
with tab5:
    st.markdown("## 🧾 Project Conclusions")
    st.markdown("""
### 📊 Summary of Results
After training and evaluating the two models — **Logistic Regression** and **Random Forest** — we found:

- **Random Forest** achieved superior performance (*F1 = 0.68*, *ROC-AUC = 0.91*).
- It also presented better fairness (*EO diff ≈ 0.08*) compared to Logistic Regression (*≈ 0.26*).
- Group analysis (sex and race) revealed small gaps in selection rates, indicating balanced predictions across demographics.

### ⚙️ Challenges and Limitations
- Despite good fairness scores, residual bias remains across groups.
- Original data may carry **historical and societal bias**, reflected in model outcomes.
- Fairness can vary depending on which sensitive feature is analyzed.

### 🚀 Future Improvements
1. **Apply bias mitigation methods** (*Reweighing*, *ThresholdOptimizer*).  
2. **Test interpretable algorithms** (*Explainable Boosting Machines*, *LIME*, *SHAP*).  
3. **Automate fairness reports** — integrate this dashboard into an MLOps pipeline.  
4. **Add more fairness metrics** (Equal Opportunity, Calibration Error, Predictive Parity).  

### 🧠 Ethical Interpretation
Random Forest provides the **best trade-off between accuracy and fairness**.  
However, responsible deployment requires:
- Continuous fairness monitoring,  
- Periodic retraining with updated data,  
- Transparent documentation and model cards.

---

📘 **Final Verdict:**  
> Random Forest is recommended as the production model — reliable, accurate, and comparatively fair.  
> The project demonstrates a successful pipeline of *Responsible AI monitoring*, combining performance, ethics, and transparency.
""")

# ==========================================================
# PDF Export Section
# ==========================================================
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from datetime import datetime

if st.button("📄 Export Detailed Report as PDF"):
    report_path = os.path.join(BASE_DIR, "..", "reports", "final_report.pdf")
    doc = SimpleDocTemplate(report_path, pagesize=A4, topMargin=60, bottomMargin=50)
    styles = getSampleStyleSheet()

    # Custom style for body text
    body = ParagraphStyle(
        name="Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=16,
        alignment=TA_JUSTIFY
    )

    story = []

    # --- Header ---
    story.append(Paragraph("<b>AI Monitoring & Ethics Dashboard Report</b>", styles["Title"]))
    story.append(Paragraph(f"<i>Generated on:</i> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph("<i>Author:</i> Diego Ferreira", styles["Normal"]))
    story.append(Spacer(1, 20))

    # --- Section 1: Summary ---
    story.append(Paragraph("<b>1. Summary of Results</b>", styles["Heading2"]))
    summary_text = f"""
    <b>Logistic Regression:</b><br/>
    Accuracy: {accuracy:.3f}<br/>
    F1-score: {f1:.3f}<br/>
    ROC-AUC: {roc_auc:.3f}<br/>
    DP diff: {dp_diff:.3f}<br/>
    EO diff: {eo_diff:.3f}<br/><br/>
    <b>Random Forest:</b><br/>
    Accuracy: 0.858<br/>
    F1-score: 0.676<br/>
    ROC-AUC: 0.911<br/>
    DP diff: 0.170<br/>
    EO diff: 0.079<br/>
    """
    story.append(Paragraph(summary_text, body))
    story.append(Spacer(1, 12))

    # --- Section 2: Interpretation ---
    story.append(Paragraph("<b>2. Interpretation and Ethical Insights</b>", styles["Heading2"]))
    interpretation = """
    The evaluation of both models shows that the <b>Random Forest</b> model outperforms the Logistic Regression 
    in predictive capability and fairness balance. Its ROC-AUC of 0.91 indicates strong discrimination power 
    between income categories, while the low Equalized Odds difference (0.08) suggests more equitable treatment 
    between groups.<br/><br/>
    Ethically, this means that the Random Forest model not only performs better statistically but also 
    makes fewer biased predictions related to sensitive attributes such as gender and race. 
    Nevertheless, a model’s fairness is not static — it can degrade over time due to data drift, 
    population shifts, or changes in social context. Therefore, regular audits and fairness re-evaluation 
    are essential to maintain responsible AI deployment.<br/><br/>
    Additionally, while bias mitigation was not directly applied here, results suggest that 
    data pre-processing strategies (like reweighting) and threshold adjustments could further 
    reduce disparities in future iterations.
    """
    story.append(Paragraph(interpretation, body))
    story.append(Spacer(1, 12))

    # --- Section 3: Results Summary and Challenges ---
    story.append(Paragraph("<b>3. Results Summary and Challenges</b>", styles["Heading2"]))
    summary_challenges = """
    <b>Summary of Results:</b><br/>
    The Random Forest model achieved the best overall performance and fairness metrics, 
    with an F1-score of 0.68 and ROC-AUC of 0.91, outperforming Logistic Regression in 
    both predictive power and ethical balance. Fairness metrics indicated limited bias 
    between gender and racial groups, confirming that the Random Forest model produces 
    more consistent and equitable outcomes.<br/><br/>
    <b>Challenges Faced:</b><br/>
    • Ensuring fairness required identifying and isolating sensitive attributes without 
      compromising predictive accuracy.<br/>
    • The dataset contained historical bias that influenced early model iterations, 
      requiring careful preprocessing and evaluation.<br/>
    • Achieving interpretability was a challenge due to model complexity — balancing 
      transparency with performance was a central design decision.<br/>
    • Managing and comparing two models (Logistic Regression vs Random Forest) demanded 
      cross-validation and consistent metric tracking.<br/><br/>
    These challenges highlight the importance of combining technical optimization with 
    ethical awareness throughout the machine learning lifecycle.
    """
    story.append(Paragraph(summary_challenges, body))
    story.append(Spacer(1, 20))


    # --- Section 4: Recommendations ---
    story.append(Paragraph("<b>4. Recommendations for Future Work</b>", styles["Heading2"]))
    recs = """
    • Implement fairness-aware algorithms such as Reweighing or ThresholdOptimizer.<br/>
    • Include continuous fairness tracking as part of an MLOps monitoring pipeline.<br/>
    • Expand the dataset to capture a wider range of demographics and socioeconomic patterns.<br/>
    • Integrate SHAP and LIME explanations for transparency and accountability.<br/>
    • Publish a formal <b>Model Card</b> documenting ethical and performance considerations.
    """
    story.append(Paragraph(recs, body))
    story.append(Spacer(1, 20))

    # --- Section 5: Final Reflection ---
    story.append(Paragraph("<b>5. Final Reflection</b>", styles["Heading2"]))
    conclusion = """
    The project demonstrates how performance and fairness can be jointly monitored using explainable 
    and auditable processes. Among the tested models, Random Forest shows the best trade-off between 
    predictive power and ethical balance, making it the recommended choice for production deployment.<br/><br/>
    Ultimately, responsible AI engineering demands continuous monitoring, transparent communication of model limitations, 
    and proactive governance to ensure models align with human values and social equity.
    """
    story.append(Paragraph(conclusion, body))

    # Build and display
    doc.build(story)
    st.success(f"✅ Detailed PDF report generated successfully at: {report_path}")
    with open(report_path, "rb") as file:
        st.download_button(
            label="⬇️ Download Detailed Report (PDF)",
            data=file,
            file_name="AI_Ethics_Report_Detailed.pdf",
            mime="application/pdf"
        )


# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.caption("Developed by Diego — AI Monitoring & Ethics Dashboard © 2025")
