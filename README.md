---
# 🧠 AI Monitoring & Ethics Dashboard

A professional **end-to-end Responsible AI project** that demonstrates how to **monitor, evaluate, and mitigate bias** in machine learning models using **Fairlearn** and **SHAP**, and visualize results through an interactive **Streamlit dashboard**.

---

## 📋 Project Overview

This project shows how organizations can **apply Responsible AI principles** by detecting and mitigating algorithmic bias.
It uses the **Adult Income Dataset** to predict whether an individual earns more than **$50K/year**, while ensuring fairness across sensitive attributes such as **gender** and **race**.

---

## ⚙️ Tech Stack

* 🐍 **Python 3.11**
* 🤖 **Scikit-learn** – model training (Logistic Regression)
* ⚖️ **Fairlearn** – fairness evaluation & mitigation
* 🧩 **SHAP** – model explainability (global & local)
* 🌐 **Streamlit** – interactive dashboard
* 📊 **Plotly** & **Matplotlib** – data visualization
* 📦 **Pandas / NumPy / Joblib / TQDM** – data processing utilities

---

## 🧩 Project Structure

```
ai-ethics-dashboard/
│
├── app/
│   └── dashboard.py          # Streamlit app (Fairness + SHAP)
│
├── data/
│   └── raw/
│       └── adult.csv         # Dataset (Adult Income)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_fairness_analysis.ipynb
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

1. **Clone this repository:**

   ```bash
   git clone https://github.com/diegoscodes/ai-ethics-dashboard.git
   cd ai-ethics-dashboard
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate     # (Windows)
   source .venv/bin/activate    # (macOS/Linux)
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the dashboard:**

   ```bash
   streamlit run app/dashboard.py
   ```

---

## 📊 Key Features

✅ **Fairness Analysis (Before vs After)**
 Compare accuracy, recall, and selection rate across sensitive groups.

✅ **Bias Mitigation (Equal Opportunity)**
 Applies Fairlearn’s ThresholdOptimizer to balance true positive rates.

✅ **Explainability (SHAP)**
 Visualizes global feature importance and local prediction insights.

✅ **Sensitive Attribute Selection**
 Switch between *gender* and *race* to analyze fairness from different perspectives.

---

## 🔎 Results Summary

| Attribute        | Before Mitigation         | After Mitigation         | Observation                   |
| ---------------- | ------------------------- | ------------------------ | ----------------------------- |
| **Gender (sex)** | Recall (M: 0.50, F: 0.20) | Recall (M/F ≈ 0.36)      | Balanced recall achieved      |
| **Race**         | Moderate bias gap         | Reduced after mitigation | Fairer classification balance |

> After mitigation, the model achieved **~0.81 accuracy** with significantly reduced bias, proving fairness can coexist with good performance.

---

## 🧠 Learnings

* Bias often mirrors **real-world inequality** present in data.
* Responsible AI focuses on **understanding, not hiding**, sensitive variables.
* **Fairness ≠ perfection** — it’s an **ongoing monitoring process**.

---

## 📷 Preview

*Add a screenshot or GIF of your Streamlit dashboard here.*
Example:
![Dashboard Preview](app/assets/dashboard_preview.png)

---

## 👤 Author

**Diego Ferreira**
🌍 [LinkedIn](https://www.linkedin.com/in/diegoscodes) • 💻 [GitHub](https://github.com/diegoscodes)

---

🧩 *Built as part of an AI & Machine Learning professional portfolio project demonstrating ethical, explainable, and fair model development.*
