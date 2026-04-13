# 👥 People Analytics — Employee Attrition AI Dashboard

A professional, end-to-end People Analytics project using machine learning to predict employee attrition with a beautiful Streamlit dashboard.

---

## 🗂️ Project Structure

```
people_analytics/
├── data/
│   └── raw.csv                  # IBM HR Analytics Dataset (1,470 employees)
├── models/
│   ├── attrition_model.pkl      # Trained ensemble model
│   ├── scaler.pkl               # StandardScaler
│   ├── label_encoders.pkl       # Categorical encoders
│   ├── feature_cols.pkl         # Feature column list
│   ├── metrics.json             # Model performance metrics
│   ├── dept_attrition.csv       # Precomputed department stats
│   ├── role_attrition.csv       # Precomputed role stats
│   └── age_attrition.csv        # Precomputed age stats
├── src/
│   └── train_model.py           # Full training pipeline
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (already done — skip if models/ exists)
```bash
python src/train_model.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🤖 Model Architecture

| Model | Weight | Details |
|---|---|---|
| XGBoost | 3× | 300 trees, depth 5, LR 0.05, scale_pos_weight=5 |
| Gradient Boosting | 2× | 200 trees, depth 4, LR 0.05, subsample 0.8 |
| Random Forest | 2× | 300 trees, depth 8, balanced class weights |
| Logistic Regression | 1× | L2 regularized, balanced class weights |

**Ensemble type:** Soft Voting (weighted average of probabilities)

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 81.6% |
| Precision | 43.1% |
| Recall | 46.8% |
| F1 Score | 44.9% |
| ROC-AUC | 73.5% |
| **CV ROC-AUC (5-fold)** | **82.4% ± 2.0%** |

> Note: Precision/Recall are intentionally balanced for imbalanced data (16% attrition rate). 
> The model uses SMOTE oversampling and a custom decision threshold (0.35) to optimize recall.

---

## 🔬 Feature Engineering

5 additional features were derived from raw columns:

- **IncomePerYear** — MonthlyIncome × 12
- **TenureRatio** — YearsAtCompany / (TotalWorkingYears + 1)
- **SatisfactionScore** — Average of Job/Env/Relationship/WorkLife satisfaction
- **CareerGrowthScore** — JobLevel / (TotalWorkingYears + 1)
- **PromotionLag** — YearsSinceLastPromotion / (YearsAtCompany + 1)

---

## 📱 Dashboard Features

| Tab | Contents |
|---|---|
| 🔮 Prediction | Live attrition prediction, probability gauge, risk signals, HR recommendations |
| 📊 Analytics | Department/Role/Age/Travel/Marital attrition analysis, income distribution |
| 🏆 Model Insights | Feature importances, radar performance chart, confusion matrix, architecture |
| 📋 Data Explorer | Interactive scatter plot, correlation heatmap, raw data table |

---

## 📦 Dataset

IBM HR Analytics Employee Attrition & Performance Dataset
- **1,470 employees**, **35 features**
- **16.1% attrition rate** (imbalanced → handled with SMOTE)
- No missing values
