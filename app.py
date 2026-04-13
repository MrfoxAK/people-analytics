"""
╔══════════════════════════════════════════════════════════════╗
║     PEOPLE ANALYTICS DASHBOARD - Employee Attrition AI       ║
║     Built with Streamlit + Scikit-learn + XGBoost            ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import os
import sys

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="People Analytics | Attrition AI",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Auto-train if model missing or incompatible ──────────────────────────────
def model_is_valid():
    required = ["attrition_model.pkl","scaler.pkl","label_encoders.pkl",
                "feature_cols.pkl","metrics.json"]
    if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required):
        return False
    try:
        joblib.load(os.path.join(MODEL_DIR, "attrition_model.pkl"))
        return True
    except Exception:
        return False

if not model_is_valid():
    with st.spinner("🔧 First launch detected — training model on your machine (takes ~60s)..."):
        src_path = os.path.join(BASE_DIR, "src", "train_model.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_model", src_path)
        tm   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tm)
        tm.train(verbose=False)
    st.success("✅ Model trained and ready!")
    st.rerun()

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .sidebar-logo { text-align:center; padding: 1rem 0 2rem; }
    .sidebar-logo h2 { font-size:1.3rem; font-weight:700; color:#7c3aed !important; }
    .sidebar-logo p  { font-size:0.75rem; color:#64748b !important; margin:0; }

    .metric-card {
        background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
        border: 1px solid #2d3748; border-radius: 16px;
        padding: 1.5rem; text-align: center; position: relative; overflow: hidden;
    }
    .metric-card::before {
        content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:16px 16px 0 0;
    }
    .metric-card.purple::before { background: linear-gradient(90deg,#7c3aed,#a855f7); }
    .metric-card.blue::before   { background: linear-gradient(90deg,#2563eb,#3b82f6); }
    .metric-card.green::before  { background: linear-gradient(90deg,#059669,#10b981); }
    .metric-card.amber::before  { background: linear-gradient(90deg,#d97706,#f59e0b); }
    .metric-card.red::before    { background: linear-gradient(90deg,#dc2626,#ef4444); }
    .metric-value  { font-size:2.2rem; font-weight:700; color:#f1f5f9; line-height:1.1; }
    .metric-label  { font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-top:0.4rem; }
    .metric-delta  { font-size:0.8rem; margin-top:0.5rem; font-weight:500; }
    .delta-up      { color:#f87171; }
    .delta-down    { color:#34d399; }

    .section-header {
        display:flex; align-items:center; gap:0.75rem;
        margin: 2rem 0 1rem; padding-bottom:0.75rem; border-bottom:1px solid #2d3748;
    }
    .section-header h3 { font-size:1.1rem; font-weight:600; color:#f1f5f9; margin:0; }
    .section-badge {
        background:#7c3aed22; color:#a78bfa; border:1px solid #7c3aed44;
        padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:600;
    }

    .prediction-box { border-radius:20px; padding:2rem; text-align:center; border:2px solid; }
    .pred-high   { background:linear-gradient(135deg,#450a0a22,#7f1d1d22); border-color:#ef4444; }
    .pred-medium { background:linear-gradient(135deg,#451a0322,#78350f22); border-color:#f59e0b; }
    .pred-low    { background:linear-gradient(135deg,#05260e22,#14532d22); border-color:#10b981; }
    .pred-title  { font-size:1rem; color:#94a3b8; margin-bottom:0.5rem; }
    .pred-result { font-size:2.5rem; font-weight:800; margin:0.5rem 0; }
    .pred-prob   { font-size:1.1rem; font-weight:600; }
    .pred-high .pred-result   { color:#ef4444; }
    .pred-high .pred-prob     { color:#fca5a5; }
    .pred-medium .pred-result { color:#f59e0b; }
    .pred-medium .pred-prob   { color:#fcd34d; }
    .pred-low .pred-result    { color:#10b981; }
    .pred-low .pred-prob      { color:#6ee7b7; }

    .risk-factor {
        display:flex; align-items:center; justify-content:space-between;
        padding:0.65rem 1rem; background:#1e2235; border-radius:10px;
        margin-bottom:0.5rem; border-left:3px solid #7c3aed;
    }
    .risk-factor-name { font-size:0.85rem; color:#cbd5e1; }
    .risk-factor-val  { font-size:0.85rem; font-weight:600; color:#f1f5f9; }

    .insight-card {
        background:#1a1d2e; border:1px solid #2d3748; border-radius:12px;
        padding:1.25rem; margin-bottom:0.75rem;
    }
    .insight-icon  { font-size:1.5rem; }
    .insight-title { font-size:0.85rem; font-weight:600; color:#c4b5fd; margin:0.25rem 0; }
    .insight-body  { font-size:0.8rem; color:#94a3b8; line-height:1.5; }

    .stTabs [data-baseweb="tab-list"] { background:#1a1d2e; border-radius:12px; padding:4px; gap:4px; }
    .stTabs [data-baseweb="tab"]      { border-radius:8px; color:#64748b !important; font-weight:500; font-size:0.85rem; }
    .stTabs [aria-selected="true"]    { background:#7c3aed !important; color:white !important; }
    hr { border-color:#2d3748; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94a3b8', family='Inter'),
    margin=dict(l=10, r=10, t=40, b=10),
    colorway=['#7c3aed','#3b82f6','#10b981','#f59e0b','#ef4444','#06b6d4','#ec4899','#84cc16'],
)
CATEGORICAL_FEATURES = ['BusinessTravel','Department','EducationField',
                         'Gender','JobRole','MaritalStatus','OverTime']
NUMERICAL_FEATURES   = [
    'Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
    'HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome',
    'MonthlyRate','NumCompaniesWorked','PercentSalaryHike','PerformanceRating',
    'RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears',
    'TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany',
    'YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'
]

# ─── Load Assets ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    model          = joblib.load(os.path.join(MODEL_DIR, "attrition_model.pkl"))
    scaler         = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    feature_cols   = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return model, scaler, label_encoders, feature_cols, metrics

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Attrition_bin'] = (df['Attrition'] == 'Yes').astype(int)
    return df

model, scaler, label_encoders, feature_cols, metrics = load_model_artifacts()
df = load_data()

# ─── Prediction Helper ────────────────────────────────────────────────────────
def predict_attrition(inputs: dict):
    row = {}
    for col in CATEGORICAL_FEATURES:
        le  = label_encoders[col]
        val = str(inputs.get(col, le.classes_[0]))
        row[col] = le.transform([val])[0] if val in le.classes_ else 0
    for col in NUMERICAL_FEATURES:
        row[col] = inputs.get(col, 0)
    row['IncomePerYear']     = row['MonthlyIncome'] * 12
    row['TenureRatio']       = row['YearsAtCompany'] / (row['TotalWorkingYears'] + 1)
    row['SatisfactionScore'] = (row['JobSatisfaction'] + row['EnvironmentSatisfaction'] +
                                 row['RelationshipSatisfaction'] + row['WorkLifeBalance']) / 4
    row['CareerGrowthScore'] = row['JobLevel'] / (row['TotalWorkingYears'] + 1)
    row['PromotionLag']      = row['YearsSinceLastPromotion'] / (row['YearsAtCompany'] + 1)
    X    = pd.DataFrame([row])[feature_cols]
    X_sc = scaler.transform(X)
    prob = model.predict_proba(X_sc)[0][1]
    pred = int(prob >= 0.35)
    return pred, prob

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>👥 PeopleAnalytics</h2>
        <p>AI-Powered HR Intelligence</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🔮 Predict Attrition")
    st.markdown("---")

    age            = st.slider("Age", 18, 65, 35)
    monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=500)
    years_at_company   = st.slider("Years at Company", 0, 40, 5)
    job_satisfaction   = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
    env_satisfaction   = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
    work_life_balance  = st.slider("Work Life Balance (1–4)", 1, 4, 3)
    overtime           = st.selectbox("OverTime", ["No", "Yes"])

    with st.expander("⚙️ More Fields", expanded=False):
        department       = st.selectbox("Department", sorted(df['Department'].unique().tolist()))
        job_role         = st.selectbox("Job Role", sorted(df['JobRole'].unique().tolist()))
        business_travel  = st.selectbox("Business Travel", sorted(df['BusinessTravel'].unique().tolist()))
        marital_status   = st.selectbox("Marital Status", sorted(df['MaritalStatus'].unique().tolist()))
        education_field  = st.selectbox("Education Field", sorted(df['EducationField'].unique().tolist()))
        gender           = st.selectbox("Gender", ["Male", "Female"])
        education        = st.slider("Education Level (1–5)", 1, 5, 3)
        job_level        = st.slider("Job Level (1–5)", 1, 5, 2)
        distance_home    = st.slider("Distance from Home (km)", 1, 30, 10)
        total_working_years   = st.slider("Total Working Years", 0, 40, 8)
        years_current_role    = st.slider("Years in Current Role", 0, 18, 3)
        years_last_promotion  = st.slider("Years Since Last Promotion", 0, 15, 2)
        years_with_manager    = st.slider("Years with Current Manager", 0, 17, 3)
        num_companies    = st.slider("Num Companies Worked", 0, 9, 2)
        job_involvement  = st.slider("Job Involvement (1–4)", 1, 4, 3)
        stock_option     = st.slider("Stock Option Level (0–3)", 0, 3, 1)
        percent_hike     = st.slider("Salary Hike %", 11, 25, 15)
        training_times   = st.slider("Training Times Last Year", 0, 6, 3)
        relationship_sat = st.slider("Relationship Satisfaction (1–4)", 1, 4, 3)
        performance_rating = st.slider("Performance Rating (1–4)", 1, 4, 3)
        daily_rate       = st.slider("Daily Rate", 100, 1500, 800)
        hourly_rate      = st.slider("Hourly Rate", 30, 100, 65)
        monthly_rate     = st.slider("Monthly Rate", 2000, 27000, 14000)

# ─── Build input dict ─────────────────────────────────────────────────────────
user_inputs = {
    'Age': age, 'MonthlyIncome': monthly_income, 'YearsAtCompany': years_at_company,
    'JobSatisfaction': job_satisfaction, 'EnvironmentSatisfaction': env_satisfaction,
    'WorkLifeBalance': work_life_balance, 'OverTime': overtime,
    'Department':      locals().get('department',     'Research & Development'),
    'JobRole':         locals().get('job_role',        'Research Scientist'),
    'BusinessTravel':  locals().get('business_travel', 'Travel_Rarely'),
    'MaritalStatus':   locals().get('marital_status',  'Single'),
    'EducationField':  locals().get('education_field', 'Life Sciences'),
    'Gender':          locals().get('gender',          'Male'),
    'Education':       locals().get('education',       3),
    'JobLevel':        locals().get('job_level',       2),
    'DistanceFromHome':      locals().get('distance_home',          10),
    'TotalWorkingYears':     locals().get('total_working_years',     8),
    'YearsInCurrentRole':    locals().get('years_current_role',      3),
    'YearsSinceLastPromotion': locals().get('years_last_promotion',  2),
    'YearsWithCurrManager':  locals().get('years_with_manager',      3),
    'NumCompaniesWorked':    locals().get('num_companies',           2),
    'JobInvolvement':        locals().get('job_involvement',         3),
    'StockOptionLevel':      locals().get('stock_option',            1),
    'PercentSalaryHike':     locals().get('percent_hike',           15),
    'TrainingTimesLastYear': locals().get('training_times',          3),
    'RelationshipSatisfaction': locals().get('relationship_sat',     3),
    'PerformanceRating':     locals().get('performance_rating',      3),
    'DailyRate':             locals().get('daily_rate',            800),
    'HourlyRate':            locals().get('hourly_rate',            65),
    'MonthlyRate':           locals().get('monthly_rate',        14000),
}

pred_result, pred_prob = predict_attrition(user_inputs)

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:1.5rem 0 1rem;border-bottom:1px solid #2d3748;margin-bottom:1.5rem;">
    <div>
        <h1 style="margin:0;font-size:1.8rem;font-weight:800;color:#f1f5f9;">
            👥 People Analytics Dashboard
        </h1>
        <p style="margin:0.3rem 0 0;color:#64748b;font-size:0.9rem;">
            IBM HR Dataset &nbsp;·&nbsp; 1,470 Employees &nbsp;·&nbsp; Ensemble AI Model
        </p>
    </div>
    <div>
        <span style="background:#7c3aed22;color:#a78bfa;border:1px solid #7c3aed44;
              padding:4px 14px;border-radius:20px;font-size:0.8rem;font-weight:600;">
            CV ROC-AUC: {metrics['cv_roc_auc_mean']:.1%}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
attr_rate  = df['Attrition_bin'].mean()
attr_count = df['Attrition_bin'].sum()
avg_salary = df['MonthlyIncome'].mean()
avg_tenure = df['YearsAtCompany'].mean()

with k1:
    st.markdown(f'<div class="metric-card purple"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Employees</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="metric-card red"><div class="metric-value">{attr_count}</div><div class="metric-label">Attrition Count</div><div class="metric-delta delta-up">▲ {attr_rate:.1%} of workforce</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="metric-card blue"><div class="metric-value">${avg_salary:,.0f}</div><div class="metric-label">Avg Monthly Income</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="metric-card green"><div class="metric-value">{avg_tenure:.1f} yrs</div><div class="metric-label">Avg Tenure</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="metric-card amber"><div class="metric-value">{metrics["roc_auc"]:.0%}</div><div class="metric-label">Model ROC-AUC</div><div class="metric-delta delta-down">✔ F1: {metrics["f1"]:.2f}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction","📊 Analytics","🏆 Model Insights","📋 Data Explorer"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_pred, col_gauge, col_tips = st.columns([1,1,1], gap="large")

    with col_pred:
        st.markdown('<div class="section-header"><h3>🎯 Attrition Prediction</h3><span class="section-badge">LIVE</span></div>', unsafe_allow_html=True)
        risk_level = "high" if pred_prob >= 0.6 else ("medium" if pred_prob >= 0.35 else "low")
        verdict    = "Will Leave" if pred_result else "Will Stay"
        emoji      = "⚠️" if pred_result else "✅"
        st.markdown(f"""
        <div class="prediction-box pred-{risk_level}">
            <div class="pred-title">AI Prediction Result</div>
            <div class="pred-result">{emoji} {verdict}</div>
            <div class="pred-prob">Attrition Probability: {pred_prob:.1%}</div>
            <div style="margin-top:0.5rem;font-size:0.8rem;color:#64748b;">
                Risk Level: <b style="text-transform:uppercase;">{risk_level}</b>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>**🔍 Key Risk Signals**", unsafe_allow_html=True)
        risk_signals = []
        if user_inputs['OverTime'] == 'Yes':
            risk_signals.append(("Overtime Work",       "High Risk",          "#ef4444"))
        if user_inputs['JobSatisfaction'] <= 2:
            risk_signals.append(("Low Job Satisfaction", f"{user_inputs['JobSatisfaction']}/4", "#f59e0b"))
        if user_inputs['YearsAtCompany'] <= 2:
            risk_signals.append(("Short Tenure",         f"{user_inputs['YearsAtCompany']} yrs", "#f59e0b"))
        if user_inputs['MonthlyIncome'] < 3000:
            risk_signals.append(("Below Market Salary",  f"${user_inputs['MonthlyIncome']:,}", "#ef4444"))
        if user_inputs['StockOptionLevel'] == 0:
            risk_signals.append(("No Stock Options",     "Level 0",            "#f59e0b"))
        if user_inputs['MaritalStatus'] == 'Single':
            risk_signals.append(("Marital Status",       "Single",             "#94a3b8"))
        if not risk_signals:
            risk_signals.append(("Profile Assessment",   "Low Risk Profile",   "#10b981"))
        for name, val, color in risk_signals[:5]:
            st.markdown(f'<div class="risk-factor" style="border-left-color:{color}"><span class="risk-factor-name">{name}</span><span class="risk-factor-val" style="color:{color}">{val}</span></div>', unsafe_allow_html=True)

    with col_gauge:
        st.markdown('<div class="section-header"><h3>📈 Probability Gauge</h3></div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_prob * 100,
            number={'suffix':'%','font':{'size':36,'color':'#f1f5f9','family':'Inter'}},
            delta={'reference':16.1,'valueformat':'.1f',
                   'increasing':{'color':'#ef4444'},'decreasing':{'color':'#10b981'}},
            gauge={
                'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#374151','tickfont':{'color':'#64748b','size':11}},
                'bar':{'color':'#7c3aed','thickness':0.3},
                'bgcolor':'#1e2235','borderwidth':0,
                'steps':[{'range':[0,35],'color':'#0d2b1a'},{'range':[35,60],'color':'#2d1b06'},{'range':[60,100],'color':'#2d0909'}],
                'threshold':{'line':{'color':'#f1f5f9','width':3},'thickness':0.9,'value':pred_prob*100}
            },
            title={'text':"Attrition Risk",'font':{'color':'#94a3b8','size':14}}
        ))
        fig_gauge.update_layout(height=280,paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Inter'),margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Percentile bar
        sample_df = df.sample(min(150, len(df)), random_state=42)
        hist_probs = []
        for _, row in sample_df.iterrows():
            inp = {c: row.get(c,0) for c in NUMERICAL_FEATURES}
            for c in CATEGORICAL_FEATURES: inp[c] = str(row.get(c,''))
            _, p = predict_attrition(inp)
            hist_probs.append(p)
        percentile = np.mean(np.array(hist_probs) < pred_prob) * 100
        st.markdown(f"""
        <div style="background:#1e2235;border-radius:10px;padding:0.75rem 1rem;text-align:center;margin-top:0.5rem;">
            <div style="color:#94a3b8;font-size:0.75rem;">HIGHER RISK THAN</div>
            <div style="font-size:1.6rem;font-weight:700;color:#f1f5f9;">{percentile:.0f}%</div>
            <div style="color:#64748b;font-size:0.75rem;">of employees in dataset</div>
        </div>""", unsafe_allow_html=True)

    with col_tips:
        st.markdown('<div class="section-header"><h3>💡 HR Recommendations</h3></div>', unsafe_allow_html=True)
        tips = []
        if user_inputs['OverTime'] == 'Yes':
            tips.append(("🕐","Reduce Overtime","Employee is working overtime. Consider workload balancing or additional headcount to reduce burnout risk."))
        if user_inputs['JobSatisfaction'] <= 2:
            tips.append(("💬","Schedule 1:1 Check-in","Low job satisfaction detected. A manager conversation to understand pain points is recommended."))
        if user_inputs['MonthlyIncome'] < 4000:
            tips.append(("💰","Compensation Review","Income is below market average. Consider a salary review or bonus structure."))
        if user_inputs['YearsSinceLastPromotion'] >= 3:
            tips.append(("📈","Promotion / Career Path","No promotion in 3+ years. Discuss career development plans and growth opportunities."))
        if user_inputs['StockOptionLevel'] == 0:
            tips.append(("📊","Offer Equity Package","No stock options. Equity compensation significantly improves long-term retention."))
        if user_inputs['TrainingTimesLastYear'] <= 1:
            tips.append(("🎓","Learning & Development","Low training investment. Enroll in L&D programs to boost engagement."))
        if not tips:
            tips.append(("✅","Maintain Engagement","Employee profile shows healthy indicators. Continue regular check-ins and recognition."))
            tips.append(("🌟","Recognize Performance","High-performing employees benefit from recognition programs and peer acknowledgment."))
        for icon, title, body in tips[:4]:
            st.markdown(f'<div class="insight-card"><div class="insight-icon">{icon}</div><div class="insight-title">{title}</div><div class="insight-body">{body}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    r1c1, r1c2 = st.columns(2, gap="medium")

    with r1c1:
        st.markdown('<div class="section-header"><h3>🏢 Attrition by Department</h3></div>', unsafe_allow_html=True)
        dept = df.groupby('Department')['Attrition_bin'].agg(['mean','sum','count']).reset_index()
        dept.columns = ['Department','Rate','Left','Total']
        fig = go.Figure(go.Bar(
            x=dept['Department'], y=dept['Rate']*100,
            marker=dict(color=['#7c3aed','#3b82f6','#10b981'], line=dict(width=0)),
            text=[f"{r:.1f}%" for r in dept['Rate']*100], textposition='outside',
            textfont=dict(color='#f1f5f9', size=12),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=300,
            yaxis=dict(title="Attrition Rate (%)", gridcolor='#2d3748', zeroline=False),
            xaxis=dict(gridcolor='#2d3748'), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.markdown('<div class="section-header"><h3>💼 Attrition by Job Role</h3></div>', unsafe_allow_html=True)
        role = df.groupby('JobRole')['Attrition_bin'].mean().reset_index().sort_values('Attrition_bin')
        fig = go.Figure(go.Bar(
            x=role['Attrition_bin']*100, y=role['JobRole'], orientation='h',
            marker=dict(color=role['Attrition_bin']*100,
                        colorscale=[[0,'#1e3a5f'],[0.5,'#7c3aed'],[1,'#ef4444']],
                        line=dict(width=0)),
            text=[f"{v:.1f}%" for v in role['Attrition_bin']*100], textposition='outside',
            textfont=dict(color='#f1f5f9', size=10),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=300,
            xaxis=dict(title="Attrition Rate (%)", gridcolor='#2d3748', zeroline=False),
            yaxis=dict(gridcolor='#2d3748', tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2, r2c3 = st.columns(3, gap="medium")

    with r2c1:
        st.markdown('<div class="section-header"><h3>🎂 Attrition by Age Group</h3></div>', unsafe_allow_html=True)
        df['AgeGroup'] = pd.cut(df['Age'], bins=[17,25,30,35,40,50,65],
                                 labels=['18-25','26-30','31-35','36-40','41-50','51+'])
        age_attr = df.groupby('AgeGroup', observed=True)['Attrition_bin'].mean().reset_index()
        fig = go.Figure(go.Scatter(
            x=age_attr['AgeGroup'].astype(str), y=age_attr['Attrition_bin']*100,
            mode='lines+markers', line=dict(color='#7c3aed', width=3),
            marker=dict(size=10, color='#a855f7', line=dict(width=2, color='#f1f5f9')),
            fill='tozeroy', fillcolor='rgba(124,58,237,0.1)',
        ))
        fig.update_layout(**PLOT_LAYOUT, height=260,
            yaxis=dict(title="%", gridcolor='#2d3748', zeroline=False),
            xaxis=dict(gridcolor='#2d3748'))
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.markdown('<div class="section-header"><h3>✈️ Business Travel Impact</h3></div>', unsafe_allow_html=True)
        travel = df.groupby('BusinessTravel')['Attrition_bin'].mean().reset_index()
        fig = go.Figure(go.Pie(
            labels=travel['BusinessTravel'], values=travel['Attrition_bin']*100, hole=0.55,
            marker=dict(colors=['#7c3aed','#3b82f6','#10b981'], line=dict(color='#0f1117', width=2)),
            textinfo='label+percent', textfont=dict(color='#f1f5f9', size=11),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=260, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with r2c3:
        st.markdown('<div class="section-header"><h3>💍 Marital Status</h3></div>', unsafe_allow_html=True)
        marital = df.groupby('MaritalStatus')['Attrition_bin'].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=marital['MaritalStatus'], y=marital['Attrition_bin']*100,
            marker=dict(color=['#7c3aed','#3b82f6','#10b981'], line=dict(width=0)),
            text=[f"{v:.1f}%" for v in marital['Attrition_bin']*100], textposition='outside',
            textfont=dict(color='#f1f5f9'),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=260,
            yaxis=dict(title="%", gridcolor='#2d3748', zeroline=False),
            xaxis=dict(gridcolor='#2d3748'), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header"><h3>💵 Income Distribution: Stayed vs Left</h3></div>', unsafe_allow_html=True)
    fig = go.Figure()
    for label, color, name in [(0,'#3b82f6','Stayed'),(1,'#ef4444','Left')]:
        fig.add_trace(go.Histogram(
            x=df[df['Attrition_bin']==label]['MonthlyIncome'], name=name, nbinsx=40,
            marker=dict(color=color, opacity=0.75, line=dict(width=0)),
        ))
    fig.update_layout(**PLOT_LAYOUT, height=280, barmode='overlay',
        xaxis=dict(title="Monthly Income ($)", gridcolor='#2d3748', zeroline=False),
        yaxis=dict(title="Count", gridcolor='#2d3748', zeroline=False),
        legend=dict(bgcolor='#1e2235', bordercolor='#2d3748', borderwidth=1))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    m1, m2 = st.columns(2, gap="medium")

    with m1:
        st.markdown('<div class="section-header"><h3>🏆 Top Feature Importances</h3></div>', unsafe_allow_html=True)
        feats = pd.DataFrame(metrics['top_features'], columns=['Feature','Importance']).sort_values('Importance')
        fig = go.Figure(go.Bar(
            x=feats['Importance'], y=feats['Feature'], orientation='h',
            marker=dict(color=feats['Importance'],
                        colorscale=[[0,'#1e3a5f'],[0.5,'#7c3aed'],[1,'#a855f7']],
                        line=dict(width=0)),
            text=[f"{v:.3f}" for v in feats['Importance']], textposition='outside',
            textfont=dict(color='#f1f5f9', size=10),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=420,
            xaxis=dict(title="Importance Score", gridcolor='#2d3748', zeroline=False),
            yaxis=dict(gridcolor='#2d3748', tickfont=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    with m2:
        st.markdown('<div class="section-header"><h3>📊 Model Performance</h3></div>', unsafe_allow_html=True)
        metric_names = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
        metric_vals  = [metrics['accuracy'], metrics['precision'], metrics['recall'],
                        metrics['f1'], metrics['roc_auc']]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=metric_vals + [metric_vals[0]], theta=metric_names + [metric_names[0]],
            fill='toself', fillcolor='rgba(124,58,237,0.2)',
            line=dict(color='#7c3aed', width=2),
            marker=dict(size=8, color='#a855f7'),
        ))
        fig.update_layout(
            polar=dict(bgcolor='#1e2235',
                radialaxis=dict(visible=True, range=[0,1], gridcolor='#2d3748',
                                tickfont=dict(color='#64748b', size=9), linecolor='#2d3748'),
                angularaxis=dict(gridcolor='#2d3748', tickfont=dict(color='#94a3b8', size=11))),
            paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter'),
            showlegend=False, height=300, margin=dict(l=40,r=40,t=40,b=40))
        st.plotly_chart(fig, use_container_width=True)

        cm = np.array(metrics['confusion_matrix'])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=['Predicted: Stay','Predicted: Leave'],
            y=['Actual: Stay','Actual: Leave'],
            colorscale=[[0,'#1e2235'],[1,'#7c3aed']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(size=20, color='white'), showscale=False,
        ))
        fig_cm.update_layout(**PLOT_LAYOUT, height=200,
            title=dict(text='Confusion Matrix', font=dict(color='#94a3b8', size=13)),
            margin=dict(l=10,r=10,t=45,b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown('<div class="section-header"><h3>🤖 Model Architecture</h3></div>', unsafe_allow_html=True)
    arch_cols = st.columns(4)
    arch_info = [
        ("XGBoost",       "Weight: 3×", "300 trees, depth 5, LR 0.05", "#7c3aed"),
        ("Gradient Boost","Weight: 2×", "200 trees, depth 4, LR 0.05", "#3b82f6"),
        ("Random Forest", "Weight: 2×", "300 trees, balanced weights",  "#10b981"),
        ("Logistic Reg.", "Weight: 1×", "L2 regularized, balanced",     "#f59e0b"),
    ]
    for col, (name, weight, detail, color) in zip(arch_cols, arch_info):
        with col:
            st.markdown(f"""
            <div style="background:#1e2235;border:1px solid #2d3748;border-radius:12px;
                 padding:1.25rem;border-top:3px solid {color};text-align:center;">
                <div style="font-size:1rem;font-weight:700;color:#f1f5f9;">{name}</div>
                <div style="font-size:0.8rem;color:{color};margin:0.25rem 0;">{weight}</div>
                <div style="font-size:0.75rem;color:#64748b;">{detail}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1a1d2e;border:1px solid #2d3748;border-radius:12px;
         padding:1rem 1.5rem;margin-top:1rem;">
        <span style="color:#7c3aed;font-weight:600;">⚡ Training Pipeline: </span>
        <span style="color:#94a3b8;font-size:0.85rem;">
        Raw CSV → Drop constants → Label Encoding → Feature Engineering (5 new features)
        → SMOTE oversampling (1:1 balance) → StandardScaler → Soft Voting Ensemble
        → Threshold tuning (0.35) → 5-Fold CV Evaluation
        </span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header"><h3>🔍 Interactive Data Explorer</h3></div>', unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        x_axis = st.selectbox("X Axis", NUMERICAL_FEATURES, index=NUMERICAL_FEATURES.index('MonthlyIncome'))
    with ec2:
        y_axis = st.selectbox("Y Axis", NUMERICAL_FEATURES, index=NUMERICAL_FEATURES.index('Age'))
    with ec3:
        color_by = st.selectbox("Color by", ['Attrition','Department','JobRole','MaritalStatus'])

    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
        color_discrete_map={'Yes':'#ef4444','No':'#3b82f6'},
        opacity=0.7, height=420, template='plotly_dark',
        hover_data=['Age','Department','JobRole','MonthlyIncome','YearsAtCompany'])
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(**PLOT_LAYOUT, height=420,
        legend=dict(bgcolor='#1e2235', bordercolor='#2d3748', borderwidth=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header"><h3>🌡️ Feature Correlation Heatmap</h3></div>', unsafe_allow_html=True)
    corr_cols = ['Age','MonthlyIncome','YearsAtCompany','TotalWorkingYears',
                 'JobSatisfaction','EnvironmentSatisfaction','WorkLifeBalance',
                 'JobLevel','YearsInCurrentRole','Attrition_bin']
    corr = df[corr_cols].corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale=[[0,'#ef4444'],[0.5,'#1e2235'],[1,'#7c3aed']], zmid=0,
        text=np.round(corr.values,2), texttemplate='%{text}',
        textfont=dict(size=9), showscale=True,
    ))
    fig_corr.update_layout(**PLOT_LAYOUT, height=420,
        margin=dict(l=10,r=10,t=10,b=10),
        xaxis=dict(tickfont=dict(size=9), tickangle=-30),
        yaxis=dict(tickfont=dict(size=9)))
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("📄 View Raw Data", expanded=False):
        cols_show = ['Age','Department','JobRole','MaritalStatus','MonthlyIncome',
                     'YearsAtCompany','JobSatisfaction','OverTime','Attrition']
        st.dataframe(df[cols_show], use_container_width=True, height=400)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin:2rem 0 1rem;border-color:#2d3748;">
<div style="text-align:center;color:#374151;font-size:0.78rem;">
    👥 People Analytics Dashboard &nbsp;·&nbsp;
    Built with Streamlit, XGBoost, Scikit-learn &nbsp;·&nbsp;
    IBM HR Analytics Dataset &nbsp;·&nbsp; Ensemble Model · CV ROC-AUC 82%
</div>
""", unsafe_allow_html=True)
