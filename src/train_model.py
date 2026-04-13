"""
People Analytics - Model Training Pipeline
Trains an ensemble model for employee attrition prediction.
Run from the project root:  python src/train_model.py
Or from inside src/:        python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                              precision_score, recall_score, f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── Paths (works from any working directory) ─────────────────────────────────
_THIS = os.path.abspath(__file__)
_SRC  = os.path.dirname(_THIS)
BASE_DIR = os.path.dirname(_SRC) if os.path.basename(_SRC) == "src" else _SRC
DATA_PATH = os.path.join(BASE_DIR, "data", "raw.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Feature Configuration ────────────────────────────────────────────────────
CATEGORICAL_FEATURES = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]

NUMERICAL_FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education',
    'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

DROP_COLS = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
TARGET    = 'Attrition'


def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    df = df.drop(columns=DROP_COLS, errors='ignore')
    df[TARGET] = (df[TARGET] == 'Yes').astype(int)

    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Feature engineering
    df['IncomePerYear']     = df['MonthlyIncome'] * 12
    df['TenureRatio']       = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    df['SatisfactionScore'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] +
                                df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4
    df['CareerGrowthScore'] = df['JobLevel'] / (df['TotalWorkingYears'] + 1)
    df['PromotionLag']      = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)

    eng_features = ['IncomePerYear', 'TenureRatio', 'SatisfactionScore',
                    'CareerGrowthScore', 'PromotionLag']
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + eng_features

    X = df[feature_cols]
    y = df[TARGET]
    return X, y, label_encoders, feature_cols


def build_model():
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=5,
        eval_metric='logloss', random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    lr = LogisticRegression(
        C=0.5, class_weight='balanced', max_iter=1000,
        random_state=42, solver='liblinear'
    )
    return VotingClassifier(
        estimators=[('xgb', xgb), ('gb', gb), ('rf', rf), ('lr', lr)],
        voting='soft', weights=[3, 2, 2, 1]
    )


def train(verbose=True):
    def log(msg):
        if verbose: print(msg)

    log("Loading and preprocessing data...")
    X, y, label_encoders, feature_cols = load_and_preprocess(DATA_PATH)
    log(f"  Dataset: {X.shape[0]} rows, {X.shape[1]} features | Attrition rate: {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    log(f"  After SMOTE: {X_res.shape[0]} samples, balance: {y_res.mean():.1%}")

    scaler = StandardScaler()
    X_res_sc = scaler.fit_transform(X_res)
    X_test_sc = scaler.transform(X_test)

    log("Training ensemble model (XGB + GradBoost + RF + LogReg)...")
    model = build_model()
    model.fit(X_res_sc, y_res)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "attrition_rate": round(float(y.mean()), 4),
        "n_samples":  int(len(y)),
        "n_features": int(X.shape[1]),
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_all_sc = scaler.transform(X)
    cv_scores = cross_val_score(model, X_all_sc, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    metrics['cv_roc_auc_mean'] = round(float(cv_scores.mean()), 4)
    metrics['cv_roc_auc_std']  = round(float(cv_scores.std()), 4)

    log(f"\nModel Performance:")
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            log(f"  {k}: {v}")

    # Feature importances (from XGB sub-model)
    xgb_model = model.estimators_[0]
    importances = dict(zip(feature_cols, xgb_model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    metrics['top_features'] = [(k, float(v)) for k, v in top_features]

    # Save model artifacts
    joblib.dump(model,          os.path.join(MODEL_DIR, "attrition_model.pkl"))
    joblib.dump(scaler,         os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(feature_cols,   os.path.join(MODEL_DIR, "feature_cols.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save dashboard stats
    df_full = pd.read_csv(DATA_PATH)
    df_full['Attrition_bin'] = (df_full['Attrition'] == 'Yes').astype(int)

    (df_full.groupby('Department')['Attrition_bin']
     .agg(['mean','count']).reset_index()
     .rename(columns={'mean':'AttritionRate','count':'Count'})
     .to_csv(os.path.join(MODEL_DIR, "dept_attrition.csv"), index=False))

    (df_full.groupby('JobRole')['Attrition_bin']
     .agg(['mean','count']).reset_index()
     .rename(columns={'mean':'AttritionRate','count':'Count'})
     .to_csv(os.path.join(MODEL_DIR, "role_attrition.csv"), index=False))

    log("\nAll artifacts saved to models/")
    return metrics


if __name__ == "__main__":
    train()
