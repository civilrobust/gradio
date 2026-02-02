"""
Cardiac Risk & Regression Lab (Synthetic Demo)
- Generates synthetic cardiac-style data (NO real patient data)
- Trains:
  1) Logistic Regression -> probability of 30-day event (classification)
  2) Ridge Regression -> NT-proBNP prediction (regression)
- Visuals: ROC curve, confusion matrix, regression fit, residuals

Run:
  python app.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Synthetic cardiac dataset
# -----------------------------

SEX = ["female", "male"]
CHEST_PAIN = ["none", "atypical", "typical"]
ECG_ST = ["normal", "depressed", "elevated"]
SMOKING = ["no", "yes"]
DIABETES = ["no", "yes"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def make_synthetic_cardiac_df(n: int, seed: int) -> pd.DataFrame:
    """
    Creates synthetic rows representing typical cardiac risk factors and findings.

    Outputs:
      - event_30d: 0/1 (cardiac event within 30 days)
      - ntprobnp: continuous value (approx, synthetic)
    """
    rng = random.Random(seed)
    rows = []

    for _ in range(n):
        age = clamp(rng.gauss(62, 12), 18, 95)
        sex = rng.choice(SEX)

        systolic_bp = clamp(rng.gauss(138, 20), 90, 220)  # mmHg
        heart_rate = clamp(rng.gauss(84, 18), 40, 180)    # beats per minute

        # Labs
        troponin = clamp(abs(rng.gauss(18, 25)), 0, 400)  # ng/L (synthetic scale)
        ldl = clamp(rng.gauss(3.2, 0.9), 0.8, 7.5)        # mmol/L
        egfr = clamp(rng.gauss(78, 22), 10, 130)          # mL/min/1.73m^2
        crp = clamp(abs(rng.gauss(8, 18)), 0, 250)        # mg/L

        chest_pain = rng.choices(CHEST_PAIN, weights=[0.45, 0.30, 0.25])[0]
        ecg_st = rng.choices(ECG_ST, weights=[0.70, 0.20, 0.10])[0]

        diabetes = rng.choices(DIABETES, weights=[0.72, 0.28])[0]
        smoking = rng.choices(SMOKING, weights=[0.75, 0.25])[0]

        # ---------- event probability (classification target) ----------
        # Sensible directionality:
        # - higher age, troponin, ST changes, typical chest pain, diabetes, smoking, low eGFR -> higher risk
        score = 0.0
        score += 0.03 * (age - 55)
        score += 0.015 * (systolic_bp - 130)
        score += 0.012 * (heart_rate - 80)
        score += 0.010 * (troponin - 10)
        score += 0.25 * (ldl - 3.0)
        score += 0.012 * (crp - 5)
        score += 0.020 * (75 - egfr)  # lower eGFR increases score

        if chest_pain == "typical":
            score += 1.00
        elif chest_pain == "atypical":
            score += 0.35

        if ecg_st == "depressed":
            score += 0.60
        elif ecg_st == "elevated":
            score += 1.20

        if diabetes == "yes":
            score += 0.55
        if smoking == "yes":
            score += 0.35

        # Small sex-related shift (purely synthetic)
        if sex == "male":
            score += 0.10

        p_event = sigmoid(score - 2.2)  # shift controls overall event rate
        event_30d = 1 if rng.random() < p_event else 0

        # ---------- NT-proBNP (regression target) ----------
        # Synthetic formula (not clinical truth). Plausible associations:
        # - older age, lower eGFR, higher HR, higher troponin, ST changes -> higher BNP
        nt_score = 5.7  # log-scale baseline
        nt_score += 0.012 * (age - 55)
        nt_score += 0.010 * (heart_rate - 80)
        nt_score += 0.0025 * (troponin)
        nt_score += 0.010 * (75 - egfr)
        if ecg_st != "normal":
            nt_score += 0.20
        if diabetes == "yes":
            nt_score += 0.12

        nt_score += rng.gauss(0, 0.25)
        ntprobnp = float(math.exp(nt_score))  # log-normal-ish

        rows.append(
            dict(
                age=round(age, 1),
                sex=sex,
                systolic_bp=round(systolic_bp, 1),
                heart_rate=round(heart_rate, 1),
                troponin=round(troponin, 1),
                ldl=round(ldl, 2),
                egfr=round(egfr, 1),
                crp=round(crp, 1),
                chest_pain=chest_pain,
                ecg_st=ecg_st,
                diabetes=diabetes,
                smoking=smoking,
                event_30d=event_30d,
                ntprobnp=round(ntprobnp, 1),
            )
        )

    return pd.DataFrame(rows)


CAT_COLS = ["sex", "chest_pain", "ecg_st", "diabetes", "smoking"]
NUM_COLS = ["age", "systolic_bp", "heart_rate", "troponin", "ldl", "egfr", "crp"]


@dataclass
class TrainedState:
    clf: Pipeline
    reg: Pipeline
    threshold: float
    feature_cols: List[str]


STATE: Dict[str, TrainedState] = {}


def plot_confusion(cm: np.ndarray, labels: list) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return fig


def plot_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[plt.Figure, plt.Figure]:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(y_true, y_pred)
    ax1.set_title("Regression: predicted vs actual NT-proBNP")
    ax1.set_xlabel("Actual NT-proBNP")
    ax1.set_ylabel("Predicted NT-proBNP")

    residuals = y_true - y_pred
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.scatter(y_pred, residuals)
    ax2.axhline(0, linestyle="--")
    ax2.set_title("Residuals (actual - predicted)")
    ax2.set_xlabel("Predicted NT-proBNP")
    ax2.set_ylabel("Residual")

    return fig1, fig2


def train_models(n: int, seed: int, test_size: float, threshold: float):
    df = make_synthetic_cardiac_df(n=n, seed=seed)

    X = df[CAT_COLS + NUM_COLS]
    y_event = df["event_30d"].astype(int)
    y_bnp = df["ntprobnp"].astype(float)

    X_train, X_test, y_event_train, y_event_test, y_bnp_train, y_bnp_test = train_test_split(
        X, y_event, y_bnp, test_size=test_size, random_state=seed, stratify=y_event
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    clf = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )

    reg = Pipeline(
        steps=[
            ("pre", pre),
            ("reg", Ridge(alpha=1.5)),
        ]
    )

    clf.fit(X_train, y_event_train)
    reg.fit(X_train, y_bnp_train)

    # --- classification eval ---
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_event_test, y_pred)
    cm = confusion_matrix(y_event_test, y_pred, labels=[0, 1])
    cm_fig = plot_confusion(cm, labels=["no_event", "event"])
    roc_fig = plot_roc(y_event_test.values, y_prob)

    # --- regression eval ---
    y_bnp_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_bnp_test, y_bnp_pred)
    rmse = math.sqrt(mean_squared_error(y_bnp_test, y_bnp_pred))
    r2 = r2_score(y_bnp_test, y_bnp_pred)
    reg_fig1, reg_fig2 = plot_regression(y_bnp_test.values, y_bnp_pred)

    summary = (
        f"Trained on {len(X_train)} rows, tested on {len(X_test)} rows.\n\n"
        f"Classification (30-day event)\n"
        f"- Decision threshold: {threshold:.2f}\n"
        f"- Accuracy: {acc:.3f}\n\n"
        f"Regression (NT-proBNP)\n"
        f"- Mean Absolute Error: {mae:.1f}\n"
        f"- Root Mean Squared Error: {rmse:.1f}\n"
        f"- R-squared: {r2:.3f}\n\n"
        f"Note: All data is synthetic and for learning/demonstration only."
    )

    STATE["trained"] = TrainedState(
        clf=clf, reg=reg, threshold=threshold, feature_cols=CAT_COLS + NUM_COLS
    )

    return df.head(25), summary, cm_fig, roc_fig, reg_fig1, reg_fig2


def predict_patient(
    age: float,
    sex: str,
    systolic_bp: float,
    heart_rate: float,
    troponin: float,
    ldl: float,
    egfr: float,
    crp: float,
    chest_pain: str,
    ecg_st: str,
    diabetes: str,
    smoking: str,
):
    if "trained" not in STATE:
        return "No model trained", "Train the models first.", "—"

    st = STATE["trained"]

    row = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "systolic_bp": systolic_bp,
            "heart_rate": heart_rate,
            "troponin": troponin,
            "ldl": ldl,
            "egfr": egfr,
            "crp": crp,
            "chest_pain": chest_pain,
            "ecg_st": ecg_st,
            "diabetes": diabetes,
            "smoking": smoking,
        }]
    )

    prob = float(st.clf.predict_proba(row)[0, 1])
    event_flag = "event likely" if prob >= st.threshold else "event less likely"
    bnp_pred = float(st.reg.predict(row)[0])

    drivers = []
    if troponin >= 40:
        drivers.append("raised troponin")
    if egfr <= 45:
        drivers.append("reduced kidney function (eGFR)")
    if heart_rate >= 110:
        drivers.append("high heart rate")
    if systolic_bp >= 170:
        drivers.append("very high systolic blood pressure")
    if crp >= 50:
        drivers.append("high inflammation marker (CRP)")
    if chest_pain == "typical":
        drivers.append("typical chest pain pattern")
    if ecg_st in ["depressed", "elevated"]:
        drivers.append(f"ECG ST segment change ({ecg_st})")
    if diabetes == "yes":
        drivers.append("diabetes risk factor")
    if smoking == "yes":
        drivers.append("smoking risk factor")

    if not drivers:
        drivers_txt = "No strong red-flag thresholds triggered; risk is driven by combined smaller factors."
    else:
        drivers_txt = "Key drivers: " + ", ".join(drivers) + "."

    risk_txt = f"{prob*100:.1f}% ({event_flag}, threshold {st.threshold:.2f})"
    bnp_txt = f"{bnp_pred:.0f} (synthetic units similar to pg/mL scale)"

    explanation = (
        f"**30-day event risk:** {risk_txt}\n\n"
        f"**Predicted NT-proBNP:** {bnp_txt}\n\n"
        f"{drivers_txt}\n\n"
        f"Reminder: synthetic data for learning/demo."
    )

    return f"{prob*100:.1f}%", f"{bnp_pred:.0f}", explanation


# -----------------------------
# Gradio UI
# -----------------------------

with gr.Blocks(title="Cardiac Risk & Regression Lab (Synthetic)") as demo:
    gr.Markdown(
        """
# Cardiac Risk & Regression Lab (synthetic demo)

A **clinical-style** Gradio app that teaches regression while looking impressive.

**Two models:**
- **Logistic Regression** → probability of a **30-day cardiac event**
- **Linear Regression (Ridge)** → predicted **NT-proBNP**

All data is synthetic for learning and demonstration.
"""
    )

    with gr.Tabs():
        with gr.TabItem("1) Train & Evaluate"):
            with gr.Row():
                n = gr.Slider(500, 20000, value=8000, step=500, label="Rows of synthetic data")
                seed = gr.Number(value=42, precision=0, label="Random seed")
                test_size = gr.Slider(0.1, 0.5, value=0.25, step=0.05, label="Test set fraction")
                threshold = gr.Slider(0.1, 0.9, value=0.50, step=0.05, label="Event decision threshold")

            train_btn = gr.Button("Train models", variant="primary")

            df_preview = gr.Dataframe(label="Preview (first 25 rows)", interactive=False)
            summary = gr.Textbox(label="Training summary", lines=10)

            with gr.Row():
                cm_plot = gr.Plot(label="Confusion matrix (classification)")
                roc_plot = gr.Plot(label="ROC curve (classification)")

            with gr.Row():
                reg_plot1 = gr.Plot(label="Predicted vs actual (regression)")
                reg_plot2 = gr.Plot(label="Residuals plot (regression)")

            train_btn.click(
                fn=train_models,
                inputs=[n, seed, test_size, threshold],
                outputs=[df_preview, summary, cm_plot, roc_plot, reg_plot1, reg_plot2],
            )

        with gr.TabItem("2) Predict (patient-style)"):
            gr.Markdown("Enter synthetic patient features to get event risk and NT-proBNP prediction.")

            with gr.Row():
                age_in = gr.Slider(18, 95, value=62, step=1, label="Age (years)")
                sex_in = gr.Dropdown(SEX, value="male", label="Sex")

            with gr.Row():
                systolic_bp_in = gr.Slider(90, 220, value=140, step=1, label="Systolic BP (mmHg)")
                heart_rate_in = gr.Slider(40, 180, value=85, step=1, label="Heart rate (bpm)")

            with gr.Row():
                troponin_in = gr.Slider(0, 400, value=18, step=1, label="Troponin (synthetic ng/L)")
                ldl_in = gr.Slider(0.8, 7.5, value=3.2, step=0.1, label="LDL (mmol/L)")

            with gr.Row():
                egfr_in = gr.Slider(10, 130, value=78, step=1, label="eGFR (mL/min/1.73m²)")
                crp_in = gr.Slider(0, 250, value=8, step=1, label="CRP (mg/L)")

            with gr.Row():
                chest_pain_in = gr.Dropdown(CHEST_PAIN, value="none", label="Chest pain pattern")
                ecg_st_in = gr.Dropdown(ECG_ST, value="normal", label="ECG ST segment")
                diabetes_in = gr.Dropdown(DIABETES, value="no", label="Diabetes")
                smoking_in = gr.Dropdown(SMOKING, value="no", label="Smoking")

            predict_btn = gr.Button("Predict", variant="primary")

            with gr.Row():
                risk_out = gr.Textbox(label="30-day event probability")
                bnp_out = gr.Textbox(label="Predicted NT-proBNP")

            explanation_out = gr.Markdown()

            predict_btn.click(
                fn=predict_patient,
                inputs=[
                    age_in,
                    sex_in,
                    systolic_bp_in,
                    heart_rate_in,
                    troponin_in,
                    ldl_in,
                    egfr_in,
                    crp_in,
                    chest_pain_in,
                    ecg_st_in,
                    diabetes_in,
                    smoking_in,
                ],
                outputs=[risk_out, bnp_out, explanation_out],
            )

    gr.Markdown(
        """
---  
### Notes (for demos)
- **Logistic regression** outputs a probability between 0 and 1.
- **Decision threshold** converts that probability into “event / no event”.
- **Ridge regression** is linear regression with a penalty that reduces overfitting.
"""
    )


if __name__ == "__main__":
    # If you’re running on a server / VM, set server_name="0.0.0.0"
    demo.launch()
