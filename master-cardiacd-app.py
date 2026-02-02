# app.py
# Clinical ML Tutor — Cardiac (Synthetic Demo)
# - Generates realistic synthetic datasets (no patient-identifiable data)
# - Loads CSV, previews data, shows summary stats (incl. class balance + missingness)
# - Trains an explainable baseline model (Logistic Regression) + ROC curve + coefficients
# - LLM assistant can explain the CURRENT results using auto-built context (exec briefing / risks / next improvements)
#
# Tested conceptually for Gradio 6.x messages-format Chatbot.
# NOTE: Some Gradio component kwargs differ across versions; this file avoids fragile params like Dataframe(height=...).

from __future__ import annotations

import os
import io
import time
import json
import math
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr

# OpenAI (your working approach)
from openai import OpenAI

# Optional ML imports (graceful if missing)
SKLEARN_OK = True
SKLEARN_ERR = ""
try:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
except Exception as e:
    SKLEARN_OK = False
    SKLEARN_ERR = str(e)

# Matplotlib for ROC plot
MPL_OK = True
MPL_ERR = ""
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    MPL_OK = False
    MPL_ERR = str(e)


# -----------------------------
# Visual theme (NHS Blue)
# -----------------------------
NHS_BLUE = "#005EB8"
NHS_DARK = "#003087"
NHS_LIGHT_BG = "#E8F1FB"
NHS_CARD_BG = "#FFFFFF"
NHS_TEXT = "#0B0F14"
NHS_MUTED = "#52606D"
NHS_BORDER = "#D5E2F2"

CUSTOM_CSS = f"""
/* Page background */
.gradio-container {{
  background: radial-gradient(1200px 600px at 15% 0%, #F7FBFF 0%, {NHS_LIGHT_BG} 45%, #F3F8FF 100%);
  color: {NHS_TEXT};
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}}

/* Header pill */
.kch-hero {{
  background: linear-gradient(90deg, #DCEBFF 0%, #EAF4FF 55%, #DCEBFF 100%);
  border: 1px solid {NHS_BORDER};
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 25px rgba(0, 48, 135, 0.10);
}}
.kch-title {{
  font-size: 28px;
  font-weight: 800;
  margin: 0;
  color: {NHS_DARK};
}}
.kch-sub {{
  margin-top: 6px;
  color: {NHS_MUTED};
  font-size: 14px;
}}
.kch-badges {{
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 10px;
}}
.kch-badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid {NHS_BORDER};
  background: #FFFFFF;
  font-size: 12px;
  color: {NHS_TEXT};
  box-shadow: 0 6px 16px rgba(0, 48, 135, 0.08);
}}
.kch-dot {{
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: {NHS_BLUE};
  display: inline-block;
}}

/* Cards */
.kch-card {{
  background: {NHS_CARD_BG};
  border: 1px solid {NHS_BORDER};
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 10px 22px rgba(0, 48, 135, 0.08);
}}
.kch-card h3 {{
  margin: 0 0 8px 0;
  font-size: 16px;
  color: {NHS_DARK};
}}
.kch-card p {{
  margin: 0;
  color: {NHS_MUTED};
  font-size: 12px;
}}

/* Buttons */
button.primary {{
  background: {NHS_BLUE} !important;
  border: 1px solid {NHS_BLUE} !important;
  color: white !important;
}}
button.primary:hover {{
  background: {NHS_DARK} !important;
  border-color: {NHS_DARK} !important;
}}
button.secondary {{
  background: white !important;
  border: 1px solid {NHS_BORDER} !important;
  color: {NHS_TEXT} !important;
}}
button.secondary:hover {{
  border-color: {NHS_BLUE} !important;
  color: {NHS_DARK} !important;
}}

/* Make Gradio sections a bit calmer */
.block {{
  border-radius: 14px !important;
}}

/* Chatbot area */
.kch-chat-note {{
  font-size: 12px;
  color: {NHS_MUTED};
  margin-top: 6px;
}}
"""


# -----------------------------
# LLM config
# -----------------------------
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are the 'AI Analyst Commentary' panel for a clinical ML demo.\n"
    "Purpose lock:\n"
    "- ONLY explain the current synthetic dataset + model outputs shown in the UI.\n"
    "- Do NOT give clinical advice or patient-specific guidance.\n"
    "- Always assume the data is synthetic and for learning/demonstration.\n"
    "- If you use a technical term, define it immediately in plain English.\n"
    "Style:\n"
    "- Executive-friendly: short, structured, practical.\n"
    "- Use headings and bullets. Avoid jargon.\n"
    "- If performance looks 'too perfect' (e.g., AUC > 0.95), flag likely leakage/over-clean synthetic rules.\n"
)


def _extract_output_text(resp) -> str:
    # Newer SDK convenience field
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text.strip()

    # Fallback: walk output message blocks
    out = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    out.append(t)
    return ("\n".join(out)).strip() or "No text returned from the model."


def ask_llm(user_message: str, auto_context: str, history_messages: List[Dict[str, str]]) -> Tuple[str, float]:
    """
    history_messages is MESSAGES FORMAT:
    [{"role": "user"/"assistant", "content": "..."}]
    """
    if not user_message or not user_message.strip():
        return "Type a question (e.g., 'Explain why AUC is high').", 0.0

    t0 = time.time()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if auto_context and auto_context.strip():
        messages.append({"role": "user", "content": "CURRENT UI SNAPSHOT (auto context):\n" + auto_context.strip()})

    for m in history_messages or []:
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": str(m["content"])})

    messages.append({"role": "user", "content": user_message.strip()})

    resp = client.responses.create(
        model=MODEL_NAME,
        input=messages,
    )
    text = _extract_output_text(resp)
    ms = (time.time() - t0) * 1000.0
    return text, ms


# -----------------------------
# Synthetic dataset generation (realistic-ish, non-perfect)
# -----------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class SynthConfig:
    n_rows: int = 250
    base_event_rate: float = 0.22
    noise: float = 0.55
    label_noise: float = 0.04
    missingness: float = 0.10
    seed: int = 42


def generate_synthetic_cardiac(cfg: SynthConfig) -> pd.DataFrame:
    """
    Create a correlated, noisy synthetic dataset with overlap (so the model isn't perfect).
    Columns (typical):
      age, sex, systolic_bp, heart_rate, troponin, ldl, egfr, crp, chest_pain, ecg_st,
      diabetes, smoking, ntprobnp, event_30d
    """
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.n_rows)

    # Age distribution: adult population
    age = np.clip(rng.normal(58, 14, n), 18, 95)

    # Sex: 0/1 (simple)
    sex = rng.binomial(1, 0.52, n)

    # Diabetes: more likely with age + sex slight effect
    p_diab = _sigmoid((age - 55) / 12 + (sex - 0.5) * 0.15 - 0.2)
    diabetes = rng.binomial(1, np.clip(p_diab, 0.05, 0.75))

    # Smoking: depends on age/sex/diabetes (messy)
    p_smoke = _sigmoid(-0.2 + (sex - 0.5) * 0.25 - (age - 50) / 40 - diabetes * 0.15)
    smoking = rng.binomial(1, np.clip(p_smoke, 0.05, 0.60))

    # Systolic BP: age + diabetes + noise
    systolic_bp = np.clip(
        rng.normal(125 + (age - 55) * 0.55 + diabetes * 8, 12 + cfg.noise * 6, n),
        85,
        220,
    )

    # Heart rate: mild effects + noise
    heart_rate = np.clip(
        rng.normal(74 + smoking * 3 + diabetes * 2 + (age - 55) * 0.08, 10 + cfg.noise * 5, n),
        40,
        160,
    )

    # eGFR: declines with age, diabetes; noisy
    egfr = np.clip(
        rng.normal(92 - (age - 40) * 0.75 - diabetes * 10, 14 + cfg.noise * 8, n),
        10,
        140,
    )

    # LDL: noisy, modest association with diabetes/smoking
    ldl = np.clip(
        rng.normal(3.2 + diabetes * 0.35 + smoking * 0.25, 0.7 + cfg.noise * 0.45, n),
        1.0,
        8.0,
    )

    # CRP: inflammatory marker, heavy tail-ish
    crp_base = rng.gamma(shape=2.0, scale=3.0, size=n)  # mean ~6
    crp = np.clip(crp_base + diabetes * 1.2 + smoking * 0.8 + rng.normal(0, 1.0 + cfg.noise * 1.5, n), 0.1, 200.0)

    # Chest pain: depends on latent "ischemia tendency" but not perfect
    ischemia_latent = (
        (age - 55) / 18
        + (systolic_bp - 130) / 35
        + (ldl - 3.5) / 1.2
        + smoking * 0.35
        + diabetes * 0.25
        + rng.normal(0, 0.7 + cfg.noise * 0.7, n)
    )
    p_chest = _sigmoid(ischemia_latent - 0.2)
    chest_pain = rng.binomial(1, np.clip(p_chest, 0.05, 0.80))

    # ECG ST change: depends on ischemia + chest_pain, still noisy
    p_ecg = _sigmoid(ischemia_latent * 0.6 + chest_pain * 0.6 - 0.7 + rng.normal(0, 0.4 + cfg.noise * 0.5, n))
    ecg_st = rng.binomial(1, np.clip(p_ecg, 0.03, 0.70))

    # Troponin: correlated with ischemia/injury but overlapping
    # Use lognormal-like behaviour (most low, some higher)
    injury_latent = ischemia_latent * 0.9 + ecg_st * 0.7 + chest_pain * 0.4 + rng.normal(0, 0.9 + cfg.noise * 1.0, n)
    troponin = np.exp(rng.normal(-3.0 + injury_latent * 0.55, 0.7 + cfg.noise * 0.55, n))  # ~0.01 typical
    troponin = np.clip(troponin, 0.001, 5.0)

    # NT-proBNP: depends on age + renal + noise
    ntprobnp = np.exp(rng.normal(5.4 + (age - 55) * 0.018 + (90 - egfr) * 0.012, 0.55 + cfg.noise * 0.35, n))
    ntprobnp = np.clip(ntprobnp, 10, 40000)

    # Event probability: not deterministic, includes unobserved noise
    risk_score = (
        (age - 55) * 0.025
        + (systolic_bp - 130) * 0.010
        + (heart_rate - 75) * 0.012
        + (3.4 - egfr / 30.0) * 0.25
        + (ldl - 3.2) * 0.22
        + np.log1p(crp) * 0.18
        + chest_pain * 0.55
        + ecg_st * 0.65
        + np.log1p(troponin) * 0.35
        + diabetes * 0.25
        + smoking * 0.18
        + rng.normal(0, 1.0 + cfg.noise * 0.9, n)  # unobserved messiness
    )

    # Calibrate to base_event_rate (approx) by shifting logits
    logits = risk_score
    # Find a shift so average sigmoid ~ base_event_rate (quick approximation)
    shift = np.quantile(logits, 1.0 - cfg.base_event_rate)
    p_event = _sigmoid(logits - shift)

    event_30d = rng.binomial(1, np.clip(p_event, 0.02, 0.85))

    # Label noise (flip a small fraction)
    if cfg.label_noise > 0:
        flip = rng.random(n) < cfg.label_noise
        event_30d = np.where(flip, 1 - event_30d, event_30d)

    df = pd.DataFrame(
        {
            "age": age.round(0).astype(int),
            "sex": sex.astype(int),
            "systolic_bp": systolic_bp.round(0).astype(int),
            "heart_rate": heart_rate.round(0).astype(int),
            "troponin": troponin.round(3),
            "ldl": ldl.round(2),
            "egfr": egfr.round(0).astype(int),
            "crp": crp.round(1),
            "chest_pain": chest_pain.astype(int),
            "ecg_st": ecg_st.astype(int),
            "diabetes": diabetes.astype(int),
            "smoking": smoking.astype(int),
            "ntprobnp": ntprobnp.round(0).astype(int),
            "event_30d": event_30d.astype(int),
        }
    )

    # Missingness: add missing values to certain columns (labs more likely)
    miss_cols = ["troponin", "ldl", "egfr", "crp", "ntprobnp"]
    for c in miss_cols:
        mask = rng.random(n) < np.clip(cfg.missingness, 0, 0.6)
        df.loc[mask, c] = np.nan

    # A little missingness in vitals too (rare)
    for c in ["systolic_bp", "heart_rate"]:
        mask = rng.random(n) < (cfg.missingness * 0.25)
        df.loc[mask, c] = np.nan

    return df


def save_df_to_temp_csv(df: pd.DataFrame, prefix: str = "synthetic_cardiac_") -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


# -----------------------------
# Data summarisation
# -----------------------------
def suggest_target(df: pd.DataFrame) -> str:
    if "event_30d" in df.columns:
        return "event_30d"
    # fallback: choose first binary-ish col
    for c in df.columns:
        vals = df[c].dropna().unique()
        if len(vals) <= 2 and set(vals).issubset({0, 1, 0.0, 1.0}):
            return c
    return ""


def compute_summary_stats(df: pd.DataFrame, target: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["rows"] = int(df.shape[0])
    out["cols"] = int(df.shape[1])
    out["columns"] = list(df.columns)

    tgt = target.strip() if target else ""
    if not tgt:
        tgt = suggest_target(df)
    out["suggested_target"] = tgt

    # class balance
    if tgt and tgt in df.columns:
        series = df[tgt].dropna()
        # accept 0/1 only
        try:
            pos_rate = float(series.mean()) if len(series) else float("nan")
        except Exception:
            pos_rate = float("nan")
        out["positive_rate"] = pos_rate
        out["positive_n"] = int((series == 1).sum()) if len(series) else 0
        out["negative_n"] = int((series == 0).sum()) if len(series) else 0
        out["target_missing"] = int(df[tgt].isna().sum())
    else:
        out["positive_rate"] = None

    # missingness
    miss = (df.isna().mean() * 100.0).round(1).sort_values(ascending=False)
    miss_df = miss.reset_index()
    miss_df.columns = ["column", "missing_%"]
    out["missing_df"] = miss_df

    # numeric ranges for key columns (exec friendly)
    key_cols = [c for c in ["age", "systolic_bp", "heart_rate", "troponin", "egfr", "crp", "ntprobnp", "ldl"] if c in df.columns]
    ranges = []
    for c in key_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        ranges.append(
            {
                "column": c,
                "min": float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else np.nan,
                "median": float(np.nanmedian(s)) if np.isfinite(np.nanmedian(s)) else np.nan,
                "max": float(np.nanmax(s)) if np.isfinite(np.nanmax(s)) else np.nan,
            }
        )
    ranges_df = pd.DataFrame(ranges)
    out["ranges_df"] = ranges_df

    return out


def summary_markdown(stats: Dict[str, Any]) -> str:
    rows = stats.get("rows", "?")
    cols = stats.get("cols", "?")
    tgt = stats.get("suggested_target", "")
    pos_rate = stats.get("positive_rate", None)

    lines = []
    lines.append("### Dataset summary")
    lines.append(f"- **Rows:** {rows}")
    lines.append(f"- **Columns:** {cols}")
    if tgt:
        lines.append(f"- **Suggested binary target:** `{tgt}`")
    if isinstance(pos_rate, (int, float)) and pos_rate is not None and not (isinstance(pos_rate, float) and math.isnan(pos_rate)):
        lines.append(f"- **Class balance (positive rate):** {pos_rate:.3f}  *(% of rows where target=1)*")
        lines.append(f"- Positives: {stats.get('positive_n', 0)}  |  Negatives: {stats.get('negative_n', 0)}  |  Target missing: {stats.get('target_missing', 0)}")
    lines.append("")
    lines.append("**Demo stance:** synthetic-only, no patient-identifiable input. Not clinical advice.")
    return "\n".join(lines)


def realism_warnings(stats: Dict[str, Any], auc: Optional[float]) -> List[str]:
    warnings = []
    # missingness too low is suspiciously clean
    miss_df: pd.DataFrame = stats.get("missing_df", pd.DataFrame())
    if isinstance(miss_df, pd.DataFrame) and not miss_df.empty:
        top_missing = float(miss_df["missing_%"].max())
        if top_missing < 1.0:
            warnings.append("Dataset is *very clean* (almost no missingness). Real clinical data usually has missing values.")
    # class balance extreme
    pr = stats.get("positive_rate", None)
    if isinstance(pr, (int, float)) and pr is not None and not (isinstance(pr, float) and math.isnan(pr)):
        if pr < 0.05 or pr > 0.60:
            warnings.append("Class balance is extreme (very low/high event rate). This can inflate or destabilise performance metrics.")
    # suspiciously high auc
    if isinstance(auc, (int, float)) and auc is not None:
        if auc > 0.95:
            warnings.append("AUC is suspiciously high. This often indicates leakage or overly rule-clean synthetic generation (too easy to separate).")
    return warnings


# -----------------------------
# Model training + outputs
# -----------------------------
@dataclass
class ModelSnapshot:
    trained: bool
    model_name: str
    target: str
    features: List[str]
    rows: int
    auc: Optional[float]
    accuracy: Optional[float]
    coef_df: pd.DataFrame
    roc_fig: Any  # matplotlib fig or None
    notes: List[str]


def train_baseline_model(df: pd.DataFrame, target: str, features: List[str]) -> ModelSnapshot:
    if not SKLEARN_OK:
        return ModelSnapshot(
            trained=False,
            model_name="Logistic Regression (baseline)",
            target=target,
            features=features,
            rows=int(df.shape[0]),
            auc=None,
            accuracy=None,
            coef_df=pd.DataFrame([{"error": f"scikit-learn is not available: {SKLEARN_ERR}"}]),
            roc_fig=None,
            notes=["Install scikit-learn to enable training: pip install scikit-learn"],
        )

    if not MPL_OK:
        return ModelSnapshot(
            trained=False,
            model_name="Logistic Regression (baseline)",
            target=target,
            features=features,
            rows=int(df.shape[0]),
            auc=None,
            accuracy=None,
            coef_df=pd.DataFrame([{"error": f"matplotlib is not available: {MPL_ERR}"}]),
            roc_fig=None,
            notes=["Install matplotlib to enable ROC plotting: pip install matplotlib"],
        )

    if not target or target not in df.columns:
        return ModelSnapshot(
            trained=False,
            model_name="Logistic Regression (baseline)",
            target=target,
            features=features,
            rows=int(df.shape[0]),
            auc=None,
            accuracy=None,
            coef_df=pd.DataFrame([{"error": "Select a valid binary target column."}]),
            roc_fig=None,
            notes=[],
        )

    if not features:
        return ModelSnapshot(
            trained=False,
            model_name="Logistic Regression (baseline)",
            target=target,
            features=features,
            rows=int(df.shape[0]),
            auc=None,
            accuracy=None,
            coef_df=pd.DataFrame([{"error": "Select at least 1 feature column."}]),
            roc_fig=None,
            notes=[],
        )

    # Prepare X/y
    X = df[features].copy()
    y = df[target].copy()

    # Ensure binary
    y_num = pd.to_numeric(y, errors="coerce")
    # drop rows with missing y
    mask = ~y_num.isna()
    X = X.loc[mask]
    y_num = y_num.loc[mask].astype(int)

    # Guard: must have both classes
    if y_num.nunique() < 2:
        return ModelSnapshot(
            trained=False,
            model_name="Logistic Regression (baseline)",
            target=target,
            features=features,
            rows=int(X.shape[0]),
            auc=None,
            accuracy=None,
            coef_df=pd.DataFrame([{"error": "Target column has only one class after filtering (need both 0 and 1)."}]),
            roc_fig=None,
            notes=[],
        )

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_num, test_size=0.25, random_state=42, stratify=y_num
    )

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    acc = float(accuracy_score(y_test, pred))

    # Coefficients
    model: LogisticRegression = pipe.named_steps["model"]
    coefs = model.coef_.reshape(-1)
    coef_df = pd.DataFrame({"feature": features, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"]).reset_index(drop=True)

    # ROC figure
    fpr, tpr, _ = roc_curve(y_test, proba)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label="Model ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("ROC Curve (Logistic Regression baseline)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)

    return ModelSnapshot(
        trained=True,
        model_name="Logistic Regression (baseline)",
        target=target,
        features=features,
        rows=int(X.shape[0]),
        auc=auc,
        accuracy=acc,
        coef_df=coef_df,
        roc_fig=fig,
        notes=[],
    )


def build_auto_context(
    stats: Dict[str, Any],
    snapshot: Optional[ModelSnapshot],
    user_context: str,
) -> str:
    """
    Build a robust text snapshot for the LLM so it can explain the CURRENT results.
    This is the key to making the assistant “gold dust”.
    """
    lines = []
    lines.append("DEMO CONTEXT")
    lines.append("- Purpose: learning + capability uplift (synthetic-only). Not clinical advice.")
    lines.append("")
    lines.append("DATASET")
    lines.append(f"- Rows: {stats.get('rows')}, Columns: {stats.get('cols')}")
    tgt = stats.get("suggested_target", "")
    if tgt:
        lines.append(f"- Suggested target: {tgt}")
    pr = stats.get("positive_rate", None)
    if isinstance(pr, (int, float)) and pr is not None and not (isinstance(pr, float) and math.isnan(pr)):
        lines.append(f"- Class balance (positive rate): {pr:.3f}")
        lines.append(f"  - Positives: {stats.get('positive_n')} | Negatives: {stats.get('negative_n')} | Target missing: {stats.get('target_missing')}")
    lines.append("")

    # Top missingness
    miss_df: pd.DataFrame = stats.get("missing_df", pd.DataFrame())
    if isinstance(miss_df, pd.DataFrame) and not miss_df.empty:
        top5 = miss_df.head(6)
        lines.append("MISSINGNESS (top columns)")
        for _, r in top5.iterrows():
            lines.append(f"- {r['column']}: {r['missing_%']}% missing")
        lines.append("")

    # Key ranges
    ranges_df: pd.DataFrame = stats.get("ranges_df", pd.DataFrame())
    if isinstance(ranges_df, pd.DataFrame) and not ranges_df.empty:
        lines.append("KEY NUMERIC RANGES (min / median / max)")
        for _, r in ranges_df.iterrows():
            lines.append(f"- {r['column']}: {r['min']:.3g} / {r['median']:.3g} / {r['max']:.3g}")
        lines.append("")

    if snapshot and snapshot.trained:
        lines.append("MODEL")
        lines.append(f"- Type: {snapshot.model_name}")
        lines.append(f"- Target: {snapshot.target}")
        lines.append(f"- Features used ({len(snapshot.features)}): {', '.join(snapshot.features)}")
        lines.append(f"- AUC: {snapshot.auc:.3f}  (AUC = Area Under the Curve; how well the model separates 1 vs 0 across thresholds)")
        lines.append(f"- Accuracy @ 0.5: {snapshot.accuracy:.3f}  (Accuracy = % correct at a fixed cut-off)")
        lines.append("")
        lines.append("TOP DRIVERS (coefficients, direction)")
        # Show top 8
        top = snapshot.coef_df.head(8)
        for _, r in top.iterrows():
            direction = "increases predicted chance" if r["coef"] > 0 else "decreases predicted chance"
            lines.append(f"- {r['feature']}: {r['coef']:+.3f} ({direction})")
        lines.append("")
    else:
        lines.append("MODEL")
        lines.append("- Not trained yet (no metrics available).")
        lines.append("")

    # User context box (optional)
    if user_context and user_context.strip():
        lines.append("USER CONTEXT (manual)")
        lines.append(user_context.strip())
        lines.append("")

    return "\n".join(lines).strip()


# -----------------------------
# Gradio callbacks
# -----------------------------
def generate_dataset_cb(n_rows, event_rate, noise, label_noise, missingness, seed):
    cfg = SynthConfig(
        n_rows=int(n_rows),
        base_event_rate=float(event_rate),
        noise=float(noise),
        label_noise=float(label_noise),
        missingness=float(missingness),
        seed=int(seed),
    )
    df = generate_synthetic_cardiac(cfg)
    path = save_df_to_temp_csv(df, prefix="synthetic_cardiac_realistic_")
    stats = compute_summary_stats(df, target="event_30d" if "event_30d" in df.columns else "")
    md = summary_markdown(stats)
    miss_df = stats["missing_df"]
    ranges_df = stats["ranges_df"]
    # return: generated file, preview df, summary md, missing df, ranges df, target choices, feature choices, status
    target_choices = [c for c in df.columns if df[c].dropna().nunique() <= 2 and set(df[c].dropna().unique()).issubset({0, 1})]
    if "event_30d" in df.columns and "event_30d" not in target_choices:
        target_choices = ["event_30d"] + target_choices
    feature_choices = [c for c in df.columns if c not in target_choices]
    status = f"Generated dataset: {os.path.basename(path)} (rows={df.shape[0]}, cols={df.shape[1]})"
    return (
        path,
        df.head(25),
        md,
        miss_df,
        ranges_df,
        gr.Dropdown(choices=target_choices, value=("event_30d" if "event_30d" in df.columns else (target_choices[0] if target_choices else ""))),
        gr.Dropdown(choices=feature_choices, value=[c for c in ["age","sex","systolic_bp","heart_rate","troponin","ldl","egfr","crp","chest_pain","ecg_st","diabetes","smoking","ntprobnp"] if c in feature_choices], multiselect=True),
        status,
        df.to_json(orient="split"),
    )


def load_csv_cb(file_obj):
    if file_obj is None:
        return (
            None,
            pd.DataFrame(),
            "### Dataset summary\n- No file loaded yet.",
            pd.DataFrame(),
            pd.DataFrame(),
            gr.Dropdown(choices=[], value=""),
            gr.Dropdown(choices=[], value=[], multiselect=True),
            "No CSV loaded.",
            None,
        )

    # file_obj can be path-like or dict depending on gradio version
    path = None
    if isinstance(file_obj, str):
        path = file_obj
    elif isinstance(file_obj, dict) and "name" in file_obj:
        path = file_obj["name"]
    else:
        try:
            path = file_obj.name
        except Exception:
            path = None

    if not path or not os.path.exists(path):
        return (
            None,
            pd.DataFrame(),
            "### Dataset summary\n- Could not read the uploaded file path.",
            pd.DataFrame(),
            pd.DataFrame(),
            gr.Dropdown(choices=[], value=""),
            gr.Dropdown(choices=[], value=[], multiselect=True),
            "Upload failed (file path not found).",
            None,
        )

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return (
            None,
            pd.DataFrame(),
            f"### Dataset summary\n- CSV parse error: {e}",
            pd.DataFrame(),
            pd.DataFrame(),
            gr.Dropdown(choices=[], value=""),
            gr.Dropdown(choices=[], value=[], multiselect=True),
            "CSV parse failed.",
            None,
        )

    tgt_guess = suggest_target(df)
    stats = compute_summary_stats(df, target=tgt_guess)
    md = summary_markdown(stats)
    miss_df = stats["missing_df"]
    ranges_df = stats["ranges_df"]

    target_choices = [c for c in df.columns if df[c].dropna().nunique() <= 2 and set(df[c].dropna().unique()).issubset({0, 1})]
    if tgt_guess and tgt_guess in df.columns and tgt_guess not in target_choices:
        target_choices = [tgt_guess] + target_choices
    feature_choices = [c for c in df.columns if c not in target_choices]
    status = f"Loaded: {os.path.basename(path)} (rows={df.shape[0]}, cols={df.shape[1]})"

    # Choose a default features set that avoids obvious leakage names (best-effort)
    default_features = [c for c in ["age","sex","systolic_bp","heart_rate","troponin","ldl","egfr","crp","chest_pain","ecg_st","diabetes","smoking","ntprobnp"] if c in feature_choices]

    return (
        path,
        df.head(25),
        md,
        miss_df,
        ranges_df,
        gr.Dropdown(choices=target_choices, value=(tgt_guess if tgt_guess in target_choices else (target_choices[0] if target_choices else ""))),
        gr.Dropdown(choices=feature_choices, value=default_features, multiselect=True),
        status,
        df.to_json(orient="split"),
    )


def train_model_cb(df_json, target, features):
    if not df_json:
        empty = ModelSnapshot(
            trained=False,
            model_name="Logistic Regression (baseline)",
            target=str(target),
            features=list(features or []),
            rows=0,
            auc=None,
            accuracy=None,
            coef_df=pd.DataFrame([{"error": "No dataset loaded yet."}]),
            roc_fig=None,
            notes=[],
        )
        return (
            "### Results (exec view)\n- No dataset loaded yet.",
            pd.DataFrame(),
            None,
            json.dumps(empty.__dict__, default=str),
        )

    df = pd.read_json(df_json, orient="split")
    snapshot = train_baseline_model(df, str(target), list(features or []))

    # Build exec markdown
    stats = compute_summary_stats(df, target=str(target))
    warn = realism_warnings(stats, snapshot.auc if snapshot.trained else None)

    lines = []
    lines.append("### Results (exec view)")
    lines.append(f"- **Model:** {snapshot.model_name}")
    lines.append(f"- **Rows used:** {snapshot.rows}")
    if snapshot.trained:
        lines.append(f"- **AUC:** {snapshot.auc:.3f}")
        lines.append(f"- **Accuracy (0.5 cut-off):** {snapshot.accuracy:.3f}")
        lines.append(f"- **Target:** `{snapshot.target}`")
        lines.append(f"- **Features:** {len(snapshot.features)} selected")
    else:
        lines.append("- **Status:** Not trained (fix errors below).")

    if warn:
        lines.append("")
        lines.append("**Caveats detected (important):**")
        for w in warn:
            lines.append(f"- {w}")

    if not SKLEARN_OK:
        lines.append("")
        lines.append(f"**Training blocked:** scikit-learn import failed: `{SKLEARN_ERR}`")
        lines.append("Install: `pip install scikit-learn`")

    if snapshot.notes:
        lines.append("")
        for n in snapshot.notes:
            lines.append(f"- {n}")

    exec_md = "\n".join(lines)

    # Return coef table and roc plot
    coef_df = snapshot.coef_df if isinstance(snapshot.coef_df, pd.DataFrame) else pd.DataFrame()
    roc_fig = snapshot.roc_fig if snapshot.trained else None

    snap_json = json.dumps(
        {
            "trained": snapshot.trained,
            "model_name": snapshot.model_name,
            "target": snapshot.target,
            "features": snapshot.features,
            "rows": snapshot.rows,
            "auc": snapshot.auc,
            "accuracy": snapshot.accuracy,
            "notes": snapshot.notes,
            # coefficients included separately in UI
        },
        default=str,
    )

    return exec_md, coef_df, roc_fig, snap_json


def build_context_state(df_json, target, features, model_snapshot_json, user_context):
    """
    Create the auto-context for the assistant (data + model outputs).
    """
    if not df_json:
        return "No dataset loaded yet. Load or generate a synthetic CSV first."

    df = pd.read_json(df_json, orient="split")
    stats = compute_summary_stats(df, target=str(target))

    snapshot = None
    try:
        ms = json.loads(model_snapshot_json) if model_snapshot_json else {}
        if ms.get("trained"):
            # We also want top coeffs for context; recompute quickly from training output if needed
            # But simplest: train_model_cb already computed coef_df separately; here we rebuild minimal snapshot.
            snapshot = ModelSnapshot(
                trained=bool(ms.get("trained")),
                model_name=str(ms.get("model_name", "Logistic Regression (baseline)")),
                target=str(ms.get("target", target)),
                features=list(ms.get("features", features or [])),
                rows=int(ms.get("rows", df.shape[0])),
                auc=float(ms.get("auc")) if ms.get("auc") is not None else None,
                accuracy=float(ms.get("accuracy")) if ms.get("accuracy") is not None else None,
                coef_df=pd.DataFrame(),  # not embedded here
                roc_fig=None,
                notes=list(ms.get("notes", [])),
            )
    except Exception:
        snapshot = None

    # If we have no coefficient df in snapshot, it's okay; the assistant still gets AUC/acc + features.
    # But we CAN attach a short coefficient list if the model exists by re-training quickly (optional).
    # To keep it reliable and avoid surprises, only include coefficients if sklearn is available AND target/features look valid.
    if snapshot and snapshot.trained and SKLEARN_OK and features and target:
        # retrain quickly to capture coefficients for context (same seed split)
        snap2 = train_baseline_model(df, str(target), list(features))
        if snap2.trained:
            snapshot.coef_df = snap2.coef_df

    return build_auto_context(stats, snapshot, str(user_context or ""))


def assistant_chat_cb(user_message, auto_context, history):
    history = history or []
    answer, ms = ask_llm(user_message, auto_context, history)
    answer = answer.strip() + f"\n\n—\n*Response time: {ms:.0f} ms · Model: {MODEL_NAME} · Synthetic demo*"
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    return "", history


def assistant_clear_cb():
    return []


def assistant_preset_cb(which: str, auto_context: str, history):
    """
    Buttons that generate locked, purpose-specific commentary.
    """
    history = history or []

    prompts = {
        "exec": "Generate an executive briefing of the CURRENT results: 1) what this demo proves, 2) key KPIs, 3) what we would do next.",
        "risks": "List risks/caveats for the CURRENT results (data realism, leakage risk, class balance, limitations). Keep it short and concrete.",
        "improve": "Suggest the next 5 improvements for THIS demo (data realism, UI, model, governance, deployment). Make them practical and low-risk.",
    }
    user_message = prompts.get(which, prompts["exec"])
    answer, ms = ask_llm(user_message, auto_context, history)
    answer = answer.strip() + f"\n\n—\n*Response time: {ms:.0f} ms · Model: {MODEL_NAME} · Synthetic demo*"
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    return history


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Clinical ML Tutor (Synthetic Demo)", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        f"""
<div class="kch-hero">
  <div class="kch-title">Clinical ML Tutor</div>
  <div class="kch-sub">
    AI partner for exploring a <b>synthetic cardiac dataset</b> — with explainable modelling + plain-English commentary
    (demo stance, not clinical advice).
_toggle: generate realistic synthetic data • train baseline model • explain outputs for executives
  </div>
  <div class="kch-badges">
    <span class="kch-badge"><span class="kch-dot"></span> Synthetic-only guidance</span>
    <span class="kch-badge"><span class="kch-dot"></span> Plain-English explanations</span>
    <span class="kch-badge"><span class="kch-dot"></span> Executive-ready summaries</span>
  </div>
</div>
"""
    )

    # States
    df_json_state = gr.State(value=None)          # dataset in json
    model_snapshot_state = gr.State(value="")     # model snapshot in json
    auto_context_state = gr.State(value="")       # built context for assistant

    with gr.Row():
        # -------------------------
        # LEFT: Data + Model
        # -------------------------
        with gr.Column(scale=1):
            gr.Markdown('<div class="kch-card"><h3>Data (synthetic CSV)</h3><p>Upload — or generate a realistic synthetic dataset (with noise, missingness, and overlap so results are not “too perfect”).</p></div>')

            with gr.Row():
                upload_csv = gr.File(label="Upload synthetic CSV", file_types=[".csv"])
                generated_csv = gr.File(label="Generated synthetic CSV (download)", interactive=False)

            status = gr.Textbox(label="Status", value="Ready.", interactive=False)

            with gr.Accordion("Generate realistic synthetic dataset", open=True):
                with gr.Row():
                    n_rows = gr.Slider(50, 2000, value=250, step=10, label="Rows")
                    event_rate = gr.Slider(0.05, 0.60, value=0.22, step=0.01, label="Base event rate (approx)")
                with gr.Row():
                    noise = gr.Slider(0.10, 1.20, value=0.55, step=0.05, label="Noise / overlap")
                    label_noise = gr.Slider(0.00, 0.15, value=0.04, step=0.01, label="Label noise (flip %)")
                with gr.Row():
                    missingness = gr.Slider(0.00, 0.35, value=0.10, step=0.01, label="Missingness (labs/vitals)")
                    seed = gr.Number(value=42, precision=0, label="Random seed")
                gen_btn = gr.Button("Generate dataset", elem_classes=["primary"])

            gr.Markdown('<div class="kch-card"><h3>Dataset preview (first 25 rows)</h3><p>Quick visual check — then confirm summary stats below (class balance + missingness).</p></div>')
            preview_df = gr.Dataframe(label="Preview", interactive=False)

            with gr.Row():
                with gr.Column(scale=1):
                    summary_md = gr.Markdown("### Dataset summary\n- No dataset loaded yet.")
                with gr.Column(scale=1):
                    miss_table = gr.Dataframe(label="Missingness (% missing)", interactive=False)
            ranges_table = gr.Dataframe(label="Key numeric ranges (min / median / max)", interactive=False)

            gr.Markdown('<div class="kch-card"><h3>Model controls (binary classification)</h3><p>Train an explainable baseline model (Logistic Regression). Then use the assistant to explain the outputs.</p></div>')
            target_dd = gr.Dropdown(label="Target (binary 0/1)")
            features_dd = gr.Dropdown(label="Features (numeric)", multiselect=True)

            train_btn = gr.Button("Train baseline model (Logistic Regression)", elem_classes=["primary"])

            gr.Markdown('<div class="kch-card"><h3>Results (exec view)</h3><p>KPI summary + coefficients + ROC curve.</p></div>')
            results_md = gr.Markdown("### Results (exec view)\n- Train a model to see KPIs.")
            coef_table = gr.Dataframe(label="Coefficients (top drivers)", interactive=False)
            roc_plot = gr.Plot(label="ROC Curve")

        # -------------------------
        # RIGHT: Assistant
        # -------------------------
        with gr.Column(scale=1):
            gr.Markdown('<div class="kch-card"><h3>AI Analyst Commentary</h3><p>Locked scope: explains the <b>current</b> dataset + model outputs and caveats — not a general chatbot.</p></div>')

            with gr.Row():
                btn_exec = gr.Button("Generate executive briefing", elem_classes=["primary"])
                btn_risks = gr.Button("Risks / caveats", elem_classes=["secondary"])
                btn_improve = gr.Button("What should we improve next?", elem_classes=["secondary"])

            user_context = gr.Textbox(
                label="Context (optional)",
                placeholder="Example: I’m learning what each header means; I just trained a baseline model; explain results simply for an exec audience.",
                lines=3,
            )

            chatbot = gr.Chatbot(label="Commentary", height=420)  # Gradio 6.x messages format by default
            gr.Markdown('<div class="kch-chat-note">Tip: Use “Generate executive briefing” after training to get a board-friendly explanation of AUC, accuracy, drivers, and caveats (e.g., “too perfect” synthetic data).</div>')

            with gr.Row():
                explain_btn = gr.Button("Explain current results", elem_classes=["primary"])
                clear_btn = gr.Button("Clear commentary", elem_classes=["secondary"])

            msg = gr.Textbox(label="Your question", placeholder="e.g., What does AUC mean? Why is performance so high? Which variables drive risk?", lines=2)
            send_btn = gr.Button("Send", elem_classes=["primary"])

    # -----------------------------
    # Wiring: generate + load
    # -----------------------------
    gen_btn.click(
        fn=generate_dataset_cb,
        inputs=[n_rows, event_rate, noise, label_noise, missingness, seed],
        outputs=[
            generated_csv,
            preview_df,
            summary_md,
            miss_table,
            ranges_table,
            target_dd,
            features_dd,
            status,
            df_json_state,
        ],
    )

    upload_csv.change(
        fn=load_csv_cb,
        inputs=[upload_csv],
        outputs=[
            generated_csv,  # show loaded file path in same slot (fine)
            preview_df,
            summary_md,
            miss_table,
            ranges_table,
            target_dd,
            features_dd,
            status,
            df_json_state,
        ],
    )

    # -----------------------------
    # Train model
    # -----------------------------
    train_btn.click(
        fn=train_model_cb,
        inputs=[df_json_state, target_dd, features_dd],
        outputs=[results_md, coef_table, roc_plot, model_snapshot_state],
    )

    # -----------------------------
    # Build auto-context (whenever key things change)
    # -----------------------------
    # Build context when:
    # - dataset changes, target/features change, model snapshot changes, or user context changes
    for trigger in [df_json_state, target_dd, features_dd, model_snapshot_state, user_context]:
        trigger.change(
            fn=build_context_state,
            inputs=[df_json_state, target_dd, features_dd, model_snapshot_state, user_context],
            outputs=[auto_context_state],
        )

    # Also bind explain button to create a “structured response” without requiring user to type
    explain_btn.click(
        fn=assistant_preset_cb,
        inputs=[gr.State("exec"), auto_context_state, chatbot],
        outputs=[chatbot],
    )

    # Preset buttons
    btn_exec.click(fn=assistant_preset_cb, inputs=[gr.State("exec"), auto_context_state, chatbot], outputs=[chatbot])
    btn_risks.click(fn=assistant_preset_cb, inputs=[gr.State("risks"), auto_context_state, chatbot], outputs=[chatbot])
    btn_improve.click(fn=assistant_preset_cb, inputs=[gr.State("improve"), auto_context_state, chatbot], outputs=[chatbot])

    # Chat send
    send_btn.click(fn=assistant_chat_cb, inputs=[msg, auto_context_state, chatbot], outputs=[msg, chatbot])
    msg.submit(fn=assistant_chat_cb, inputs=[msg, auto_context_state, chatbot], outputs=[msg, chatbot])

    # Clear
    clear_btn.click(fn=assistant_clear_cb, inputs=None, outputs=[chatbot])

    # Footer safety note
    gr.Markdown(
        f"""
<div style="margin-top:16px; padding:10px 12px; border-radius:12px; border:1px solid {NHS_BORDER}; background:#FFFFFF; color:{NHS_MUTED}; font-size:12px;">
<b>Safety:</b> Synthetic demo only. No patient-identifiable input. Not clinical advice. Designed for learning and capability uplift.
</div>
"""
    )

if __name__ == "__main__":
    demo.launch()
