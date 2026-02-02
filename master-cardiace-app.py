import os
import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr

from openai import OpenAI

# Optional ML/plot deps (we handle cleanly if missing)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False


# =============================================================================
# CONFIG
# =============================================================================
APP_TITLE = "Clinical ML Tutor"

# NOTE: This is your existing pattern. Keep it.
MODEL_NAME_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # cheaper default; override via env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT_LOCKED = (
    "You are the Clinical ML Tutor.\n"
    "You are NOT a general chatbot.\n"
    "Your job: explain ONLY the CURRENT synthetic dataset + CURRENT baseline model outputs.\n"
    "Style: plain English, short sections, bullet points, executive-friendly.\n"
    "If you use a technical term, define it immediately.\n"
    "Always include: (1) what we trained, (2) what the metrics mean, (3) top drivers direction,\n"
    "(4) caveats (synthetic, leakage, perfect-metric suspicion), (5) next improvement steps.\n"
    "Never give clinical advice. Assume synthetic demo only.\n"
)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =============================================================================
# UTIL: OpenAI response extraction (keeps your working approach)
# =============================================================================
def _extract_output_text(resp) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        return resp.output_text.strip()

    out = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str) and t.strip():
                    out.append(t)
    return ("\n".join(out)).strip() or "No text returned from the model."


def ask_llm_locked(user_message: str, context: str, history_messages: List[Dict[str, str]]) -> str:
    if not client:
        return (
            "OPENAI_API_KEY not set.\n\n"
            "Set it in your environment and restart:\n"
            "  setx OPENAI_API_KEY \"your_key_here\"\n"
        )

    if not user_message or not user_message.strip():
        return "Type a question (e.g., “What does AUC mean?”)."

    messages = [{"role": "system", "content": SYSTEM_PROMPT_LOCKED}]

    if context and context.strip():
        messages.append({"role": "user", "content": "CONTEXT (current run):\n" + context.strip()})

    # History MUST be role/content dicts (Gradio 6.x Chatbot messages format)
    for m in history_messages or []:
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": str(m["content"])})

    messages.append({"role": "user", "content": user_message.strip()})

    model_name = os.getenv("OPENAI_MODEL", MODEL_NAME_DEFAULT)
    t0 = time.time()
    resp = client.responses.create(model=model_name, input=messages)
    dt_ms = int((time.time() - t0) * 1000)

    txt = _extract_output_text(resp)
    footer = f"\n\n—\nResponse time: {dt_ms} ms • Model: {model_name} • Synthetic demo"
    return (txt + footer).strip()


# =============================================================================
# DATA GENERATION: realistic synthetic cardiac dataset
# =============================================================================
CARDIAC_COLS = [
    "age", "sex", "systolic_bp", "heart_rate", "troponin",
    "ldl", "egfr", "crp", "chest_pain", "ecg_st", "diabetes",
    "smoking", "event_30d", "ntprobnp"
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_realistic_synthetic(
    n_rows: int = 500,
    base_event_rate: float = 0.20,
    overlap: float = 0.55,
    label_flip: float = 0.04,
    missingness: float = 0.10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a deliberately 'not-too-perfect' synthetic dataset:
    - correlated features
    - overlapping classes
    - missingness in labs/vitals
    - small label noise (flip %)
    """
    rng = np.random.default_rng(seed)

    n = int(max(50, min(2000, n_rows)))
    base_event_rate = float(np.clip(base_event_rate, 0.05, 0.60))
    overlap = float(np.clip(overlap, 0.10, 1.20))
    label_flip = float(np.clip(label_flip, 0.00, 0.15))
    missingness = float(np.clip(missingness, 0.00, 0.35))

    # Demographics
    age = rng.normal(62, 12, n).clip(18, 95)
    sex = rng.integers(0, 2, n)  # 0/1

    # Vitals (correlated with age + some randomness)
    systolic_bp = (120 + 0.45 * (age - 50) + rng.normal(0, 14, n)).clip(85, 220)
    heart_rate = (72 + 0.18 * (systolic_bp - 120) + rng.normal(0, 10, n)).clip(40, 160)

    # Comorbidities
    diabetes = rng.binomial(1, _sigmoid((age - 55) / 10) * 0.55, n)
    smoking = rng.binomial(1, 0.22 + 0.05 * sex - 0.0008 * (age - 50), n).clip(0, 1)

    # Symptoms/ECG (weak-to-moderate correlation)
    chest_pain = rng.binomial(1, _sigmoid((systolic_bp - 135) / 18) * 0.55, n)
    ecg_st = rng.binomial(1, _sigmoid((heart_rate - 85) / 15) * 0.40, n)

    # Labs (skewed, noisy; correlated)
    # eGFR: lower with age + diabetes
    egfr = (95 - 0.65 * (age - 45) - 8 * diabetes + rng.normal(0, 12, n)).clip(10, 130)

    # CRP: skewed inflammation marker, higher with diabetes & smoking
    crp = rng.lognormal(mean=1.2 + 0.25 * diabetes + 0.18 * smoking, sigma=0.55, size=n).clip(0.1, 200)

    # LDL: mildly related to age, smoking
    ldl = (3.2 + 0.01 * (age - 55) + 0.25 * smoking + rng.normal(0, 0.7, n)).clip(0.8, 8.5)

    # NT-proBNP: skewed; higher with age + lower egfr
    ntprobnp = rng.lognormal(mean=5.0 + 0.015 * (age - 60) + 0.008 * (90 - egfr), sigma=0.7, size=n).clip(10, 40000)

    # Troponin: mostly low, rises with ecg changes + pain + renal impairment
    # We'll create baseline low values + spikes
    trop_base = rng.lognormal(mean=-4.8, sigma=0.7, size=n)  # tiny baseline
    trop_spike = rng.lognormal(mean=-2.3 + 0.55 * ecg_st + 0.45 * chest_pain + 0.010 * (90 - egfr), sigma=0.75, size=n)
    troponin = (0.6 * trop_base + 0.4 * trop_spike).clip(0.0005, 20)

    # Create event probability from a "true" latent risk model, then force base rate
    # overlap controls separation (bigger overlap -> harder classification)
    z = (
        0.035 * (age - 60)
        + 0.018 * (systolic_bp - 130)
        + 0.020 * (heart_rate - 75)
        + 0.65 * ecg_st
        + 0.45 * chest_pain
        + 0.35 * diabetes
        + 0.22 * smoking
        + 0.30 * np.log1p(crp)
        + 0.25 * np.log1p(ntprobnp / 100)
        + 0.40 * np.log1p(troponin * 1000)
        - 0.020 * (egfr - 80)
        + rng.normal(0, 1.0 * overlap, n)
    )

    p_raw = _sigmoid(z)

    # Calibrate to approximate base_event_rate by shifting logits
    # Find shift s such that mean(sigmoid(logit(p_raw)+s)) ~= base_event_rate
    eps = 1e-6
    logits = np.log(np.clip(p_raw, eps, 1 - eps) / np.clip(1 - p_raw, eps, 1 - eps))
    # simple binary search
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p_mid = _sigmoid(logits + mid)
        if p_mid.mean() > base_event_rate:
            hi = mid
        else:
            lo = mid
    p = _sigmoid(logits + (lo + hi) / 2)

    event_30d = rng.binomial(1, p, n)

    # Label noise (flip a small %)
    if label_flip > 0:
        flip_mask = rng.random(n) < label_flip
        event_30d = np.where(flip_mask, 1 - event_30d, event_30d)

    df = pd.DataFrame({
        "age": np.round(age).astype(int),
        "sex": sex.astype(int),
        "systolic_bp": np.round(systolic_bp).astype(int),
        "heart_rate": np.round(heart_rate).astype(int),
        "troponin": np.round(troponin, 4),
        "ldl": np.round(ldl, 2),
        "egfr": np.round(egfr, 1),
        "crp": np.round(crp, 1),
        "chest_pain": chest_pain.astype(int),
        "ecg_st": ecg_st.astype(int),
        "diabetes": diabetes.astype(int),
        "smoking": smoking.astype(int),
        "event_30d": event_30d.astype(int),
        "ntprobnp": np.round(ntprobnp).astype(int),
    })

    # Missingness applied mainly to labs (troponin/ldl/egfr/crp/ntprobnp) and some vitals
    miss_cols = ["troponin", "ldl", "egfr", "crp", "ntprobnp", "systolic_bp", "heart_rate"]
    if missingness > 0:
        for col in miss_cols:
            mask = rng.random(n) < missingness
            df.loc[mask, col] = np.nan

    return df


# =============================================================================
# SUMMARY STATS (no tabulate / no markdown tables)
# =============================================================================
def dataset_summary(df: pd.DataFrame, target_col: str = "event_30d") -> Tuple[str, pd.DataFrame]:
    if df is None or df.empty:
        return "No dataset loaded.", pd.DataFrame()

    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_pct = (df.isna().mean() * 100).round(1)

    target_rate = None
    if target_col in df.columns:
        try:
            target_rate = float(df[target_col].dropna().mean())
        except Exception:
            target_rate = None

    # Numeric summary
    if numeric_cols:
        desc = df[numeric_cols].describe(percentiles=[0.05, 0.5, 0.95]).T
        desc = desc.rename(columns={
            "count": "count_non_null",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "5%": "p05",
            "50%": "p50",
            "95%": "p95",
            "max": "max",
        })
        # Keep it compact
        keep_cols = ["count_non_null", "mean", "std", "min", "p05", "p50", "p95", "max"]
        desc = desc[keep_cols].round(3)
        desc.insert(0, "missing_%", missing_pct.loc[desc.index].values)
        desc = desc.reset_index().rename(columns={"index": "feature"})
    else:
        desc = pd.DataFrame()

    lines = []
    lines.append(f"Rows: {rows:,} • Columns: {cols:,}")
    lines.append(f"Numeric features detected: {len(numeric_cols)}")
    if target_rate is not None and not math.isnan(target_rate):
        lines.append(f"Target '{target_col}' rate: {target_rate:.3f} (approx {target_rate*100:.1f}%)")
    else:
        lines.append(f"Target '{target_col}' rate: n/a")

    # Top missingness callout
    top_missing = missing_pct.sort_values(ascending=False).head(5)
    top_missing = top_missing[top_missing > 0]
    if len(top_missing) > 0:
        miss_str = ", ".join([f"{k}={v:.1f}%" for k, v in top_missing.items()])
        lines.append(f"Most missing: {miss_str}")
    else:
        lines.append("Missingness: none detected")

    summary_text = "\n".join(lines)
    return summary_text, desc


# =============================================================================
# MODEL TRAINING
# =============================================================================
@dataclass
class ModelSnapshot:
    target: str
    features: List[str]
    rows: int
    auc: Optional[float]
    acc: Optional[float]
    top_coef: pd.DataFrame
    roc_path: Optional[str]
    notes: List[str]


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def train_baseline_logreg(
    df: pd.DataFrame,
    target_col: str,
    features: List[str]
) -> ModelSnapshot:
    notes: List[str] = []

    if not SKLEARN_OK:
        notes.append("scikit-learn is not installed. Install it: pip install scikit-learn")
        return ModelSnapshot(target=target_col, features=features, rows=0, auc=None, acc=None,
                             top_coef=pd.DataFrame(), roc_path=None, notes=notes)

    if df is None or df.empty:
        notes.append("No dataset loaded.")
        return ModelSnapshot(target=target_col, features=features, rows=0, auc=None, acc=None,
                             top_coef=pd.DataFrame(), roc_path=None, notes=notes)

    if target_col not in df.columns:
        notes.append(f"Target column '{target_col}' not found in dataset.")
        return ModelSnapshot(target=target_col, features=features, rows=len(df), auc=None, acc=None,
                             top_coef=pd.DataFrame(), roc_path=None, notes=notes)

    if not features:
        notes.append("No features selected.")
        return ModelSnapshot(target=target_col, features=features, rows=len(df), auc=None, acc=None,
                             top_coef=pd.DataFrame(), roc_path=None, notes=notes)

    # Ensure numeric features exist
    features = [f for f in features if f in df.columns]
    if not features:
        notes.append("Selected features are not in dataset.")
        return ModelSnapshot(target=target_col, features=features, rows=len(df), auc=None, acc=None,
                             top_coef=pd.DataFrame(), roc_path=None, notes=notes)

    # Build training frame: drop rows where target missing
    work = df.copy()
    work = work.dropna(subset=[target_col])

    # Convert target to int 0/1
    y = work[target_col].astype(int)

    # Model uses numeric only; coerce features to numeric
    X = work[features].apply(pd.to_numeric, errors="coerce")

    # Simple impute: median for numeric
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    # Train/test split (stratify if possible)
    strat = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    # Pipeline: scale + logistic regression
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    pipe.fit(X_train, y_train)

    # Metrics
    try:
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else None
    except Exception:
        auc = None

    try:
        pred = (proba >= 0.5).astype(int) if "proba" in locals() else pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
    except Exception:
        acc = None

    auc_f = _safe_float(auc)
    acc_f = _safe_float(acc)

    # Coefficients (from LogisticRegression in pipeline)
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_.reshape(-1)
    coef_df = pd.DataFrame({"feature": features, "coef": coefs})
    coef_df["abs"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs", ascending=False).drop(columns=["abs"])

    top_coef = coef_df.head(12).copy()
    top_coef["direction"] = np.where(top_coef["coef"] >= 0, "↑ increases risk", "↓ decreases risk")
    top_coef = top_coef[["feature", "coef", "direction"]]
    top_coef["coef"] = top_coef["coef"].round(4)

    # ROC plot
    roc_path = None
    if MPL_OK and auc_f is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, proba)
            fig = plt.figure(figsize=(5.8, 4.0), dpi=140)
            ax = fig.add_subplot(111)
            ax.plot(fpr, tpr)
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title("ROC Curve (Logistic Regression)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.grid(True, alpha=0.2)

            os.makedirs("reports", exist_ok=True)
            roc_path = os.path.join("reports", f"roc_{uuid.uuid4().hex[:8]}.png")
            fig.tight_layout()
            fig.savefig(roc_path)
            plt.close(fig)
        except Exception:
            roc_path = None

    # Caveats
    if auc_f is not None and auc_f >= 0.98:
        notes.append(
            "AUC is extremely high. In real-world clinical data this is unusual; it may indicate synthetic rules are too clean, "
            "or a feature leaks the outcome (recorded after the event)."
        )
    if acc_f is not None and acc_f >= 0.98:
        notes.append(
            "Accuracy is extremely high. Treat as a demo indicator; check for leakage and increase overlap/noise in synthetic generation."
        )

    return ModelSnapshot(
        target=target_col,
        features=features,
        rows=len(work),
        auc=auc_f,
        acc=acc_f,
        top_coef=top_coef,
        roc_path=roc_path,
        notes=notes
    )


# =============================================================================
# CONTEXT PACKING for the LLM (so it can talk about *this* run)
# =============================================================================
def build_llm_context(
    df: Optional[pd.DataFrame],
    summary_text: str,
    numeric_stats: Optional[pd.DataFrame],
    snapshot: Optional[ModelSnapshot]
) -> str:
    parts = []
    parts.append("DEMO STANCE: Synthetic data only. Not clinical advice.")
    parts.append("")
    parts.append("DATASET SUMMARY:")
    parts.append(summary_text.strip() if summary_text else "n/a")

    if numeric_stats is not None and not numeric_stats.empty:
        # Provide a compact JSON-like excerpt (not huge)
        head = numeric_stats.head(10).copy()
        parts.append("")
        parts.append("NUMERIC STATS (first 10 features):")
        parts.append(head.to_json(orient="records"))

    if snapshot:
        parts.append("")
        parts.append("MODEL OUTPUTS (current run):")
        parts.append(f"Target: {snapshot.target}")
        parts.append(f"Rows used: {snapshot.rows}")
        parts.append(f"Features: {', '.join(snapshot.features)}")
        parts.append(f"AUC: {snapshot.auc if snapshot.auc is not None else 'n/a'}")
        parts.append(f"Accuracy@0.5: {snapshot.acc if snapshot.acc is not None else 'n/a'}")
        if snapshot.top_coef is not None and not snapshot.top_coef.empty:
            parts.append("Top coefficients:")
            parts.append(snapshot.top_coef.to_json(orient="records"))
        if snapshot.notes:
            parts.append("Caveats:")
            parts.append(json.dumps(snapshot.notes, ensure_ascii=False))
    else:
        parts.append("")
        parts.append("MODEL OUTPUTS: none (model not trained yet).")

    return "\n".join(parts).strip()


# =============================================================================
# GRADIO CALLBACKS
# =============================================================================
def load_csv(file: Optional[str]) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    if not file:
        return pd.DataFrame(), "No CSV loaded.", pd.DataFrame()
    try:
        df = pd.read_csv(file)
        # Standardise: keep columns as-is; preview will show whatever exists
        summary_text, numeric_stats = dataset_summary(df, target_col="event_30d")
        return df.head(25), summary_text, numeric_stats
    except Exception as e:
        return pd.DataFrame(), f"Failed to read CSV: {e}", pd.DataFrame()


def do_generate_dataset(
    n_rows: int,
    base_event_rate: float,
    overlap: float,
    label_flip: float,
    missingness: float,
    seed: int
) -> Tuple[str, str, pd.DataFrame, str, pd.DataFrame]:
    """
    Returns:
      status_text,
      generated_file_path,
      preview_df,
      summary_text,
      numeric_stats_df
    """
    try:
        df = generate_realistic_synthetic(
            n_rows=n_rows,
            base_event_rate=base_event_rate,
            overlap=overlap,
            label_flip=label_flip,
            missingness=missingness,
            seed=seed
        )
        os.makedirs("reports", exist_ok=True)
        out_path = os.path.join("reports", f"synthetic_cardiac_realistic_{uuid.uuid4().hex[:8]}.csv")
        df.to_csv(out_path, index=False)

        summary_text, numeric_stats = dataset_summary(df, target_col="event_30d")
        status = f"Generated dataset: {os.path.basename(out_path)} (rows={len(df)}, cols={df.shape[1]})"
        return status, out_path, df.head(25), summary_text, numeric_stats
    except Exception as e:
        return f"Dataset generation failed: {e}", "", pd.DataFrame(), "n/a", pd.DataFrame()


def do_train_model(
    df_full: pd.DataFrame,
    target_col: str,
    features: List[str]
) -> Tuple[str, str, str, pd.DataFrame, Optional[str], Dict[str, Any]]:
    """
    Outputs:
      kpi_model, kpi_auc, kpi_acc, coef_table, roc_image_path, model_state_json
    """
    try:
        if df_full is None or df_full.empty:
            return "No dataset", "n/a", "n/a", pd.DataFrame(), None, {}

        snap = train_baseline_logreg(df_full, target_col, features)

        kpi_model = "Logistic Regression (baseline)"
        kpi_auc = f"{snap.auc:.3f}" if snap.auc is not None else "n/a"
        kpi_acc = f"{snap.acc:.3f}" if snap.acc is not None else "n/a"

        # Minimal serialisable state for the LLM context
        state = {
            "target": snap.target,
            "features": snap.features,
            "rows": snap.rows,
            "auc": snap.auc,
            "acc": snap.acc,
            "notes": snap.notes,
            "top_coef": snap.top_coef.to_dict(orient="records") if snap.top_coef is not None else []
        }

        return kpi_model, kpi_auc, kpi_acc, snap.top_coef, snap.roc_path, state

    except Exception as e:
        # Always return valid outputs to prevent Gradio "Error" badges
        return (
            f"Training error: {e}",
            "n/a",
            "n/a",
            pd.DataFrame(),
            None,
            {"error": str(e)}
        )


def _history_append(history: List[Dict[str, str]], role: str, content: str) -> List[Dict[str, str]]:
    history = history or []
    history.append({"role": role, "content": content})
    return history


def chat(user_message: str, user_context: str, history: List[Dict[str, str]], llm_context: str) -> Tuple[str, List[Dict[str, str]]]:
    history = history or []
    user_message = (user_message or "").strip()
    if not user_message:
        return "", history

    # lock context = the current snapshot + optional user text box context
    merged_context = (llm_context or "").strip()
    if user_context and user_context.strip():
        merged_context = (merged_context + "\n\nUSER CONTEXT:\n" + user_context.strip()).strip()

    answer = ask_llm_locked(user_message=user_message, context=merged_context, history_messages=history)
    history = _history_append(history, "user", user_message)
    history = _history_append(history, "assistant", answer)
    return "", history


def clear_chat() -> List[Dict[str, str]]:
    return []


def explain_current_results(history: List[Dict[str, str]], llm_context: str) -> List[Dict[str, str]]:
    history = history or []
    prompt = (
        "Explain the current results for an executive audience.\n"
        "Use short sections:\n"
        "- What we trained (1–2 lines)\n"
        "- What AUC and Accuracy mean (define both)\n"
        "- Top drivers (direction + what it suggests)\n"
        "- Risks/caveats (including perfect-metric / leakage suspicion)\n"
        "- Next improvement steps (3 bullets)\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)
    return history


def exec_brief(history: List[Dict[str, str]], llm_context: str) -> List[Dict[str, str]]:
    history = history or []
    prompt = (
        "Write an executive briefing (max ~12 bullets) describing:\n"
        "- What this demo is\n"
        "- Why it matters (capability uplift)\n"
        "- Safety stance (synthetic only, not clinical advice)\n"
        "- What the current model output indicates\n"
        "- What we should do next\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)
    return history


def risks_caveats(history: List[Dict[str, str]], llm_context: str) -> List[Dict[str, str]]:
    history = history or []
    prompt = (
        "List the main risks / caveats of the current run.\n"
        "Include:\n"
        "- Synthetic data limitations\n"
        "- Leakage / perfect metrics suspicion\n"
        "- Small sample size / representativeness\n"
        "- Model simplicity limitations\n"
        "- Governance note (no patient-identifiable data)\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)
    return history


def improve_next(history: List[Dict[str, str]], llm_context: str) -> List[Dict[str, str]]:
    history = history or []
    prompt = (
        "Recommend next improvements, prioritised.\n"
        "Keep it practical and demo-friendly.\n"
        "Include:\n"
        "- data realism (noise, missingness, calibration)\n"
        "- evaluation (cross-validation, calibration curve)\n"
        "- interpretability (feature scaling note, SHAP optional)\n"
        "- UI/UX (what would make execs instantly understand)\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)
    return history


# =============================================================================
# THEME (NHS Blue) + CSS FIXES
# - No black-on-black
# - Make tables readable
# - Avoid relying on Dataframe(height=...)
# =============================================================================
NHS_BLUE = "#005EB8"
NHS_DARK = "#003087"
BG_SOFT = "#EEF5FF"
CARD_BG = "#FFFFFF"
TEXT = "#0B1F3A"
MUTED = "#4B5E77"
BORDER = "rgba(0, 94, 184, 0.18)"

CUSTOM_CSS = f"""
:root {{
  --nhs-blue: {NHS_BLUE};
  --nhs-dark: {NHS_DARK};
  --bg-soft: {BG_SOFT};
  --card: {CARD_BG};
  --text: {TEXT};
  --muted: {MUTED};
  --border: {BORDER};
}}

.gradio-container {{
  background: radial-gradient(1200px 700px at 20% 0%, rgba(0,94,184,0.12), transparent 55%),
              radial-gradient(900px 600px at 80% 0%, rgba(0,48,135,0.10), transparent 50%),
              var(--bg-soft) !important;
  color: var(--text) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}}

h1, h2, h3, h4, h5, h6, label, .prose {{
  color: var(--text) !important;
}}

.small-muted {{
  color: var(--muted);
  font-size: 12px;
}}

.hero {{
  background: linear-gradient(135deg, rgba(0,94,184,0.14), rgba(255,255,255,0.65));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}}

.badge-row {{
  margin-top: 10px;
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}}

.badge {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.7);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 12px;
  color: var(--text);
}}

.dot {{
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: var(--nhs-blue);
  box-shadow: 0 0 0 3px rgba(0,94,184,0.12);
}}

.card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.06);
}}

.section-title {{
  font-weight: 700;
  color: var(--nhs-dark);
  margin: 6px 0 8px 0;
}}

.primary-btn button {{
  background: var(--nhs-blue) !important;
  border: 1px solid rgba(0,94,184,0.35) !important;
  color: white !important;
  border-radius: 12px !important;
  height: 44px !important;
  font-weight: 700 !important;
}}

.secondary-btn button {{
  background: white !important;
  border: 1px solid rgba(0,94,184,0.35) !important;
  color: var(--nhs-blue) !important;
  border-radius: 12px !important;
  height: 44px !important;
  font-weight: 700 !important;
}}

.kpi-grid {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}}

.kpi {{
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.85);
}}

.kpi .label {{
  font-size: 12px;
  color: var(--muted);
}}

.kpi .value {{
  font-size: 18px;
  font-weight: 800;
  color: var(--text);
  margin-top: 4px;
}}

@media (max-width: 1100px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}

@media (max-width: 600px) {{
  .kpi-grid {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
}}

/* Dataframe readability */
table {{
  color: var(--text) !important;
}}
thead th {{
  background: rgba(0,94,184,0.08) !important;
  color: var(--text) !important;
}}
tbody td {{
  background: white !important;
  color: var(--text) !important;
}}

/* Make dataframe area scroll without using Dataframe(height=...) */
[data-testid="dataframe"] {{
  max-height: 340px;
  overflow: auto;
  border-radius: 12px;
}}
"""


# =============================================================================
# UI
# =============================================================================
with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
    gr.Markdown(
        f"""
<div class="hero">
  <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:16px;">
    <div>
      <div style="font-size:28px; font-weight:900; color:{NHS_DARK}; line-height:1.1;">Clinical ML Tutor</div>
      <div style="margin-top:8px; color:{MUTED}; max-width:980px;">
        AI partner for exploring a <b>synthetic</b> cardiac dataset — with explainable modelling + plain-English commentary
        (<b>demo stance</b>, not clinical advice).
      </div>
      <div class="badge-row">
        <span class="badge"><span class="dot"></span> Synthetic-only guidance</span>
        <span class="badge"><span class="dot"></span> Plain-English explanations</span>
        <span class="badge"><span class="dot"></span> Executive-ready summaries</span>
      </div>
    </div>
    <div style="text-align:right; color:{MUTED}; font-size:12px;">
      Demo stance<br/>
      <b>Learning & capability uplift</b> (not clinical advice)
    </div>
  </div>
</div>
"""
    )

    # App state
    df_state = gr.State(pd.DataFrame())
    model_state = gr.State({})          # serialisable snapshot for context
    llm_context_state = gr.State("")    # packed context string

    with gr.Row():
        # LEFT: Data + Model
        with gr.Column(scale=1):
            gr.Markdown("<div class='section-title'>Data (synthetic CSV)</div>")
            with gr.Group(elem_classes=["card"]):
                with gr.Row():
                    csv_in = gr.File(label="Upload synthetic CSV", file_types=[".csv"])
                    csv_generated = gr.File(label="Generated synthetic CSV (download)", interactive=False)

                status = gr.Textbox(label="Status", value="Ready.", interactive=False)

                with gr.Accordion("Generate realistic synthetic dataset", open=True):
                    with gr.Row():
                        gen_rows = gr.Slider(50, 2000, value=490, step=10, label="Rows")
                        gen_event = gr.Slider(0.05, 0.60, value=0.22, step=0.01, label="Base event rate (approx)")
                    with gr.Row():
                        gen_overlap = gr.Slider(0.10, 1.20, value=0.55, step=0.01, label="Noise / overlap")
                        gen_flip = gr.Slider(0.00, 0.15, value=0.04, step=0.01, label="Label noise (flip %)")
                    with gr.Row():
                        gen_missing = gr.Slider(0.00, 0.35, value=0.10, step=0.01, label="Missingness (labs/vitals)")
                        gen_seed = gr.Number(value=42, precision=0, label="Random seed")
                    gen_btn = gr.Button("Generate dataset", elem_classes=["primary-btn"])

                gr.Markdown("<div class='section-title'>Dataset preview (first 25 rows)</div>")
                preview_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True)

                gr.Markdown("<div class='section-title'>Dataset summary</div>")
                summary_text = gr.Textbox(value="No dataset loaded.", lines=4, interactive=False, label="Summary (plain)")
                stats_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True)

            gr.Markdown("<div class='section-title'>Model controls (binary classification)</div>")
            with gr.Group(elem_classes=["card"]):
                target = gr.Dropdown(
                    label="Target (binary 0/1)",
                    choices=["event_30d"],
                    value="event_30d",
                    allow_custom_value=True
                )
                features = gr.Dropdown(
                    label="Features (numeric)",
                    multiselect=True,
                    choices=["age", "sex", "systolic_bp", "heart_rate", "troponin", "ldl", "egfr", "crp",
                             "chest_pain", "ecg_st", "diabetes", "smoking", "ntprobnp"],
                    value=["age", "sex", "systolic_bp", "heart_rate", "troponin", "ldl", "egfr", "crp",
                           "chest_pain", "ecg_st", "diabetes", "smoking", "ntprobnp"]
                )

                train_btn = gr.Button("Train baseline model (Logistic Regression)", elem_classes=["primary-btn"])

            gr.Markdown("<div class='section-title'>Results (exec view)</div>")
            with gr.Group(elem_classes=["card"]):
                gr.Markdown(
                    """
<div class="kpi-grid">
  <div class="kpi"><div class="label">Model</div><div class="value" id="kpi_model">—</div></div>
  <div class="kpi"><div class="label">AUC</div><div class="value" id="kpi_auc">—</div></div>
  <div class="kpi"><div class="label">Accuracy</div><div class="value" id="kpi_acc">—</div></div>
  <div class="kpi"><div class="label">Rows / Features</div><div class="value" id="kpi_rf">—</div></div>
</div>
"""
                )

                # KPI values as hidden textboxes that we also show clearly below (Gradio can't bind HTML IDs directly)
                kpi_model = gr.Textbox(label="Model", value="—", interactive=False)
                kpi_auc = gr.Textbox(label="AUC", value="—", interactive=False)
                kpi_acc = gr.Textbox(label="Accuracy", value="—", interactive=False)
                kpi_rf = gr.Textbox(label="Rows / Features", value="—", interactive=False)

                gr.Markdown("<div class='section-title'>Coefficients (top drivers)</div>")
                coef_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True)

                gr.Markdown("<div class='section-title'>ROC curve</div>")
                roc_img = gr.Image(value=None, label="ROC Curve", type="filepath")

                caveats = gr.Textbox(label="Caveats (auto-detected)", value="", lines=3, interactive=False)

        # RIGHT: Analyst Commentary
        with gr.Column(scale=1):
            gr.Markdown("<div class='section-title'>AI Analyst Commentary</div>")
            with gr.Group(elem_classes=["card"]):
                gr.Markdown(
                    "<div class='small-muted'>Locked scope: explains the <b>current dataset</b> + <b>current model outputs</b> and caveats — not a general chatbot.</div>"
                )

                with gr.Row():
                    btn_exec = gr.Button("Generate executive briefing", elem_classes=["secondary-btn"])
                    btn_risks = gr.Button("Risks / caveats", elem_classes=["secondary-btn"])
                    btn_improve = gr.Button("What should we improve next?", elem_classes=["secondary-btn"])

                user_context = gr.Textbox(
                    label="Context (optional)",
                    placeholder="Example: I’m learning what each header means; I just trained a baseline model; explain results simply for an exec audience.",
                    lines=3
                )

                chatbot = gr.Chatbot(label="Commentary", height=420)

                explain_btn = gr.Button("Explain current results", elem_classes=["primary-btn"])
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="e.g., What does AUC mean? Why is troponin a top driver?",
                    )
                with gr.Row():
                    send = gr.Button("Send", elem_classes=["primary-btn"])
                    clear = gr.Button("Clear commentary", elem_classes=["secondary-btn"])

    gr.Markdown(
        "<div class='small-muted' style='margin-top:10px;'>"
        "<b>Safety:</b> Synthetic demo only. No patient-identifiable input. Not clinical advice. Designed for learning and analytics capability uplift."
        "</div>"
    )

    # ---- Wiring: Load CSV
    def _on_csv_upload(file_obj):
        if file_obj is None:
            df = pd.DataFrame()
            summary, stats = "No CSV loaded.", pd.DataFrame()
            packed = build_llm_context(df, summary, stats, snapshot=None)
            return df, pd.DataFrame(), summary, pd.DataFrame(), "Ready.", "", packed

        # Gradio File gives dict-like or path depending on version; handle both
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

        if not path:
            df = pd.DataFrame()
            summary, stats = "Could not read uploaded file path.", pd.DataFrame()
            packed = build_llm_context(df, summary, stats, snapshot=None)
            return df, pd.DataFrame(), summary, pd.DataFrame(), "Upload failed.", "", packed

        try:
            df = pd.read_csv(path)
        except Exception as e:
            df = pd.DataFrame()
            summary, stats = f"Failed to read CSV: {e}", pd.DataFrame()
            packed = build_llm_context(df, summary, stats, snapshot=None)
            return df, pd.DataFrame(), summary, pd.DataFrame(), "CSV read error.", "", packed

        preview = df.head(25)
        summary, stats = dataset_summary(df, target_col="event_30d")
        status_txt = f"Loaded: {os.path.basename(path)} (rows={len(df)}, cols={df.shape[1]})"
        packed = build_llm_context(df, summary, stats, snapshot=None)

        return df, preview, summary, stats, status_txt, "", packed

    csv_in.change(
        _on_csv_upload,
        inputs=[csv_in],
        outputs=[df_state, preview_df, summary_text, stats_df, status, caveats, llm_context_state],
    )

    # ---- Wiring: Generate dataset
    def _on_generate(n_rows, base_event_rate, overlap, label_flip, missingness, seed):
        status_txt, out_path, preview, summary, stats = do_generate_dataset(
            int(n_rows), float(base_event_rate), float(overlap), float(label_flip), float(missingness), int(seed)
        )
        # Also load full df into state
        try:
            df = pd.read_csv(out_path) if out_path else pd.DataFrame()
        except Exception:
            df = pd.DataFrame()

        packed = build_llm_context(df, summary, stats, snapshot=None)
        return df, out_path, preview, summary, stats, status_txt, "", packed

    gen_btn.click(
        _on_generate,
        inputs=[gen_rows, gen_event, gen_overlap, gen_flip, gen_missing, gen_seed],
        outputs=[df_state, csv_generated, preview_df, summary_text, stats_df, status, caveats, llm_context_state]
    )

    # ---- Wiring: Train model
    def _on_train(df_full, target_col, feats, summary_txt, stats_frame):
        if df_full is None or (isinstance(df_full, pd.DataFrame) and df_full.empty):
            snap = None
            packed = build_llm_context(pd.DataFrame(), summary_txt or "No dataset loaded.", stats_frame, snapshot=None)
            return (
                "No dataset", "n/a", "n/a", "n/a",
                pd.DataFrame(), None, "", {}, packed
            )

        k_model, k_auc, k_acc, coef_table, roc_path, state = do_train_model(df_full, target_col, feats or [])

        rows = len(df_full)
        feat_count = len([f for f in (feats or []) if f in df_full.columns])
        rf = f"{rows} / {feat_count}"

        # Caveats text from state
        notes = state.get("notes", []) if isinstance(state, dict) else []
        cave = "\n".join([f"- {n}" for n in notes]) if notes else ""

        # Build ModelSnapshot-like object for context
        snap_obj = None
        try:
            top_coef_df = pd.DataFrame(state.get("top_coef", []))
            snap_obj = ModelSnapshot(
                target=state.get("target", target_col),
                features=state.get("features", feats or []),
                rows=int(state.get("rows", rows)),
                auc=_safe_float(state.get("auc", None)),
                acc=_safe_float(state.get("acc", None)),
                top_coef=top_coef_df,
                roc_path=roc_path,
                notes=notes
            )
        except Exception:
            snap_obj = None

        packed = build_llm_context(df_full, summary_txt or "", stats_frame, snapshot=snap_obj)

        return (
            k_model, k_auc, k_acc, rf,
            coef_table, roc_path, cave, state, packed
        )

    train_btn.click(
        _on_train,
        inputs=[df_state, target, features, summary_text, stats_df],
        outputs=[kpi_model, kpi_auc, kpi_acc, kpi_rf, coef_df, roc_img, caveats, model_state, llm_context_state]
    )

    # ---- Wiring: Chat + Buttons
    send.click(chat, inputs=[msg, user_context, chatbot, llm_context_state], outputs=[msg, chatbot])
    msg.submit(chat, inputs=[msg, user_context, chatbot, llm_context_state], outputs=[msg, chatbot])
    clear.click(clear_chat, inputs=None, outputs=chatbot)

    explain_btn.click(explain_current_results, inputs=[chatbot, llm_context_state], outputs=chatbot)
    btn_exec.click(exec_brief, inputs=[chatbot, llm_context_state], outputs=chatbot)
    btn_risks.click(risks_caveats, inputs=[chatbot, llm_context_state], outputs=chatbot)
    btn_improve.click(improve_next, inputs=[chatbot, llm_context_state], outputs=chatbot)


if __name__ == "__main__":
    demo.launch()
