import os
import re
import json
import math
import time
import uuid
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr

from openai import OpenAI

# Optional ML/plot deps (we handle cleanly if missing)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        roc_auc_score,
        accuracy_score,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
        brier_score_loss,
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False

# Voice (Edge TTS)
try:
    import edge_tts
    EDGE_TTS_OK = True
except Exception:
    EDGE_TTS_OK = False


# =============================================================================
# CONFIG
# =============================================================================
APP_TITLE = "Clinical ML Tutor"

MODEL_NAME_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # cheaper default; override via env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT_LOCKED = (
    "You are the Clinical ML Tutor.\n"
    "You are NOT a general chatbot.\n"
    "Your job: explain ONLY the CURRENT synthetic dataset + CURRENT model outputs.\n"
    "Style: plain English, short sections, bullet points, executive-friendly.\n"
    "If you use a technical term, define it immediately.\n"
    "Always include: (1) what we trained, (2) what the metrics mean, (3) top drivers direction,\n"
    "(4) caveats (synthetic, leakage, perfect-metric suspicion), (5) next improvement steps.\n"
    "Never give clinical advice. Assume synthetic demo only.\n"
)

# A separate voice prompt that forbids symbols/markdown entirely
VOICE_PROMPT_LOCKED = (
    "You are a voice narrator for an NHS-style demo.\n"
    "Convert the provided analysis into natural spoken English.\n"
    "Rules:\n"
    "- Do NOT read punctuation like backslashes, asterisks, hashes, braces, brackets, underscores.\n"
    "- Do NOT output markdown, bullet symbols, tables, code, JSON, or headings with #.\n"
    "- Expand abbreviations the FIRST time you say them: "
    "AUC = area under the curve; ROC = receiver operating characteristic; "
    "FN = false negative; FP = false positive; TP = true positive; TN = true negative; "
    "ECE = expected calibration error.\n"
    "- Keep it concise: 60 to 120 seconds of speech.\n"
    "- Keep it calm, executive-friendly.\n"
)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =============================================================================
# UTIL: OpenAI response extraction
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
        return "Type a question (e.g., “What does area under the curve mean?”)."

    messages = [{"role": "system", "content": SYSTEM_PROMPT_LOCKED}]

    if context and context.strip():
        messages.append({"role": "user", "content": "CONTEXT (current run):\n" + context.strip()})

    history_messages = history_messages or []
    if len(history_messages) > 8:
        history_messages = history_messages[-8:]

    for m in history_messages:
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
# VOICE SAFE TEXT (no symbols / no markdown / no code)
# =============================================================================
_ABBR_MAP = {
    "AUC": "area under the curve",
    "ROC": "receiver operating characteristic",
    "ECE": "expected calibration error",
    "FN": "false negative",
    "FP": "false positive",
    "TP": "true positive",
    "TN": "true negative",
    "TNR": "true negative rate",
    "TPR": "true positive rate",
    "FPR": "false positive rate",
    "MAE": "mean absolute error",
    "RMSE": "root mean squared error",
}

def _expand_abbr_once(text: str) -> str:
    # Expand FIRST occurrence only: "AUC (area under the curve)" style
    for abbr, full in _ABBR_MAP.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        m = re.search(pattern, text)
        if m:
            text = re.sub(pattern, f"{abbr} ({full})", text, count=1)
    return text


def strip_markdown_for_tts(text: str) -> str:
    """
    Aggressive sanitizer so TTS won't speak "backslash backslash" "asterisk asterisk" "hash".
    Keeps sentences, removes code/markdown/json-ish noise.
    """
    if not text:
        return ""

    t = str(text)

    # Remove the footer line used in the UI
    t = re.sub(r"\n—\n.*$", "", t, flags=re.DOTALL)

    # Remove fenced code blocks
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)

    # Remove inline code
    t = re.sub(r"`[^`]*`", " ", t)

    # Remove markdown links [text](url) -> text
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)

    # Remove headings like ### Title
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.MULTILINE)

    # Remove bullet markers but keep text
    t = re.sub(r"^\s*[-•*+]\s+", "", t, flags=re.MULTILINE)

    # Remove table pipes and separator rows
    t = re.sub(r"\|", " ", t)
    t = re.sub(r"^\s*:?-{2,}:?\s*$", " ", t, flags=re.MULTILINE)

    # Remove JSON-ish braces/brackets that cause "brace brace"
    t = t.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")

    # Kill backslashes/underscores/asterisks/hashes
    t = t.replace("\\", " ").replace("_", " ").replace("*", " ").replace("#", " ")

    # Replace repeated weird tokens (NaN, None, etc.) with spoken-friendly words
    t = re.sub(r"\bNaN\b", "not available", t, flags=re.IGNORECASE)
    t = re.sub(r"\bn/?a\b", "not available", t, flags=re.IGNORECASE)

    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Expand common abbreviations once
    t = _expand_abbr_once(t)

    return t


def make_voice_summary(answer_text: str, llm_context: str) -> str:
    """
    Preferred path: ask model to rewrite for speech.
    Fallback: sanitizer.
    """
    # If no API key, fallback immediately
    if not client:
        cleaned = strip_markdown_for_tts(answer_text)
        if len(cleaned) > 12000:
            cleaned = cleaned[:12000].rsplit(" ", 1)[0] + "…"
        return cleaned

    try:
        model_name = os.getenv("OPENAI_MODEL", MODEL_NAME_DEFAULT)
        # Give the model the answer and (optionally) a tiny bit of context
        # but keep it short so it doesn't start dumping structured stuff.
        user_payload = (
            "Rewrite this for speech.\n\n"
            "TEXT TO NARRATE:\n"
            f"{answer_text}\n"
        )

        resp = client.responses.create(
            model=model_name,
            input=[
                {"role": "system", "content": VOICE_PROMPT_LOCKED},
                {"role": "user", "content": user_payload},
            ],
        )
        spoken = _extract_output_text(resp)
        spoken = strip_markdown_for_tts(spoken)  # still sanitize just in case

        if len(spoken) > 12000:
            spoken = spoken[:12000].rsplit(" ", 1)[0] + "…"
        return spoken or strip_markdown_for_tts(answer_text)
    except Exception:
        cleaned = strip_markdown_for_tts(answer_text)
        if len(cleaned) > 12000:
            cleaned = cleaned[:12000].rsplit(" ", 1)[0] + "…"
        return cleaned


# =============================================================================
# VOICE (Edge TTS)
# =============================================================================
DEFAULT_VOICE = "en-GB-SoniaNeural"
VOICE_CHOICES = [
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-AU-NatashaNeural",
    "en-IN-NeerjaNeural",
]

def _run_async(coro):
    """
    Safe coroutine runner for Gradio callbacks.
    - If no loop is running: asyncio.run
    - If a loop is running (rare in Gradio threads): create a new loop
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _last_assistant_text(history: List[Dict[str, str]]) -> str:
    history = history or []
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "assistant":
            txt = str(m.get("content", "")).strip()
            if txt:
                return txt
    return ""


def speak_text(text: str, voice: str) -> Tuple[Optional[str], str]:
    if not EDGE_TTS_OK:
        return None, "edge-tts not installed. Run:  pip install edge-tts"

    text = (text or "").strip()
    if not text:
        return None, "No voice text available yet — generate an explanation first."

    os.makedirs("reports/tts", exist_ok=True)
    fn = os.path.join("reports", "tts", f"tts_{uuid.uuid4().hex}.mp3")

    v = (voice or DEFAULT_VOICE).strip()
    try:
        _run_async(edge_tts.Communicate(text, v).save(fn))
        return fn, f"Spoken with {v}"
    except Exception as e:
        return None, f"TTS error: {e}"


def speak_last_answer(voice_text: str, history: List[Dict[str, str]], voice: str) -> Tuple[Optional[str], str]:
    """
    Prefer the pre-built voice-safe summary. If missing, sanitize the last assistant message.
    """
    vtxt = (voice_text or "").strip()
    if not vtxt:
        fallback = _last_assistant_text(history)
        vtxt = strip_markdown_for_tts(fallback)

    return speak_text(vtxt, voice)


# =============================================================================
# DATA GENERATION: realistic synthetic cardiac dataset
# =============================================================================
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
    rng = np.random.default_rng(seed)

    n = int(max(50, min(2000, n_rows)))
    base_event_rate = float(np.clip(base_event_rate, 0.05, 0.60))
    overlap = float(np.clip(overlap, 0.10, 1.20))
    label_flip = float(np.clip(label_flip, 0.00, 0.15))
    missingness = float(np.clip(missingness, 0.00, 0.35))

    # Demographics
    age = rng.normal(62, 12, n).clip(18, 95)
    sex = rng.integers(0, 2, n)  # 0/1

    # Vitals
    systolic_bp = (120 + 0.45 * (age - 50) + rng.normal(0, 14, n)).clip(85, 220)
    heart_rate = (72 + 0.18 * (systolic_bp - 120) + rng.normal(0, 10, n)).clip(40, 160)

    # Comorbidities
    diabetes = rng.binomial(1, _sigmoid((age - 55) / 10) * 0.55, n)
    smoking = rng.binomial(1, 0.22 + 0.05 * sex - 0.0008 * (age - 50), n).clip(0, 1)

    # Symptoms/ECG
    chest_pain = rng.binomial(1, _sigmoid((systolic_bp - 135) / 18) * 0.55, n)
    ecg_st = rng.binomial(1, _sigmoid((heart_rate - 85) / 15) * 0.40, n)

    # Labs (skewed, noisy; correlated)
    egfr = (95 - 0.65 * (age - 45) - 8 * diabetes + rng.normal(0, 12, n)).clip(10, 130)
    crp = rng.lognormal(mean=1.2 + 0.25 * diabetes + 0.18 * smoking, sigma=0.55, size=n).clip(0.1, 200)
    ldl = (3.2 + 0.01 * (age - 55) + 0.25 * smoking + rng.normal(0, 0.7, n)).clip(0.8, 8.5)
    ntprobnp = rng.lognormal(
        mean=5.0 + 0.015 * (age - 60) + 0.008 * (90 - egfr),
        sigma=0.7,
        size=n
    ).clip(10, 40000)

    trop_base = rng.lognormal(mean=-4.8, sigma=0.7, size=n)
    trop_spike = rng.lognormal(
        mean=-2.3 + 0.55 * ecg_st + 0.45 * chest_pain + 0.010 * (90 - egfr),
        sigma=0.75,
        size=n
    )
    troponin = (0.6 * trop_base + 0.4 * trop_spike).clip(0.0005, 20)

    # Latent risk model -> event probability
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

    # Calibrate to base_event_rate by shifting logits
    eps = 1e-6
    logits = np.log(np.clip(p_raw, eps, 1 - eps) / np.clip(1 - p_raw, eps, 1 - eps))
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

    # Label noise
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

    # Missingness mainly in labs/vitals
    miss_cols = ["troponin", "ldl", "egfr", "crp", "ntprobnp", "systolic_bp", "heart_rate"]
    if missingness > 0:
        for col in miss_cols:
            mask = rng.random(n) < missingness
            df.loc[mask, col] = np.nan

    return df


# =============================================================================
# SUMMARY STATS (no markdown tables)
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

    top_missing = missing_pct.sort_values(ascending=False).head(5)
    top_missing = top_missing[top_missing > 0]
    if len(top_missing) > 0:
        miss_str = ", ".join([f"{k}={v:.1f}%" for k, v in top_missing.items()])
        lines.append(f"Most missing: {miss_str}")
    else:
        lines.append("Missingness: none detected")

    return "\n".join(lines), desc


# =============================================================================
# METRICS HELPERS
# =============================================================================
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


def _binary_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Optional[float]]:
    sens = tp / (tp + fn) if (tp + fn) > 0 else None  # sensitivity/recall
    spec = tn / (tn + fp) if (tn + fp) > 0 else None  # specificity
    prec = tp / (tp + fp) if (tp + fp) > 0 else None  # precision
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else None
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "accuracy": acc}


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:.1f}%"


# =============================================================================
# PLOTTING: FN-HIGHLIGHTED CONFUSION MATRIX + ROC + CALIBRATION
# =============================================================================
def plot_confusion_matrix_highlight_fn(cm: np.ndarray, threshold: float, title: str = "Confusion Matrix") -> Optional[str]:
    if not MPL_OK:
        return None
    try:
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"cm_{uuid.uuid4().hex[:8]}.png")

        fig = plt.figure(figsize=(5.4, 4.3), dpi=140)
        ax = fig.add_subplot(111)
        ax.imshow(cm)

        ax.set_title(f"{title} (threshold={threshold:.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0 (No event)", "1 (Event)"])
        ax.set_yticklabels(["0 (No event)", "1 (Event)"])

        for (i, j), v in np.ndenumerate(cm):
            label = str(int(v))
            extra = ""
            if i == 1 and j == 0:
                extra = "\n(MISSED EVENT)"
            ax.text(j, i, label + extra, ha="center", va="center", fontsize=10, fontweight="bold")

        fn_i, fn_j = 1, 0
        rect = plt.Rectangle((fn_j - 0.5, fn_i - 0.5), 1, 1, fill=False, linewidth=3)
        ax.add_patch(rect)

        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=1, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        return None


def plot_roc_curve(y_true: np.ndarray, proba: np.ndarray) -> Optional[str]:
    if not MPL_OK:
        return None
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, proba)

        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"roc_{uuid.uuid4().hex[:8]}.png")

        fig = plt.figure(figsize=(5.8, 4.0), dpi=140)
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title("ROC Curve (Classification)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        return None


def plot_calibration_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration (Reliability) Curve"
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not MPL_OK or not SKLEARN_OK:
        return None, None, None
    try:
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        y_true = np.array(y_true).astype(int)

        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=int(n_bins), strategy="quantile")

        brier = None
        try:
            brier = float(brier_score_loss(y_true, proba))
        except Exception:
            brier = None

        ece = None
        try:
            gaps = np.abs(frac_pos - mean_pred)
            ece = float(np.mean(gaps))
        except Exception:
            ece = None

        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"cal_{uuid.uuid4().hex[:8]}.png")

        fig = plt.figure(figsize=(5.8, 4.0), dpi=140)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.plot(mean_pred, frac_pos, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

        return path, brier, ece
    except Exception:
        return None, None, None


def plot_regression_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> Optional[str]:
    if not MPL_OK:
        return None
    try:
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"reg_scatter_{uuid.uuid4().hex[:8]}.png")

        fig = plt.figure(figsize=(5.6, 4.2), dpi=140)
        ax = fig.add_subplot(111)
        ax.scatter(y_true, y_pred, s=18, alpha=0.7)
        mn = float(np.nanmin([y_true.min(), y_pred.min()]))
        mx = float(np.nanmax([y_true.max(), y_pred.max()]))
        ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        return None


def plot_regression_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> Optional[str]:
    if not MPL_OK:
        return None
    try:
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"reg_resid_{uuid.uuid4().hex[:8]}.png")

        resid = y_pred - y_true

        fig = plt.figure(figsize=(5.6, 4.2), dpi=140)
        ax = fig.add_subplot(111)
        ax.scatter(y_true, resid, s=18, alpha=0.7)
        ax.axhline(0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Error (Predicted - Actual)")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        return None


def plot_regression_error_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> Optional[str]:
    if not MPL_OK:
        return None
    try:
        os.makedirs("reports", exist_ok=True)
        path = os.path.join("reports", f"reg_hist_{uuid.uuid4().hex[:8]}.png")

        err = y_pred - y_true

        fig = plt.figure(figsize=(5.6, 4.2), dpi=140)
        ax = fig.add_subplot(111)
        ax.hist(err, bins=25, alpha=0.85)
        ax.axvline(0, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Error (Predicted - Actual)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path
    except Exception:
        return None


# =============================================================================
# MODEL SNAPSHOTS
# =============================================================================
@dataclass
class ClassifSnapshot:
    target: str
    features: List[str]
    rows: int
    auc: Optional[float]
    accuracy_at_threshold: Optional[float]
    threshold: float
    sensitivity: Optional[float]
    specificity: Optional[float]
    precision: Optional[float]
    tp: int
    fp: int
    fn: int
    tn: int
    top_coef: pd.DataFrame
    roc_path: Optional[str]
    cm_path: Optional[str]
    cal_path: Optional[str]
    brier: Optional[float]
    ece: Optional[float]
    fn_table: pd.DataFrame
    notes: List[str]
    y_test: Optional[np.ndarray]
    proba_test: Optional[np.ndarray]
    x_test: Optional[pd.DataFrame]


@dataclass
class RegSnapshot:
    target: str
    features: List[str]
    rows: int
    mae: Optional[float]
    rmse: Optional[float]
    notes: List[str]
    scatter_path: Optional[str]
    resid_path: Optional[str]
    hist_path: Optional[str]


# =============================================================================
# TOP ERRORS: FN panel
# =============================================================================
def build_top_fn_table(
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    max_rows: int = 10
) -> pd.DataFrame:
    try:
        thr = float(np.clip(threshold, 0.01, 0.99))
        pred = (proba >= thr).astype(int)
        mask_fn = (y_test == 1) & (pred == 0)
        if mask_fn.sum() == 0:
            return pd.DataFrame(columns=["predicted_risk", "threshold", "actual_event"] + list(x_test.columns))

        fn_idx = np.where(mask_fn)[0]
        rows = x_test.iloc[fn_idx].copy()
        rows.insert(0, "actual_event", 1)
        rows.insert(0, "threshold", thr)
        rows.insert(0, "predicted_risk", proba[fn_idx])

        rows = rows.sort_values("predicted_risk", ascending=False).head(int(max_rows))
        rows["predicted_risk"] = rows["predicted_risk"].round(4)
        return rows.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["predicted_risk", "threshold", "actual_event"] + list(x_test.columns))


# =============================================================================
# CLASSIFICATION TRAINING
# =============================================================================
def train_baseline_logreg(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
    threshold: float = 0.50,
    cal_bins: int = 10,
    top_fn_n: int = 10
) -> ClassifSnapshot:
    notes: List[str] = []

    if not SKLEARN_OK:
        notes.append("scikit-learn is not installed. Install it: pip install scikit-learn")
        return ClassifSnapshot(
            target=target_col, features=features, rows=0,
            auc=None, accuracy_at_threshold=None, threshold=threshold,
            sensitivity=None, specificity=None, precision=None,
            tp=0, fp=0, fn=0, tn=0,
            top_coef=pd.DataFrame(), roc_path=None, cm_path=None,
            cal_path=None, brier=None, ece=None, fn_table=pd.DataFrame(),
            notes=notes, y_test=None, proba_test=None, x_test=None
        )

    if df is None or df.empty:
        notes.append("No dataset loaded.")
        return ClassifSnapshot(
            target=target_col, features=features, rows=0,
            auc=None, accuracy_at_threshold=None, threshold=threshold,
            sensitivity=None, specificity=None, precision=None,
            tp=0, fp=0, fn=0, tn=0,
            top_coef=pd.DataFrame(), roc_path=None, cm_path=None,
            cal_path=None, brier=None, ece=None, fn_table=pd.DataFrame(),
            notes=notes, y_test=None, proba_test=None, x_test=None
        )

    if target_col not in df.columns:
        notes.append(f"Target column '{target_col}' not found in dataset.")
        return ClassifSnapshot(
            target=target_col, features=features, rows=len(df),
            auc=None, accuracy_at_threshold=None, threshold=threshold,
            sensitivity=None, specificity=None, precision=None,
            tp=0, fp=0, fn=0, tn=0,
            top_coef=pd.DataFrame(), roc_path=None, cm_path=None,
            cal_path=None, brier=None, ece=None, fn_table=pd.DataFrame(),
            notes=notes, y_test=None, proba_test=None, x_test=None
        )

    if not features:
        notes.append("No features selected.")
        return ClassifSnapshot(
            target=target_col, features=features, rows=len(df),
            auc=None, accuracy_at_threshold=None, threshold=threshold,
            sensitivity=None, specificity=None, precision=None,
            tp=0, fp=0, fn=0, tn=0,
            top_coef=pd.DataFrame(), roc_path=None, cm_path=None,
            cal_path=None, brier=None, ece=None, fn_table=pd.DataFrame(),
            notes=notes, y_test=None, proba_test=None, x_test=None
        )

    features = [f for f in features if f in df.columns]
    if not features:
        notes.append("Selected features are not in dataset.")
        return ClassifSnapshot(
            target=target_col, features=features, rows=len(df),
            auc=None, accuracy_at_threshold=None, threshold=threshold,
            sensitivity=None, specificity=None, precision=None,
            tp=0, fp=0, fn=0, tn=0,
            top_coef=pd.DataFrame(), roc_path=None, cm_path=None,
            cal_path=None, brier=None, ece=None, fn_table=pd.DataFrame(),
            notes=notes, y_test=None, proba_test=None, x_test=None
        )

    work = df.copy().dropna(subset=[target_col])
    y = work[target_col].astype(int)

    X = work[features].apply(pd.to_numeric, errors="coerce")
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    strat = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]

    try:
        auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else None
    except Exception:
        auc = None
    auc_f = _safe_float(auc)

    threshold = float(np.clip(threshold, 0.01, 0.99))
    pred = (proba >= threshold).astype(int)

    try:
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    except Exception:
        tn = fp = fn = tp = 0
        cm = np.array([[0, 0], [0, 0]])

    m = _binary_metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)
    acc_thr = m.get("accuracy", None)

    clf = pipe.named_steps["clf"]
    coefs = clf.coef_.reshape(-1)
    coef_df = pd.DataFrame({"feature": features, "coef": coefs})
    coef_df["abs"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs", ascending=False).drop(columns=["abs"])

    top_coef = coef_df.head(12).copy()
    top_coef["direction"] = np.where(top_coef["coef"] >= 0, "↑ increases risk", "↓ decreases risk")
    top_coef = top_coef[["feature", "coef", "direction"]]
    top_coef["coef"] = top_coef["coef"].round(4)

    roc_path = plot_roc_curve(np.array(y_test), np.array(proba)) if (MPL_OK and auc_f is not None) else None
    cm_path = plot_confusion_matrix_highlight_fn(cm, threshold=threshold, title="Confusion Matrix (FN highlighted)") if MPL_OK else None

    cal_path, brier, ece = plot_calibration_curve(
        y_true=np.array(y_test),
        proba=np.array(proba),
        n_bins=int(max(5, min(20, cal_bins))),
        title="Calibration (Reliability) Curve"
    )

    fn_table = build_top_fn_table(
        x_test=X_test,
        y_test=np.array(y_test),
        proba=np.array(proba),
        threshold=threshold,
        max_rows=int(max(5, min(50, top_fn_n)))
    )

    if auc_f is not None and auc_f >= 0.98:
        notes.append(
            "AUC is extremely high. In real-world clinical data this is unusual; it may indicate synthetic rules are too clean, "
            "or a feature leaks the outcome (recorded after the event)."
        )
    if acc_thr is not None and acc_thr >= 0.98:
        notes.append(
            "Accuracy is extremely high. Treat as a demo indicator; check for leakage and increase overlap/noise in synthetic generation."
        )

    if fn > 0:
        notes.append(
            f"Safety note: FN (missed events) at threshold {threshold:.2f} = {fn}. "
            "In cardiac screening, FN is usually the most clinically concerning error type."
        )

    if brier is not None:
        notes.append(f"Calibration note: Brier score = {brier:.4f} (lower is better).")
    if ece is not None:
        notes.append(f"Calibration note: ECE approx = {ece:.4f} (lower is better).")

    return ClassifSnapshot(
        target=target_col,
        features=features,
        rows=len(work),
        auc=auc_f,
        accuracy_at_threshold=_safe_float(acc_thr),
        threshold=threshold,
        sensitivity=_safe_float(m.get("sensitivity", None)),
        specificity=_safe_float(m.get("specificity", None)),
        precision=_safe_float(m.get("precision", None)),
        tp=tp, fp=fp, fn=fn, tn=tn,
        top_coef=top_coef,
        roc_path=roc_path,
        cm_path=cm_path,
        cal_path=cal_path,
        brier=_safe_float(brier),
        ece=_safe_float(ece),
        fn_table=fn_table,
        notes=notes,
        y_test=np.array(y_test),
        proba_test=np.array(proba),
        x_test=X_test.copy()
    )


def recalc_threshold_metrics_and_tables(
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    top_fn_n: int = 10,
    cal_bins: int = 10
) -> Tuple[Dict[str, Any], Optional[str], pd.DataFrame, Optional[str], Optional[float], Optional[float]]:
    threshold = float(np.clip(threshold, 0.01, 0.99))
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    m = _binary_metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)

    cm_path = plot_confusion_matrix_highlight_fn(cm, threshold=threshold, title="Confusion Matrix (FN highlighted)") if MPL_OK else None

    fn_table = build_top_fn_table(
        x_test=x_test,
        y_test=y_test,
        proba=proba,
        threshold=threshold,
        max_rows=int(max(5, min(50, top_fn_n)))
    )

    cal_path, brier, ece = plot_calibration_curve(
        y_true=y_test,
        proba=proba,
        n_bins=int(max(5, min(20, cal_bins))),
        title="Calibration (Reliability) Curve"
    )

    out = {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": _safe_float(m.get("sensitivity", None)),
        "specificity": _safe_float(m.get("specificity", None)),
        "precision": _safe_float(m.get("precision", None)),
        "accuracy": _safe_float(m.get("accuracy", None)),
    }
    return out, cm_path, fn_table, cal_path, _safe_float(brier), _safe_float(ece)


# =============================================================================
# REGRESSION TRAINING (Troponin / NT-proBNP)
# =============================================================================
def _prepare_regression_xy(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    notes = []
    if target_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), [], [f"Target '{target_col}' not found."]

    used_features = [f for f in features if f in df.columns and f != target_col]
    if not used_features:
        return pd.DataFrame(), pd.Series(dtype=float), [], ["No valid features selected (and target removed)."]

    work = df.copy().dropna(subset=[target_col])
    y_raw = pd.to_numeric(work[target_col], errors="coerce")
    work = work.loc[~y_raw.isna()].copy()
    y_raw = y_raw.loc[work.index]

    X = work[used_features].apply(pd.to_numeric, errors="coerce")
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    y_log = np.log1p(y_raw.clip(lower=0))
    notes.append("Target is log-transformed (log1p) to handle heavy skew; metrics shown in original units after inverse transform.")
    return X, y_log, used_features, notes


def train_regression_baseline(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
) -> RegSnapshot:
    notes: List[str] = []

    if not SKLEARN_OK:
        notes.append("scikit-learn is not installed. Install it: pip install scikit-learn")
        return RegSnapshot(target=target_col, features=features, rows=0, mae=None, rmse=None, notes=notes,
                           scatter_path=None, resid_path=None, hist_path=None)

    if df is None or df.empty:
        notes.append("No dataset loaded.")
        return RegSnapshot(target=target_col, features=features, rows=0, mae=None, rmse=None, notes=notes,
                           scatter_path=None, resid_path=None, hist_path=None)

    X, y_log, used_features, prep_notes = _prepare_regression_xy(df, target_col, features)
    notes.extend(prep_notes)

    if X.empty or y_log.empty or not used_features:
        return RegSnapshot(target=target_col, features=used_features, rows=0, mae=None, rmse=None, notes=notes,
                           scatter_path=None, resid_path=None, hist_path=None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.25, random_state=42
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0, random_state=42))
    ])
    pipe.fit(X_train, y_train)

    y_pred_log = pipe.predict(X_test)

    y_test_raw = np.expm1(y_test.to_numpy())
    y_pred_raw = np.expm1(y_pred_log)

    try:
        mae = mean_absolute_error(y_test_raw, y_pred_raw)
    except Exception:
        mae = None
    try:
        rmse = math.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    except Exception:
        rmse = None

    mae_f = _safe_float(mae)
    rmse_f = _safe_float(rmse)

    scatter = plot_regression_pred_vs_actual(y_test_raw, y_pred_raw, title=f"{target_col}: Predicted vs Actual")
    resid = plot_regression_residuals(y_test_raw, y_pred_raw, title=f"{target_col}: Residuals (Error vs Actual)")
    hist = plot_regression_error_hist(y_test_raw, y_pred_raw, title=f"{target_col}: Error Distribution")

    return RegSnapshot(
        target=target_col,
        features=used_features,
        rows=int(len(X)),
        mae=mae_f,
        rmse=rmse_f,
        notes=notes,
        scatter_path=scatter,
        resid_path=resid,
        hist_path=hist
    )


# =============================================================================
# CONTEXT PACKING for the LLM
# =============================================================================
def build_llm_context(
    df: Optional[pd.DataFrame],
    summary_text: str,
    numeric_stats: Optional[pd.DataFrame],
    classif_state: Optional[Dict[str, Any]],
    reg_state: Optional[Dict[str, Any]],
) -> str:
    parts = []
    parts.append("DEMO STANCE: Synthetic data only. Not clinical advice.")
    parts.append("")
    parts.append("DATASET SUMMARY:")
    parts.append(summary_text.strip() if summary_text else "n/a")

    if numeric_stats is not None and not numeric_stats.empty:
        head = numeric_stats.head(10).copy()
        parts.append("")
        parts.append("NUMERIC STATS (first 10 features):")
        parts.append(head.to_json(orient="records"))

    if classif_state and isinstance(classif_state, dict) and classif_state.get("trained"):
        parts.append("")
        parts.append("CLASSIFICATION OUTPUTS (current run):")
        parts.append(f"Target: {classif_state.get('target')}")
        parts.append(f"Rows used: {classif_state.get('rows_used')}")
        parts.append(f"Features: {', '.join(classif_state.get('features', []))}")
        parts.append(f"AUC: {classif_state.get('auc')}")
        parts.append(f"Threshold: {classif_state.get('threshold')}")
        parts.append(f"Accuracy@threshold: {classif_state.get('acc_thr')}")
        parts.append(f"Sensitivity: {classif_state.get('sensitivity')}")
        parts.append(f"Specificity: {classif_state.get('specificity')}")
        parts.append(f"Precision: {classif_state.get('precision')}")
        parts.append(f"Counts: TP={classif_state.get('tp')} FP={classif_state.get('fp')} FN={classif_state.get('fn')} TN={classif_state.get('tn')}")
        parts.append(f"Brier score: {classif_state.get('brier')}")
        parts.append(f"ECE approx: {classif_state.get('ece')}")
        if classif_state.get("top_coef"):
            parts.append("Top coefficients:")
            parts.append(json.dumps(classif_state.get("top_coef"), ensure_ascii=False))
        if classif_state.get("notes"):
            parts.append("Caveats:")
            parts.append(json.dumps(classif_state.get("notes"), ensure_ascii=False))
        if classif_state.get("top_fn_table"):
            parts.append("Top missed events (FN) table:")
            parts.append(json.dumps(classif_state.get("top_fn_table"), ensure_ascii=False))
    else:
        parts.append("")
        parts.append("CLASSIFICATION OUTPUTS: none (model not trained yet).")

    if reg_state and isinstance(reg_state, dict) and reg_state.get("trained"):
        parts.append("")
        parts.append("REGRESSION OUTPUTS (current run):")
        parts.append(f"Target: {reg_state.get('target')}")
        parts.append(f"Rows used: {reg_state.get('rows_used')}")
        parts.append(f"Features: {', '.join(reg_state.get('features', []))}")
        parts.append(f"MAE: {reg_state.get('mae')}")
        parts.append(f"RMSE: {reg_state.get('rmse')}")
        if reg_state.get("notes"):
            parts.append("Caveats:")
            parts.append(json.dumps(reg_state.get("notes"), ensure_ascii=False))
    else:
        parts.append("")
        parts.append("REGRESSION OUTPUTS: none (model not trained yet).")

    return "\n".join(parts).strip()


# =============================================================================
# GRADIO CALLBACKS
# =============================================================================
def _history_append(history: List[Dict[str, str]], role: str, content: str) -> List[Dict[str, str]]:
    history = history or []
    history.append({"role": role, "content": content})
    return history


def chat(user_message: str, user_context: str, history: List[Dict[str, str]], llm_context: str) -> Tuple[str, List[Dict[str, str]], str]:
    history = history or []
    user_message = (user_message or "").strip()
    if not user_message:
        return "", history, ""

    merged_context = (llm_context or "").strip()
    if user_context and user_context.strip():
        merged_context = (merged_context + "\n\nUSER CONTEXT:\n" + user_context.strip()).strip()

    answer = ask_llm_locked(user_message=user_message, context=merged_context, history_messages=history)
    history = _history_append(history, "user", user_message)
    history = _history_append(history, "assistant", answer)

    voice_text = make_voice_summary(answer, llm_context=merged_context)
    return "", history, voice_text


def clear_chat() -> Tuple[List[Dict[str, str]], str]:
    return [], ""


def explain_current_results(history: List[Dict[str, str]], llm_context: str) -> Tuple[List[Dict[str, str]], str]:
    history = history or []
    prompt = (
        "Explain the current results for an executive audience.\n"
        "Use short sections:\n"
        "- What we trained (1–2 lines)\n"
        "- What AUC / Accuracy / Sensitivity / Specificity mean (define each)\n"
        "- What calibration means (define it), include Brier score + ECE approx\n"
        "- Highlight FN (missed events) and what it implies\n"
        "- Top drivers (direction + what it suggests)\n"
        "- Top missed events (FN) table: explain what it shows and why it matters\n"
        "- Risks/caveats (including perfect-metric / leakage suspicion)\n"
        "- Next improvement steps (3 bullets)\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)

    voice_text = make_voice_summary(answer, llm_context=llm_context or "")
    return history, voice_text


def exec_brief(history: List[Dict[str, str]], llm_context: str) -> Tuple[List[Dict[str, str]], str]:
    history = history or []
    prompt = (
        "Write an executive briefing (max ~12 bullets) describing:\n"
        "- What this demo is\n"
        "- Why it matters (capability uplift)\n"
        "- Safety stance (synthetic only, not clinical advice)\n"
        "- What the current outputs indicate (use the numbers from context)\n"
        "- Specifically call out FN (missed events), and mention the 'top FN' panel exists\n"
        "- Mention calibration (Brier/ECE) and what it means\n"
        "- What we should do next\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)

    voice_text = make_voice_summary(answer, llm_context=llm_context or "")
    return history, voice_text


def risks_caveats(history: List[Dict[str, str]], llm_context: str) -> Tuple[List[Dict[str, str]], str]:
    history = history or []
    prompt = (
        "List the main risks / caveats of the current run.\n"
        "Include:\n"
        "- Synthetic data limitations\n"
        "- Leakage / perfect metrics suspicion\n"
        "- FN risk (missed events) and threshold trade-offs\n"
        "- Calibration risk (probabilities may not reflect reality)\n"
        "- Small sample size / representativeness\n"
        "- Model simplicity limitations\n"
        "- Governance note (no patient-identifiable data)\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)

    voice_text = make_voice_summary(answer, llm_context=llm_context or "")
    return history, voice_text


def improve_next(history: List[Dict[str, str]], llm_context: str) -> Tuple[List[Dict[str, str]], str]:
    history = history or []
    prompt = (
        "Recommend next improvements, prioritised.\n"
        "Keep it practical and demo-friendly.\n"
        "Include:\n"
        "- Data realism (noise, missingness, calibration)\n"
        "- Evaluation (cross-validation, calibration curve, threshold selection)\n"
        "- Safety lens: reduce FN where appropriate, explain trade-offs\n"
        "- Error analysis: use the top FN panel to learn patterns\n"
        "- Interpretability (feature scaling note, SHAP optional)\n"
        "- UI/UX (what would make execs instantly understand)\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)

    voice_text = make_voice_summary(answer, llm_context=llm_context or "")
    return history, voice_text


# =============================================================================
# THEME (NHS Blue) + CSS
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

[data-testid="dataframe"] table,
[data-testid="dataframe"] th,
[data-testid="dataframe"] td {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial !important;
  font-size: 12.5px !important;
  line-height: 1.25 !important;
  font-variant-numeric: tabular-nums !important;
}}
[data-testid="dataframe"] td {{
  white-space: nowrap !important;
}}

[data-testid="dataframe"] {{
  max-height: 360px;
  overflow: auto;
  border-radius: 12px;
}}

[data-testid="chatbot"] {{
  min-height: 740px !important;
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
        AI partner for exploring a <b>synthetic</b> cardiac dataset — with explainable modelling + plain-English commentary.
      </div>
      <div class="badge-row">
        <span class="badge"><span class="dot"></span> Synthetic-only guidance</span>
        <span class="badge"><span class="dot"></span> Plain-English explanations</span>
        <span class="badge"><span class="dot"></span> FN (missed events) highlighted</span>
        <span class="badge"><span class="dot"></span> Calibration + Top FN panel</span>
        <span class="badge"><span class="dot"></span> Voice is <b>speech-safe</b> (no code/no symbols)</span>
      </div>
    </div>
    <div style="text-align:right; color:{MUTED}; font-size:12px;">
      Demo stance<br/>
      <b>Learning & capability uplift</b>
    </div>
  </div>
</div>
"""
    )

    # App state
    df_state = gr.State(pd.DataFrame())
    classif_state = gr.State({})
    reg_state = gr.State({})
    llm_context_state = gr.State("")
    voice_text_state = gr.State("")  # NEW: the thing we speak

    with gr.Row():
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

            with gr.Tabs():
                with gr.Tab("Classification • Event risk"):
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

                        threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Decision threshold (lower = fewer FN, more FP)")
                        cal_bins = gr.Slider(5, 20, value=10, step=1, label="Calibration bins (reliability curve)")
                        top_fn_n = gr.Slider(5, 50, value=10, step=1, label="Top FN rows to display (worst misses)")
                        train_btn = gr.Button("Train baseline model (Logistic Regression)", elem_classes=["primary-btn"])

                    gr.Markdown("<div class='section-title'>Results (exec view)</div>")
                    with gr.Group(elem_classes=["card"]):
                        kpi_model = gr.Textbox(label="Model", value="—", interactive=False)
                        kpi_auc = gr.Textbox(label="AUC", value="—", interactive=False)
                        kpi_acc = gr.Textbox(label="Accuracy @ threshold", value="—", interactive=False)
                        kpi_sens = gr.Textbox(label="Sensitivity (catch events)", value="—", interactive=False)
                        kpi_spec = gr.Textbox(label="Specificity (dismiss non-events)", value="—", interactive=False)
                        kpi_fn = gr.Textbox(label="Missed events (FN)", value="—", interactive=False)
                        kpi_rf = gr.Textbox(label="Rows / Features", value="—", interactive=False)

                        gr.Markdown("<div class='section-title'>Calibration (probabilities you can trust?)</div>")
                        with gr.Row():
                            cal_img = gr.Image(value=None, label="Reliability curve", type="filepath")
                            with gr.Group():
                                kpi_brier = gr.Textbox(label="Brier score (lower is better)", value="—", interactive=False)
                                kpi_ece = gr.Textbox(label="ECE approx (lower is better)", value="—", interactive=False)

                        gr.Markdown("<div class='section-title'>Coefficients (top drivers)</div>")
                        coef_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True)

                        with gr.Row():
                            roc_img = gr.Image(value=None, label="ROC Curve", type="filepath")
                            cm_img = gr.Image(value=None, label="Confusion Matrix (FN highlighted)", type="filepath")

                        fn_sentence = gr.Textbox(label="Plain-English FN statement", value="—", interactive=False)
                        caveats = gr.Textbox(label="Caveats (auto-detected)", value="", lines=5, interactive=False)

                        gr.Markdown("<div class='section-title'>Top missed events (FN) — worst 10</div>")
                        gr.Markdown(
                            "<div class='small-muted'>These are the <b>false negatives</b> (actual event=1, predicted 0). "
                            "Sorted by predicted risk so you can see the most dangerous near-misses.</div>"
                        )
                        fn_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True)

                with gr.Tab("Regression • Lab estimation"):
                    gr.Markdown("<div class='section-title'>Regression targets (predict a number)</div>")
                    with gr.Group(elem_classes=["card"]):
                        reg_target = gr.Dropdown(
                            label="Regression target",
                            choices=["troponin", "ntprobnp"],
                            value="troponin",
                            allow_custom_value=False
                        )
                        reg_features = gr.Dropdown(
                            label="Features (numeric) — target is auto-removed to prevent leakage",
                            multiselect=True,
                            choices=["age", "sex", "systolic_bp", "heart_rate", "troponin", "ldl", "egfr", "crp",
                                     "chest_pain", "ecg_st", "diabetes", "smoking", "ntprobnp", "event_30d"],
                            value=["age", "sex", "systolic_bp", "heart_rate", "ldl", "egfr", "crp",
                                   "chest_pain", "ecg_st", "diabetes", "smoking", "event_30d"]
                        )
                        reg_train_btn = gr.Button("Train baseline regression (Ridge)", elem_classes=["primary-btn"])

                    gr.Markdown("<div class='section-title'>Regression results</div>")
                    with gr.Group(elem_classes=["card"]):
                        reg_kpi_target = gr.Textbox(label="Target", value="—", interactive=False)
                        reg_kpi_mae = gr.Textbox(label="MAE (typical absolute error)", value="—", interactive=False)
                        reg_kpi_rmse = gr.Textbox(label="RMSE (penalises big misses)", value="—", interactive=False)
                        reg_kpi_rf = gr.Textbox(label="Rows / Features", value="—", interactive=False)
                        reg_notes = gr.Textbox(label="Caveats / notes", value="", lines=3, interactive=False)

                        with gr.Row():
                            reg_scatter_img = gr.Image(value=None, label="Predicted vs Actual", type="filepath")
                            reg_resid_img = gr.Image(value=None, label="Residuals (Error vs Actual)", type="filepath")
                        reg_hist_img = gr.Image(value=None, label="Error distribution", type="filepath")

        with gr.Column(scale=1.15):
            gr.Markdown("<div class='section-title'>AI Analyst Commentary</div>")
            with gr.Group(elem_classes=["card"]):
                gr.Markdown(
                    "<div class='small-muted'>Locked scope: explains the <b>current dataset</b> + <b>current model outputs</b> and caveats.</div>"
                )

                # Voice panel
                gr.Markdown("<div class='section-title'>Voice (TTS)</div>")
                with gr.Row():
                    voice_choice = gr.Dropdown(
                        label="Voice",
                        choices=VOICE_CHOICES,
                        value=DEFAULT_VOICE,
                        interactive=True
                    )
                    speak_btn = gr.Button("🔊 Speak executive voice summary", elem_classes=["secondary-btn"])
                tts_audio = gr.Audio(label="Voice output", type="filepath")
                tts_status = gr.Textbox(
                    label="TTS status",
                    value=("edge-tts ready" if EDGE_TTS_OK else "Install edge-tts: pip install edge-tts"),
                    interactive=False
                )

                with gr.Row():
                    btn_exec = gr.Button("Generate executive briefing", elem_classes=["secondary-btn"])
                    btn_risks = gr.Button("Risks / caveats", elem_classes=["secondary-btn"])
                    btn_improve = gr.Button("What should we improve next?", elem_classes=["secondary-btn"])

                user_context = gr.Textbox(
                    label="Context (optional)",
                    placeholder="Example: Explain results simply for an exec audience; focus on FN safety risk, calibration, and threshold trade-offs.",
                    lines=3
                )

                chatbot = gr.Chatbot(label="Commentary", height=740)

                explain_btn = gr.Button("Explain current results", elem_classes=["primary-btn"])
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your question",
                        placeholder="e.g., Why are false negatives dangerous? What does calibration mean? Why log-transform troponin?",
                    )
                with gr.Row():
                    send = gr.Button("Send", elem_classes=["primary-btn"])
                    clear = gr.Button("Clear commentary", elem_classes=["secondary-btn"])

    gr.Markdown(
        "<div class='small-muted' style='margin-top:10px;'>"
        "<b>Safety:</b> Synthetic demo only. No patient-identifiable input."
        "</div>"
    )

    # ---- CSV upload handler
    def _on_csv_upload(file_obj):
        if file_obj is None:
            df = pd.DataFrame()
            summary, stats = "No CSV loaded.", pd.DataFrame()
            packed = build_llm_context(df, summary, stats, classif_state={}, reg_state={})
            return df, pd.DataFrame(), summary, stats, "Ready.", {}, {}, packed

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
            packed = build_llm_context(df, summary, stats, classif_state={}, reg_state={})
            return df, pd.DataFrame(), summary, stats, "Upload failed.", {}, {}, packed

        try:
            df = pd.read_csv(path)
        except Exception as e:
            df = pd.DataFrame()
            summary, stats = f"Failed to read CSV: {e}", pd.DataFrame()
            packed = build_llm_context(df, summary, stats, classif_state={}, reg_state={})
            return df, pd.DataFrame(), summary, stats, "CSV read error.", {}, {}, packed

        preview = df.head(25)
        summary, stats = dataset_summary(df, target_col="event_30d")
        status_txt = f"Loaded: {os.path.basename(path)} (rows={len(df)}, cols={df.shape[1]})"
        packed = build_llm_context(df, summary, stats, classif_state={}, reg_state={})
        return df, preview, summary, stats, status_txt, {}, {}, packed

    csv_in.change(
        _on_csv_upload,
        inputs=[csv_in],
        outputs=[df_state, preview_df, summary_text, stats_df, status, classif_state, reg_state, llm_context_state],
    )

    # ---- Generate dataset handler
    def _on_generate(n_rows, base_event_rate, overlap, label_flip, missingness, seed):
        try:
            df = generate_realistic_synthetic(
                n_rows=int(n_rows),
                base_event_rate=float(base_event_rate),
                overlap=float(overlap),
                label_flip=float(label_flip),
                missingness=float(missingness),
                seed=int(seed),
            )
            os.makedirs("reports", exist_ok=True)
            out_path = os.path.join("reports", f"synthetic_cardiac_realistic_{uuid.uuid4().hex[:8]}.csv")
            df.to_csv(out_path, index=False)

            summary, stats = dataset_summary(df, target_col="event_30d")
            status_txt = f"Generated dataset: {os.path.basename(out_path)} (rows={len(df)}, cols={df.shape[1]})"
            packed = build_llm_context(df, summary, stats, classif_state={}, reg_state={})
            return df, out_path, df.head(25), summary, stats, status_txt, {}, {}, packed
        except Exception as e:
            df = pd.DataFrame()
            packed = build_llm_context(df, "n/a", pd.DataFrame(), classif_state={}, reg_state={})
            return df, "", pd.DataFrame(), f"Dataset generation failed: {e}", pd.DataFrame(), "Error.", {}, {}, packed

    gen_btn.click(
        _on_generate,
        inputs=[gen_rows, gen_event, gen_overlap, gen_flip, gen_missing, gen_seed],
        outputs=[df_state, csv_generated, preview_df, summary_text, stats_df, status, classif_state, reg_state, llm_context_state]
    )

    # ---- Train classification handler
    def _on_train_classif(df_full, target_col, feats, thr, bins, topn, summary_txt, stats_frame, reg_state_dict):
        if df_full is None or (isinstance(df_full, pd.DataFrame) and df_full.empty):
            packed = build_llm_context(pd.DataFrame(), summary_txt or "No dataset loaded.", stats_frame, classif_state={}, reg_state=reg_state_dict or {})
            return (
                "No dataset", "n/a", "n/a", "n/a", "n/a", "n/a", "—",
                None, "—", "—",
                pd.DataFrame(), None, None, None, "—", "", pd.DataFrame(), {}, packed
            )

        snap = train_baseline_logreg(
            df_full,
            target_col,
            feats or [],
            threshold=float(thr),
            cal_bins=int(bins),
            top_fn_n=int(topn),
        )

        rows = snap.rows
        feat_count = len([f for f in (snap.features or []) if f in df_full.columns])
        rf = f"{rows} / {feat_count}"

        k_model = "Logistic Regression (baseline)"
        k_auc = f"{snap.auc:.3f}" if snap.auc is not None else "n/a"
        k_acc = _format_pct(snap.accuracy_at_threshold)
        k_sens = _format_pct(snap.sensitivity)
        k_spec = _format_pct(snap.specificity)
        k_fn = str(snap.fn)

        brier_txt = f"{snap.brier:.4f}" if snap.brier is not None else "n/a"
        ece_txt = f"{snap.ece:.4f}" if snap.ece is not None else "n/a"

        true_events = snap.tp + snap.fn
        fn_sentence_txt = f"At threshold {snap.threshold:.2f}, the model missed {snap.fn} out of {true_events} true events." if true_events > 0 else "—"
        cave = "\n".join([f"- {n}" for n in (snap.notes or [])]) if (snap.notes) else ""

        state = {
            "trained": True,
            "target": snap.target,
            "features": snap.features,
            "rows_used": snap.rows,
            "auc": snap.auc,
            "threshold": snap.threshold,
            "acc_thr": snap.accuracy_at_threshold,
            "sensitivity": snap.sensitivity,
            "specificity": snap.specificity,
            "precision": snap.precision,
            "tp": snap.tp, "fp": snap.fp, "fn": snap.fn, "tn": snap.tn,
            "brier": snap.brier,
            "ece": snap.ece,
            "cal_bins": int(bins),
            "top_fn_n": int(topn),
            "notes": snap.notes or [],
            "top_coef": snap.top_coef.to_dict(orient="records") if isinstance(snap.top_coef, pd.DataFrame) else [],
            "top_fn_table": snap.fn_table.to_dict(orient="records") if isinstance(snap.fn_table, pd.DataFrame) else [],
            "y_test": snap.y_test.tolist() if snap.y_test is not None else [],
            "proba_test": snap.proba_test.tolist() if snap.proba_test is not None else [],
            "x_test": snap.x_test.to_dict(orient="list") if isinstance(snap.x_test, pd.DataFrame) else {},
            "roc_path": snap.roc_path,
            "cm_path": snap.cm_path,
            "cal_path": snap.cal_path,
        }

        packed = build_llm_context(df_full, summary_txt or "", stats_frame, classif_state=state, reg_state=reg_state_dict or {})

        return (
            k_model, k_auc, k_acc, k_sens, k_spec, k_fn, rf,
            snap.cal_path, brier_txt, ece_txt,
            snap.top_coef, snap.roc_path, snap.cm_path, fn_sentence_txt, cave, snap.fn_table, state, packed
        )

    train_btn.click(
        _on_train_classif,
        inputs=[df_state, target, features, threshold, cal_bins, top_fn_n, summary_text, stats_df, reg_state],
        outputs=[kpi_model, kpi_auc, kpi_acc, kpi_sens, kpi_spec, kpi_fn, kpi_rf,
                 cal_img, kpi_brier, kpi_ece,
                 coef_df, roc_img, cm_img, fn_sentence, caveats, fn_df, classif_state, llm_context_state]
    )

    # ---- Threshold slider live update (no retrain)
    def _on_threshold_change(thr, state_dict, summary_txt, stats_frame, df_full, reg_state_dict):
        if not isinstance(state_dict, dict) or not state_dict.get("trained"):
            packed = build_llm_context(df_full, summary_txt or "", stats_frame, classif_state=state_dict or {}, reg_state=reg_state_dict or {})
            return "—", "—", "—", "—", None, "—", pd.DataFrame(), "—", "—", None, state_dict, packed

        y_test = np.array(state_dict.get("y_test", []), dtype=int)
        proba = np.array(state_dict.get("proba_test", []), dtype=float)

        x_test_dict = state_dict.get("x_test", {}) or {}
        try:
            x_test = pd.DataFrame(x_test_dict)
        except Exception:
            x_test = pd.DataFrame()

        if y_test.size == 0 or proba.size == 0 or x_test.empty:
            packed = build_llm_context(df_full, summary_txt or "", stats_frame, classif_state=state_dict, reg_state=reg_state_dict or {})
            return "—", "—", "—", "—", None, "—", pd.DataFrame(), "—", "—", None, state_dict, packed

        out, cm_path, fn_table, cal_path, brier, ece = recalc_threshold_metrics_and_tables(
            x_test=x_test,
            y_test=y_test,
            proba=proba,
            threshold=float(thr),
            top_fn_n=int(state_dict.get("top_fn_n", 10)),
            cal_bins=int(state_dict.get("cal_bins", 10)),
        )

        state_dict["threshold"] = out["threshold"]
        state_dict["tp"] = out["tp"]
        state_dict["fp"] = out["fp"]
        state_dict["fn"] = out["fn"]
        state_dict["tn"] = out["tn"]
        state_dict["sensitivity"] = out["sensitivity"]
        state_dict["specificity"] = out["specificity"]
        state_dict["precision"] = out["precision"]
        state_dict["acc_thr"] = out["accuracy"]
        state_dict["cm_path"] = cm_path
        state_dict["cal_path"] = cal_path
        state_dict["brier"] = brier
        state_dict["ece"] = ece
        state_dict["top_fn_table"] = fn_table.to_dict(orient="records") if isinstance(fn_table, pd.DataFrame) else []

        k_acc = _format_pct(out["accuracy"])
        k_sens = _format_pct(out["sensitivity"])
        k_spec = _format_pct(out["specificity"])
        k_fn = str(out["fn"])

        true_events = out["tp"] + out["fn"]
        fn_sentence_txt = f"At threshold {out['threshold']:.2f}, the model missed {out['fn']} out of {true_events} true events." if true_events > 0 else "—"

        brier_txt = f"{brier:.4f}" if brier is not None else "n/a"
        ece_txt = f"{ece:.4f}" if ece is not None else "n/a"

        packed = build_llm_context(df_full, summary_txt or "", stats_frame, classif_state=state_dict, reg_state=reg_state_dict or {})

        return k_acc, k_sens, k_spec, k_fn, cm_path, fn_sentence_txt, fn_table, brier_txt, ece_txt, cal_path, state_dict, packed

    threshold.change(
        _on_threshold_change,
        inputs=[threshold, classif_state, summary_text, stats_df, df_state, reg_state],
        outputs=[kpi_acc, kpi_sens, kpi_spec, kpi_fn, cm_img, fn_sentence, fn_df, kpi_brier, kpi_ece, cal_img, classif_state, llm_context_state]
    )

    # ---- Train regression handler
    def _on_train_reg(df_full, tgt, feats, summary_txt, stats_frame, classif_state_dict):
        if df_full is None or (isinstance(df_full, pd.DataFrame) and df_full.empty):
            packed = build_llm_context(pd.DataFrame(), summary_txt or "No dataset loaded.", stats_frame, classif_state=classif_state_dict or {}, reg_state={})
            return "—", "n/a", "n/a", "—", "", None, None, None, {}, packed

        snap = train_regression_baseline(df_full, str(tgt), feats or [])

        rows = snap.rows
        feat_count = len(snap.features or [])
        rf = f"{rows} / {feat_count}"

        k_tgt = snap.target
        k_mae = f"{snap.mae:.4g}" if snap.mae is not None else "n/a"
        k_rmse = f"{snap.rmse:.4g}" if snap.rmse is not None else "n/a"
        notes = "\n".join([f"- {n}" for n in (snap.notes or [])])

        state = {
            "trained": True,
            "target": snap.target,
            "features": snap.features,
            "rows_used": snap.rows,
            "mae": snap.mae,
            "rmse": snap.rmse,
            "notes": snap.notes or [],
        }

        packed = build_llm_context(df_full, summary_txt or "", stats_frame, classif_state=classif_state_dict or {}, reg_state=state)

        return k_tgt, k_mae, k_rmse, rf, notes, snap.scatter_path, snap.resid_path, snap.hist_path, state, packed

    reg_train_btn.click(
        _on_train_reg,
        inputs=[df_state, reg_target, reg_features, summary_text, stats_df, classif_state],
        outputs=[reg_kpi_target, reg_kpi_mae, reg_kpi_rmse, reg_kpi_rf, reg_notes,
                 reg_scatter_img, reg_resid_img, reg_hist_img, reg_state, llm_context_state]
    )

    # ---- Wiring: Chat + Buttons (now also update voice_text_state)
    send.click(chat, inputs=[msg, user_context, chatbot, llm_context_state], outputs=[msg, chatbot, voice_text_state])
    msg.submit(chat, inputs=[msg, user_context, chatbot, llm_context_state], outputs=[msg, chatbot, voice_text_state])

    clear.click(clear_chat, inputs=None, outputs=[chatbot, voice_text_state])

    explain_btn.click(explain_current_results, inputs=[chatbot, llm_context_state], outputs=[chatbot, voice_text_state])
    btn_exec.click(exec_brief, inputs=[chatbot, llm_context_state], outputs=[chatbot, voice_text_state])
    btn_risks.click(risks_caveats, inputs=[chatbot, llm_context_state], outputs=[chatbot, voice_text_state])
    btn_improve.click(improve_next, inputs=[chatbot, llm_context_state], outputs=[chatbot, voice_text_state])

    # ---- Wiring: Voice
    speak_btn.click(
        speak_last_answer,
        inputs=[voice_text_state, chatbot, voice_choice],
        outputs=[tts_audio, tts_status]
    )


if __name__ == "__main__":
    demo.launch()
