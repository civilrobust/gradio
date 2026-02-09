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

# OpenAI SDK is optional; the app should still run without it.
try:
    from openai import OpenAI  # type: ignore
    OPENAI_SDK_OK = True
except Exception:
    OpenAI = None  # type: ignore
    OPENAI_SDK_OK = False

# Optional ML/plot deps (we handle cleanly if missing)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        roc_auc_score,
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
APP_TITLE = "Decision Support Tutor (ICT-first)"

MODEL_NAME_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT_LOCKED = (
    "You are the Decision Support Tutor.\n"
    "You are NOT a general chatbot.\n"
    "Your job: explain ONLY the CURRENT dataset + CURRENT evaluation outputs.\n"
    "Style: plain English, short sections, bullet points, executive-friendly.\n"
    "If you use a technical term, define it immediately.\n"
    "Always include: (1) what we evaluated or trained, (2) what the metrics mean,\n"
    "(3) FN vs FP trade-off and why FN is often the main risk, (4) caveats,\n"
    "(5) next improvement steps.\n"
    "Assume demo/sandbox use. Do not present as operational instruction.\n"
)

VOICE_PROMPT_LOCKED = (
    "You are a voice narrator for an ICT decision-support demo.\n"
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

client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_SDK_OK and OPENAI_API_KEY) else None


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
    if not OPENAI_SDK_OK:
        return (
            "OpenAI Python SDK is not installed (or is too old).\n\n"
            "Install or upgrade it, then restart:\n"
            "  pip install -U openai\n"
        )

    if not OPENAI_API_KEY:
        return (
            "OPENAI_API_KEY not set.\n\n"
            "Set it in your environment and restart:\n"
            "Windows (PowerShell):  $env:OPENAI_API_KEY=\"your_key_here\"\n"
            "Windows (CMD):         setx OPENAI_API_KEY \"your_key_here\"\n"
            "macOS/Linux:           export OPENAI_API_KEY=\"your_key_here\"\n"
        )

    if not client:
        return "OpenAI client not initialised. Check OPENAI_API_KEY and restart."

    if not user_message or not user_message.strip():
        return "Type a question (e.g., “What does false negative mean in this context?”)."

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

    # Support both newer (Responses API) and older (Chat Completions) OpenAI SDKs.
    try:
        if hasattr(client, "responses"):
            resp = client.responses.create(model=model_name, input=messages)
            txt = _extract_output_text(resp)
        else:
            # Fallback: chat.completions
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
            )
            txt = (resp.choices[0].message.content or "").strip()  # type: ignore
            if not txt:
                txt = "No text returned from the model."
    except Exception as e:
        return f"OpenAI call failed: {e}"

    dt_ms = int((time.time() - t0) * 1000)
    footer = f"\n\n—\nResponse time: {dt_ms} ms • Model: {model_name} • Demo"
    return (txt + footer).strip()


# =============================================================================
# VOICE SAFE TEXT
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
    for abbr, full in _ABBR_MAP.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        m = re.search(pattern, text)
        if m:
            text = re.sub(pattern, f"{abbr} ({full})", text, count=1)
    return text


def strip_markdown_for_tts(text: str) -> str:
    if not text:
        return ""

    t = str(text)
    t = re.sub(r"\n—\n.*$", "", t, flags=re.DOTALL)
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-•*+]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\|", " ", t)
    t = re.sub(r"^\s*:?-{2,}:?\s*$", " ", t, flags=re.MULTILINE)

    t = t.replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ")
    t = t.replace("\\", " ").replace("_", " ").replace("*", " ").replace("#", " ")

    t = re.sub(r"\bNaN\b", "not available", t, flags=re.IGNORECASE)
    t = re.sub(r"\bn/?a\b", "not available", t, flags=re.IGNORECASE)

    t = re.sub(r"\s+", " ", t).strip()
    t = _expand_abbr_once(t)
    return t


def make_voice_summary(answer_text: str) -> str:
    if not client:
        cleaned = strip_markdown_for_tts(answer_text)
        return cleaned[:1200].rsplit(" ", 1)[0] + "…" if len(cleaned) > 1200 else cleaned

    try:
        model_name = os.getenv("OPENAI_MODEL", MODEL_NAME_DEFAULT)
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
        spoken = strip_markdown_for_tts(spoken)
        return spoken[:1200].rsplit(" ", 1)[0] + "…" if len(spoken) > 1200 else spoken
    except Exception:
        cleaned = strip_markdown_for_tts(answer_text)
        return cleaned[:1200].rsplit(" ", 1)[0] + "…" if len(cleaned) > 1200 else cleaned


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
    vtxt = (voice_text or "").strip()
    if not vtxt:
        fallback = _last_assistant_text(history)
        vtxt = strip_markdown_for_tts(fallback)
    return speak_text(vtxt, voice)


# =============================================================================
# SYNTHETIC DATA: ICT INCIDENT ESCALATION
# =============================================================================
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_synthetic_ict_incident_escalation(
    n_rows: int = 600,
    base_escalation_rate: float = 0.18,
    noise: float = 0.70,
    label_flip: float = 0.03,
    missingness: float = 0.08,
    seed: int = 42
) -> pd.DataFrame:
    """
    Synthetic incident escalation dataset.
    Target: needs_escalation (0/1)
    """
    rng = np.random.default_rng(int(seed))
    n = int(max(80, min(5000, n_rows)))

    base_escalation_rate = float(np.clip(base_escalation_rate, 0.03, 0.60))
    noise = float(np.clip(noise, 0.05, 2.00))
    label_flip = float(np.clip(label_flip, 0.00, 0.20))
    missingness = float(np.clip(missingness, 0.00, 0.35))

    # Operational signals
    alerts_15m = rng.poisson(lam=2.2, size=n).clip(0, 25)
    error_rate = rng.lognormal(mean=-1.6 + 0.06 * alerts_15m, sigma=0.55, size=n).clip(0.0001, 1.0)  # 0-1
    p95_latency_ms = rng.lognormal(mean=5.6 + 0.02 * alerts_15m, sigma=0.55, size=n).clip(50, 60000)
    cpu_pct = (rng.normal(42, 18, n) + 10 * (error_rate > 0.10)).clip(1, 100)
    mem_pct = (rng.normal(55, 15, n) + 8 * (p95_latency_ms > 800)).clip(5, 100)
    disk_pct = (rng.normal(62, 16, n) + 10 * (alerts_15m > 10)).clip(5, 100)

    # Human signals
    user_reports_30m = rng.poisson(lam=0.8 + 0.20 * (alerts_15m > 6), size=n).clip(0, 40)
    vip_involved = rng.binomial(1, 0.06 + 0.02 * (user_reports_30m > 5), size=n)

    # Context signals
    after_hours = rng.binomial(1, 0.35, size=n)
    change_window = rng.binomial(1, 0.18 + 0.15 * after_hours, size=n)
    clinical_system = rng.binomial(1, 0.28, size=n)  # e.g., EPR, PAS, lab system
    integration_related = rng.binomial(1, 0.22 + 0.10 * change_window, size=n)
    vendor_dependency = rng.binomial(1, 0.15, size=n)

    # Impact proxy
    affected_services = rng.poisson(lam=1.3 + 0.25 * (clinical_system == 1) + 0.15 * (integration_related == 1), size=n).clip(0, 15)
    est_users_impacted = rng.lognormal(mean=3.2 + 0.12 * affected_services + 0.20 * clinical_system, sigma=0.8, size=n).clip(1, 20000)

    # Latent escalation “need”
    z = (
        0.22 * np.log1p(alerts_15m)
        + 1.10 * np.log1p(error_rate * 1000)
        + 0.18 * np.log1p(p95_latency_ms)
        + 0.010 * cpu_pct
        + 0.009 * mem_pct
        + 0.007 * disk_pct
        + 0.25 * np.log1p(user_reports_30m)
        + 0.55 * clinical_system
        + 0.35 * integration_related
        + 0.30 * change_window
        + 0.25 * after_hours
        + 0.25 * vendor_dependency
        + 0.20 * np.log1p(affected_services)
        + 0.22 * np.log1p(est_users_impacted)
        + 0.40 * vip_involved
        + rng.normal(0, 1.0 * noise, n)
    )

    p_raw = _sigmoid(z)

    # Calibrate base rate by shifting logits
    eps = 1e-6
    logits = np.log(np.clip(p_raw, eps, 1 - eps) / np.clip(1 - p_raw, eps, 1 - eps))
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p_mid = _sigmoid(logits + mid)
        if p_mid.mean() > base_escalation_rate:
            hi = mid
        else:
            lo = mid
    p = _sigmoid(logits + (lo + hi) / 2)

    needs_escalation = rng.binomial(1, p, n)

    # Label noise
    if label_flip > 0:
        flip_mask = rng.random(n) < label_flip
        needs_escalation = np.where(flip_mask, 1 - needs_escalation, needs_escalation)

    df = pd.DataFrame({
        "alerts_15m": alerts_15m.astype(int),
        "error_rate": np.round(error_rate, 4),
        "p95_latency_ms": np.round(p95_latency_ms).astype(int),
        "cpu_pct": np.round(cpu_pct, 1),
        "mem_pct": np.round(mem_pct, 1),
        "disk_pct": np.round(disk_pct, 1),
        "user_reports_30m": user_reports_30m.astype(int),
        "vip_involved": vip_involved.astype(int),
        "after_hours": after_hours.astype(int),
        "change_window": change_window.astype(int),
        "clinical_system": clinical_system.astype(int),
        "integration_related": integration_related.astype(int),
        "vendor_dependency": vendor_dependency.astype(int),
        "affected_services": affected_services.astype(int),
        "est_users_impacted": np.round(est_users_impacted).astype(int),
        "needs_escalation": needs_escalation.astype(int),
    })

    # Missingness in noisy operational signals
    miss_cols = ["error_rate", "p95_latency_ms", "cpu_pct", "mem_pct", "disk_pct", "est_users_impacted"]
    if missingness > 0:
        for col in miss_cols:
            mask = rng.random(n) < missingness
            df.loc[mask, col] = np.nan

    return df


# =============================================================================
# SUMMARY STATS
# =============================================================================
def dataset_summary(df: pd.DataFrame, target_col: str) -> Tuple[str, pd.DataFrame]:
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
def plot_confusion_matrix_highlight_fn(cm: np.ndarray, threshold: float, title: str, labels: Tuple[str, str]) -> Optional[str]:
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
        ax.set_xticklabels([f"0 ({labels[0]})", f"1 ({labels[1]})"])
        ax.set_yticklabels([f"0 ({labels[0]})", f"1 ({labels[1]})"])

        for (i, j), v in np.ndenumerate(cm):
            label = str(int(v))
            extra = ""
            if i == 1 and j == 0:
                extra = "\n(FALSE NEGATIVE)"
            ax.text(j, i, label + extra, ha="center", va="center", fontsize=10, fontweight="bold")

        # highlight FN cell (actual=1, pred=0) => row 1, col 0
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
        ax.set_title("ROC Curve")
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
        ax.set_ylabel("Observed positive rate")
        ax.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

        return path, brier, ece
    except Exception:
        return None, None, None


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
            return pd.DataFrame(columns=["predicted_score", "threshold", "actual_positive"] + list(x_test.columns))

        fn_idx = np.where(mask_fn)[0]
        rows = x_test.iloc[fn_idx].copy()
        rows.insert(0, "actual_positive", 1)
        rows.insert(0, "threshold", thr)
        rows.insert(0, "predicted_score", proba[fn_idx])

        rows = rows.sort_values("predicted_score", ascending=False).head(int(max_rows))
        rows["predicted_score"] = rows["predicted_score"].round(4)
        return rows.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["predicted_score", "threshold", "actual_positive"] + list(x_test.columns))


# =============================================================================
# TRAINING (Classification baseline)
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

    features = [f for f in features if f in df.columns and f != target_col]
    if not features:
        notes.append("Selected features are not in dataset (or only the target was selected).")
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
    y = pd.to_numeric(work[target_col], errors="coerce")
    work = work.loc[~y.isna()].copy()
    y = y.loc[work.index].astype(int)

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
    top_coef["direction"] = np.where(top_coef["coef"] >= 0, "↑ increases positive risk", "↓ decreases positive risk")
    top_coef = top_coef[["feature", "coef", "direction"]]
    top_coef["coef"] = top_coef["coef"].round(4)

    roc_path = plot_roc_curve(np.array(y_test), np.array(proba)) if (MPL_OK and auc_f is not None) else None
    cm_path = plot_confusion_matrix_highlight_fn(
        cm,
        threshold=threshold,
        title="Confusion Matrix (FN highlighted)",
        labels=("No escalation", "Needs escalation")
    ) if MPL_OK else None

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
            "AUC is extremely high. In real operational data this is unusual; it may indicate synthetic rules are too clean, "
            "or a feature leaks the outcome (recorded after the escalation decision)."
        )

    if fn > 0:
        notes.append(
            f"Safety note: FN (missed positives) at threshold {threshold:.2f} = {fn}. "
            "For escalation, FN often means missing a genuinely critical incident."
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

    cm_path = plot_confusion_matrix_highlight_fn(
        cm,
        threshold=threshold,
        title="Confusion Matrix (FN highlighted)",
        labels=("No escalation", "Needs escalation")
    ) if MPL_OK else None

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
# EVALUATE EXISTING DECISIONS (NO TRAINING REQUIRED)
# =============================================================================
def evaluate_decisions_from_columns(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    score_col: str,
    threshold: float,
    top_fn_n: int,
    labels: Tuple[str, str]
) -> Dict[str, Any]:
    """
    If predicted_col provided: use it.
    Else if score_col provided: generate predicted via score >= threshold.
    Requires actual_col always.
    """
    out: Dict[str, Any] = {"ok": False, "msg": "", "cm_path": None, "cal_path": None}

    if df is None or df.empty:
        out["msg"] = "No dataset loaded."
        return out

    if not actual_col or actual_col not in df.columns:
        out["msg"] = "Actual column not set or not found."
        return out

    y = pd.to_numeric(df[actual_col], errors="coerce").dropna().astype(int)
    work = df.loc[y.index].copy()

    proba = None
    if predicted_col and predicted_col in work.columns:
        pred = pd.to_numeric(work[predicted_col], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        if not score_col or score_col not in work.columns:
            out["msg"] = "Provide either a Predicted column (0/1) or a Score/Probability column."
            return out
        proba = pd.to_numeric(work[score_col], errors="coerce").to_numpy()
        proba = np.where(np.isnan(proba), np.nanmedian(proba), proba)
        threshold = float(np.clip(threshold, 0.01, 0.99))
        pred = (proba >= threshold).astype(int)

    y_true = y.to_numpy()

    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    m = _binary_metrics_from_counts(tp=tp, fp=fp, fn=fn, tn=tn)

    cm_path = plot_confusion_matrix_highlight_fn(cm, threshold=threshold, title="Confusion Matrix (FN highlighted)", labels=labels) if MPL_OK else None

    # If we have proba, do calibration; else skip
    cal_path = None
    brier = None
    ece = None
    if proba is not None and SKLEARN_OK and MPL_OK:
        cal_path, brier, ece = plot_calibration_curve(y_true=y_true, proba=proba, n_bins=10, title="Calibration (Reliability) Curve")

    # Top FN table (if score exists, else we can’t rank)
    fn_table = pd.DataFrame()
    if proba is not None:
        # use all non-target cols as context (safe: no PII assumed)
        x_cols = [c for c in work.columns if c not in [actual_col, predicted_col, score_col]]
        x_test = work[x_cols].copy() if x_cols else pd.DataFrame(index=work.index)
        fn_table = build_top_fn_table(
            x_test=x_test if not x_test.empty else pd.DataFrame({"row_index": np.arange(len(work))}),
            y_test=y_true,
            proba=proba,
            threshold=threshold,
            max_rows=int(max(5, min(50, top_fn_n)))
        )

    out.update({
        "ok": True,
        "rows": int(len(work)),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": _safe_float(m.get("sensitivity")),
        "specificity": _safe_float(m.get("specificity")),
        "precision": _safe_float(m.get("precision")),
        "accuracy": _safe_float(m.get("accuracy")),
        "cm_path": cm_path,
        "cal_path": cal_path,
        "brier": _safe_float(brier),
        "ece": _safe_float(ece),
        "fn_table": fn_table,
        "msg": "Evaluation complete."
    })
    return out


# =============================================================================
# CONTEXT PACKING for the LLM
# =============================================================================
def build_llm_context(
    df: Optional[pd.DataFrame],
    summary_text: str,
    numeric_stats: Optional[pd.DataFrame],
    mode_state: Optional[Dict[str, Any]],
) -> str:
    parts = []
    parts.append("DEMO STANCE: Sandbox/demo use. Explain current data and outputs only.")
    parts.append("")
    parts.append("DATASET SUMMARY:")
    parts.append(summary_text.strip() if summary_text else "n/a")

    if numeric_stats is not None and not numeric_stats.empty:
        head = numeric_stats.head(10).copy()
        parts.append("")
        parts.append("NUMERIC STATS (first 10 features):")
        parts.append(head.to_json(orient="records"))

    if mode_state and isinstance(mode_state, dict) and mode_state.get("trained_or_evaluated"):
        parts.append("")
        parts.append("CURRENT OUTPUTS (this run):")
        for k in ["mode", "target", "rows_used", "threshold", "auc", "accuracy", "sensitivity", "specificity", "precision", "tp", "fp", "fn", "tn", "brier", "ece"]:
            if k in mode_state:
                parts.append(f"{k}: {mode_state.get(k)}")
        if mode_state.get("top_coef"):
            parts.append("Top coefficients:")
            parts.append(json.dumps(mode_state.get("top_coef"), ensure_ascii=False))
        if mode_state.get("notes"):
            parts.append("Caveats:")
            parts.append(json.dumps(mode_state.get("notes"), ensure_ascii=False))
        if mode_state.get("top_fn_table"):
            parts.append("Top false negatives (FN) table:")
            parts.append(json.dumps(mode_state.get("top_fn_table"), ensure_ascii=False))
    else:
        parts.append("")
        parts.append("CURRENT OUTPUTS: none yet (train or evaluate).")

    return "\n".join(parts).strip()


# =============================================================================
# UI THEME (NHS-ish blue)
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
  font-weight: 800;
  color: var(--nhs-dark);
  margin: 6px 0 8px 0;
}}

.primary-btn button {{
  background: var(--nhs-blue) !important;
  border: 1px solid rgba(0,94,184,0.35) !important;
  color: white !important;
  border-radius: 12px !important;
  height: 44px !important;
  font-weight: 800 !important;
}}

.secondary-btn button {{
  background: white !important;
  border: 1px solid rgba(0,94,184,0.35) !important;
  color: var(--nhs-blue) !important;
  border-radius: 12px !important;
  height: 44px !important;
  font-weight: 800 !important;
}}

[data-testid="chatbot"] {{
  min-height: 740px !important;
}}
"""


# =============================================================================
# GRADIO APP
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

    voice_text = make_voice_summary(answer)
    return "", history, voice_text


def clear_chat() -> Tuple[List[Dict[str, str]], str]:
    return [], ""


def explain_current_results(history: List[Dict[str, str]], llm_context: str) -> Tuple[List[Dict[str, str]], str]:
    history = history or []
    prompt = (
        "Explain the current results for an executive audience.\n"
        "Use short sections:\n"
        "- What we evaluated or trained (1–2 lines)\n"
        "- What Accuracy / Sensitivity / Specificity mean (define each)\n"
        "- Highlight FN (missed positives) and what it implies for escalation risk\n"
        "- If calibration exists, explain it and include Brier + ECE\n"
        "- If coefficients exist, summarise top drivers (direction)\n"
        "- If a Top FN panel exists, explain what it shows and why it matters\n"
        "- Risks/caveats (including leakage suspicion)\n"
        "- Next improvement steps (3 bullets)\n"
    )
    answer = ask_llm_locked(user_message=prompt, context=llm_context or "", history_messages=history)
    history = _history_append(history, "assistant", answer)
    voice_text = make_voice_summary(answer)
    return history, voice_text


with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
    gr.Markdown(
        f"""
<div class="hero">
  <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:16px;">
    <div>
      <div style="font-size:28px; font-weight:900; color:{NHS_DARK}; line-height:1.1;">Decision Support Tutor</div>
      <div style="margin-top:8px; color:{MUTED}; max-width:980px;">
        Domain-agnostic decision support framework, demonstrated with <b>ICT incident escalation</b>.
        Works with any dataset that supports <b>Actual vs Predicted</b> (and optional score/probability).
      </div>
      <div class="badge-row">
        <span class="badge"><span class="dot"></span> FN highlighted (missed criticals)</span>
        <span class="badge"><span class="dot"></span> Threshold tuning</span>
        <span class="badge"><span class="dot"></span> Calibration when score exists</span>
        <span class="badge"><span class="dot"></span> Voice summary (speech-safe)</span>
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
    mode_state = gr.State({})
    llm_context_state = gr.State("")
    voice_text_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div class='section-title'>Data</div>")
            with gr.Group(elem_classes=["card"]):
                with gr.Row():
                    csv_in = gr.File(label="Upload CSV", file_types=[".csv"])
                    csv_generated = gr.File(label="Generated synthetic CSV (download)", interactive=False)
                status = gr.Textbox(label="Status", value="Ready.", interactive=False)

                with gr.Accordion("Generate synthetic ICT Incident Escalation dataset", open=True):
                    with gr.Row():
                        gen_rows = gr.Slider(80, 5000, value=700, step=20, label="Rows")
                        gen_rate = gr.Slider(0.03, 0.60, value=0.18, step=0.01, label="Base escalation rate (approx)")
                    with gr.Row():
                        gen_noise = gr.Slider(0.05, 2.00, value=0.70, step=0.01, label="Noise / overlap")
                        gen_flip = gr.Slider(0.00, 0.20, value=0.03, step=0.01, label="Label noise (flip %)")
                    with gr.Row():
                        gen_missing = gr.Slider(0.00, 0.35, value=0.08, step=0.01, label="Missingness")
                        gen_seed = gr.Number(value=42, precision=0, label="Random seed")
                    gen_btn = gr.Button("Generate dataset", elem_classes=["primary-btn"])

                gr.Markdown("<div class='section-title'>Dataset preview (first 25 rows)</div>")
                preview_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True)

                gr.Markdown("<div class='section-title'>Dataset summary</div>")
                summary_text = gr.Textbox(value="No dataset loaded.", lines=4, interactive=False, label="Summary (plain)")
                stats_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True)

            with gr.Tabs():
                with gr.Tab("Train model (optional)"):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("<div class='small-muted'>Train a baseline classifier so you can explore thresholds, FN vs FP, ROC and calibration.</div>")
                        target = gr.Dropdown(label="Target (binary 0/1)", choices=["needs_escalation"], value="needs_escalation", allow_custom_value=True)
                        features = gr.Dropdown(label="Features (numeric)", multiselect=True, choices=[], value=[])
                        threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Decision threshold (lower = fewer FN, more FP)")
                        cal_bins = gr.Slider(5, 20, value=10, step=1, label="Calibration bins")
                        top_fn_n = gr.Slider(5, 50, value=10, step=1, label="Top FN rows to display")
                        train_btn = gr.Button("Train baseline model (Logistic Regression)", elem_classes=["primary-btn"])

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("<div class='section-title'>Results</div>")
                        kpi_auc = gr.Textbox(label="AUC", value="—", interactive=False)
                        kpi_acc = gr.Textbox(label="Accuracy @ threshold", value="—", interactive=False)
                        kpi_sens = gr.Textbox(label="Sensitivity (catch positives)", value="—", interactive=False)
                        kpi_spec = gr.Textbox(label="Specificity (dismiss negatives)", value="—", interactive=False)
                        kpi_fn = gr.Textbox(label="False Negatives (FN)", value="—", interactive=False)
                        kpi_rf = gr.Textbox(label="Rows / Features", value="—", interactive=False)

                        with gr.Row():
                            roc_img = gr.Image(value=None, label="ROC", type="filepath")
                            cm_img = gr.Image(value=None, label="Confusion Matrix (FN highlighted)", type="filepath")

                        cal_img = gr.Image(value=None, label="Calibration (if available)", type="filepath")
                        kpi_brier = gr.Textbox(label="Brier score", value="—", interactive=False)
                        kpi_ece = gr.Textbox(label="ECE approx", value="—", interactive=False)

                        coef_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True, label="Top coefficients (drivers)")
                        fn_sentence = gr.Textbox(label="Plain-English FN statement", value="—", interactive=False)
                        caveats = gr.Textbox(label="Caveats", value="", lines=4, interactive=False)
                        fn_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True, label="Top missed positives (FN)")

                with gr.Tab("Evaluate existing decisions (no training)"):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("<div class='small-muted'>Use this when your CSV already has Actual vs Predicted (0/1), and optionally a score/probability.</div>")
                        actual_col = gr.Dropdown(label="Actual column (0/1)", choices=[], value=None, allow_custom_value=True)
                        predicted_col = gr.Dropdown(label="Predicted column (0/1) (optional)", choices=[], value=None, allow_custom_value=True)
                        score_col = gr.Dropdown(label="Score/Probability column (optional)", choices=[], value=None, allow_custom_value=True)
                        eval_threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Threshold (used only if score/probability is provided)")
                        eval_top_fn = gr.Slider(5, 50, value=10, step=1, label="Top FN rows to display (requires score)")
                        eval_btn = gr.Button("Evaluate confusion matrix", elem_classes=["primary-btn"])

                    with gr.Group(elem_classes=["card"]):
                        eval_acc = gr.Textbox(label="Accuracy", value="—", interactive=False)
                        eval_sens = gr.Textbox(label="Sensitivity", value="—", interactive=False)
                        eval_spec = gr.Textbox(label="Specificity", value="—", interactive=False)
                        eval_fn = gr.Textbox(label="False Negatives (FN)", value="—", interactive=False)
                        eval_counts = gr.Textbox(label="Counts (TP/FP/FN/TN)", value="—", interactive=False)
                        eval_cm_img = gr.Image(value=None, label="Confusion Matrix (FN highlighted)", type="filepath")
                        eval_cal_img = gr.Image(value=None, label="Calibration (if score exists)", type="filepath")
                        eval_brier = gr.Textbox(label="Brier score", value="—", interactive=False)
                        eval_ece = gr.Textbox(label="ECE approx", value="—", interactive=False)
                        eval_fn_df = gr.Dataframe(value=pd.DataFrame(), interactive=False, wrap=True, label="Top missed positives (FN)")

        with gr.Column(scale=1.15):
            gr.Markdown("<div class='section-title'>AI Analyst Commentary</div>")
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("<div class='small-muted'>Locked scope: explains the <b>current dataset</b> + <b>current outputs</b>.</div>")

                gr.Markdown("<div class='section-title'>Voice (TTS)</div>")
                with gr.Row():
                    voice_choice = gr.Dropdown(label="Voice", choices=VOICE_CHOICES, value=DEFAULT_VOICE, interactive=True)
                    speak_btn = gr.Button("🔊 Speak voice summary", elem_classes=["secondary-btn"])
                tts_audio = gr.Audio(label="Voice output", type="filepath")
                tts_status = gr.Textbox(label="TTS status", value=("edge-tts ready" if EDGE_TTS_OK else "Install edge-tts: pip install edge-tts"), interactive=False)

                user_context = gr.Textbox(
                    label="Context (optional)",
                    placeholder="Example: Explain FN risk for escalation, threshold trade-offs, and what we should improve next.",
                    lines=3
                )
                chatbot = gr.Chatbot(label="Commentary", height=740)
                explain_btn = gr.Button("Explain current results", elem_classes=["primary-btn"])

                msg = gr.Textbox(label="Your question", placeholder="e.g., Why is FN the main risk in escalation? What does sensitivity mean?")
                with gr.Row():
                    send = gr.Button("Send", elem_classes=["primary-btn"])
                    clear = gr.Button("Clear commentary", elem_classes=["secondary-btn"])

    # --------------------------
    # Handlers
    # --------------------------
    def _refresh_column_choices(df: pd.DataFrame):
        if df is None or df.empty:
            return [], [], [], [], [], []
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return cols, cols, cols, cols, num_cols, num_cols

    def _on_csv_upload(file_obj):
        if file_obj is None:
            df = pd.DataFrame()
            summary, stats = "No CSV loaded.", pd.DataFrame()
            packed = build_llm_context(df, summary, stats, mode_state={})
            return df, pd.DataFrame(), summary, stats, "Ready.", {}, packed, *(_refresh_column_choices(df))

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
            packed = build_llm_context(df, summary, stats, mode_state={})
            return df, pd.DataFrame(), summary, stats, "Upload failed.", {}, packed, *(_refresh_column_choices(df))

        try:
            df = pd.read_csv(path)
        except Exception as e:
            df = pd.DataFrame()
            summary, stats = f"Failed to read CSV: {e}", pd.DataFrame()
            packed = build_llm_context(df, summary, stats, mode_state={})
            return df, pd.DataFrame(), summary, stats, "CSV read error.", {}, packed, *(_refresh_column_choices(df))

        preview = df.head(25)
        # best-guess target for summary
        guess_target = "needs_escalation" if "needs_escalation" in df.columns else (df.columns[-1] if df.shape[1] else "target")
        summary, stats = dataset_summary(df, target_col=str(guess_target))
        status_txt = f"Loaded: {os.path.basename(path)} (rows={len(df)}, cols={df.shape[1]})"
        packed = build_llm_context(df, summary, stats, mode_state={})
        return df, preview, summary, stats, status_txt, {}, packed, *(_refresh_column_choices(df))

    def _on_generate(n_rows, base_rate, noise, label_flip, missingness, seed):
        try:
            df = generate_synthetic_ict_incident_escalation(
                n_rows=int(n_rows),
                base_escalation_rate=float(base_rate),
                noise=float(noise),
                label_flip=float(label_flip),
                missingness=float(missingness),
                seed=int(seed),
            )
            os.makedirs("reports", exist_ok=True)
            out_path = os.path.join("reports", f"synthetic_ict_escalation_{uuid.uuid4().hex[:8]}.csv")
            df.to_csv(out_path, index=False)

            summary, stats = dataset_summary(df, target_col="needs_escalation")
            status_txt = f"Generated dataset: {os.path.basename(out_path)} (rows={len(df)}, cols={df.shape[1]})"
            packed = build_llm_context(df, summary, stats, mode_state={})
            return df, out_path, df.head(25), summary, stats, status_txt, {}, packed, *(_refresh_column_choices(df))
        except Exception as e:
            df = pd.DataFrame()
            packed = build_llm_context(df, "n/a", pd.DataFrame(), mode_state={})
            return df, "", pd.DataFrame(), f"Dataset generation failed: {e}", pd.DataFrame(), "Error.", {}, packed, *(_refresh_column_choices(df))

    def _on_train_classif(df_full, target_col, feats, thr, bins, topn, summary_txt, stats_frame):
        if df_full is None or (isinstance(df_full, pd.DataFrame) and df_full.empty):
            packed = build_llm_context(pd.DataFrame(), summary_txt or "No dataset loaded.", stats_frame, mode_state={})
            return ("n/a","n/a","n/a","n/a","n/a","—",None,None,"—","—",pd.DataFrame(),"—","",pd.DataFrame(),{},packed)

        snap = train_baseline_logreg(
            df_full,
            str(target_col),
            feats or [],
            threshold=float(thr),
            cal_bins=int(bins),
            top_fn_n=int(topn),
        )

        rows = snap.rows
        feat_count = len(snap.features or [])
        rf = f"{rows} / {feat_count}"

        k_auc = f"{snap.auc:.3f}" if snap.auc is not None else "n/a"
        k_acc = _format_pct(snap.accuracy_at_threshold)
        k_sens = _format_pct(snap.sensitivity)
        k_spec = _format_pct(snap.specificity)
        k_fn = str(snap.fn)

        brier_txt = f"{snap.brier:.4f}" if snap.brier is not None else "n/a"
        ece_txt = f"{snap.ece:.4f}" if snap.ece is not None else "n/a"

        true_pos = snap.tp + snap.fn
        fn_sentence_txt = f"At threshold {snap.threshold:.2f}, the model missed {snap.fn} out of {true_pos} true positives." if true_pos > 0 else "—"
        cave = "\n".join([f"- {n}" for n in (snap.notes or [])]) if snap.notes else ""

        state = {
            "trained_or_evaluated": True,
            "mode": "train_model",
            "target": snap.target,
            "rows_used": snap.rows,
            "threshold": snap.threshold,
            "auc": snap.auc,
            "accuracy": snap.accuracy_at_threshold,
            "sensitivity": snap.sensitivity,
            "specificity": snap.specificity,
            "precision": snap.precision,
            "tp": snap.tp, "fp": snap.fp, "fn": snap.fn, "tn": snap.tn,
            "brier": snap.brier,
            "ece": snap.ece,
            "notes": snap.notes or [],
            "top_coef": snap.top_coef.to_dict(orient="records") if isinstance(snap.top_coef, pd.DataFrame) else [],
            "top_fn_table": snap.fn_table.to_dict(orient="records") if isinstance(snap.fn_table, pd.DataFrame) else [],
            "roc_path": snap.roc_path,
            "cm_path": snap.cm_path,
            "cal_path": snap.cal_path,
        }

        packed = build_llm_context(df_full, summary_txt or "", stats_frame, mode_state=state)

        return (
            k_auc, k_acc, k_sens, k_spec, k_fn, rf,
            snap.roc_path, snap.cm_path,
            snap.cal_path, brier_txt, ece_txt,
            snap.top_coef, fn_sentence_txt, cave, snap.fn_table, state, packed
        )

    def _on_eval(df_full, actual, predicted, score, thr, topn, summary_txt, stats_frame):
        if df_full is None or df_full.empty:
            packed = build_llm_context(pd.DataFrame(), summary_txt or "No dataset loaded.", stats_frame, mode_state={})
            return ("—","—","—","—","—",None,None,"—","—",pd.DataFrame(),{},packed)

        res = evaluate_decisions_from_columns(
            df=df_full,
            actual_col=str(actual) if actual else "",
            predicted_col=str(predicted) if predicted else "",
            score_col=str(score) if score else "",
            threshold=float(thr),
            top_fn_n=int(topn),
            labels=("No escalation", "Needs escalation")
        )

        if not res.get("ok"):
            packed = build_llm_context(df_full, summary_txt or "", stats_frame, mode_state={})
            return ("—","—","—","—",res.get("msg","Error"),None,None,"—","—",pd.DataFrame(),{},packed)

        counts = f"TP={res['tp']} FP={res['fp']} FN={res['fn']} TN={res['tn']}"
        eval_state = {
            "trained_or_evaluated": True,
            "mode": "evaluate_decisions",
            "target": str(actual),
            "rows_used": res["rows"],
            "threshold": res["threshold"],
            "accuracy": res["accuracy"],
            "sensitivity": res["sensitivity"],
            "specificity": res["specificity"],
            "precision": res["precision"],
            "tp": res["tp"], "fp": res["fp"], "fn": res["fn"], "tn": res["tn"],
            "brier": res.get("brier"),
            "ece": res.get("ece"),
            "notes": [],
            "top_fn_table": res["fn_table"].to_dict(orient="records") if isinstance(res["fn_table"], pd.DataFrame) else [],
        }

        packed = build_llm_context(df_full, summary_txt or "", stats_frame, mode_state=eval_state)

        brier_txt = f"{res['brier']:.4f}" if res.get("brier") is not None else "n/a"
        ece_txt = f"{res['ece']:.4f}" if res.get("ece") is not None else "n/a"

        return (
            _format_pct(res.get("accuracy")),
            _format_pct(res.get("sensitivity")),
            _format_pct(res.get("specificity")),
            str(res.get("fn")),
            counts,
            res.get("cm_path"),
            res.get("cal_path"),
            brier_txt,
            ece_txt,
            res.get("fn_table", pd.DataFrame()),
            eval_state,
            packed
        )

    # Wiring: upload/generate
    csv_in.change(
        _on_csv_upload,
        inputs=[csv_in],
        outputs=[
            df_state, preview_df, summary_text, stats_df, status,
            mode_state, llm_context_state,
            target, actual_col, predicted_col, score_col, features, features
        ],
    )

    gen_btn.click(
        _on_generate,
        inputs=[gen_rows, gen_rate, gen_noise, gen_flip, gen_missing, gen_seed],
        outputs=[
            df_state, csv_generated, preview_df, summary_text, stats_df, status,
            mode_state, llm_context_state,
            target, actual_col, predicted_col, score_col, features, features
        ],
    )

    # Train model
    train_btn.click(
        _on_train_classif,
        inputs=[df_state, target, features, threshold, cal_bins, top_fn_n, summary_text, stats_df],
        outputs=[
            kpi_auc, kpi_acc, kpi_sens, kpi_spec, kpi_fn, kpi_rf,
            roc_img, cm_img,
            cal_img, kpi_brier, kpi_ece,
            coef_df, fn_sentence, caveats, fn_df,
            mode_state, llm_context_state
        ],
    )

    # Evaluate existing
    eval_btn.click(
        _on_eval,
        inputs=[df_state, actual_col, predicted_col, score_col, eval_threshold, eval_top_fn, summary_text, stats_df],
        outputs=[eval_acc, eval_sens, eval_spec, eval_fn, eval_counts, eval_cm_img, eval_cal_img, eval_brier, eval_ece, eval_fn_df, mode_state, llm_context_state],
    )

    # Commentary + voice
    send.click(chat, inputs=[msg, user_context, chatbot, llm_context_state], outputs=[msg, chatbot, voice_text_state])
    msg.submit(chat, inputs=[msg, user_context, chatbot, llm_context_state], outputs=[msg, chatbot, voice_text_state])
    clear.click(clear_chat, inputs=None, outputs=[chatbot, voice_text_state])

    explain_btn.click(explain_current_results, inputs=[chatbot, llm_context_state], outputs=[chatbot, voice_text_state])

    speak_btn.click(
        speak_last_answer,
        inputs=[voice_text_state, chatbot, voice_choice],
        outputs=[tts_audio, tts_status]
    )


if __name__ == "__main__":
    # Friendly defaults for local run AND most hosting platforms.
    # - PORT is used by many providers (Render, Railway, etc.)
    # - server_name=0.0.0.0 lets the app bind externally in containers/VMs
    port = int(os.getenv("PORT", "7860"))
    server = os.getenv("SERVER_NAME", "0.0.0.0")
    demo.queue().launch(server_name=server, server_port=port, show_error=True)
