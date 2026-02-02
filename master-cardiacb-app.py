import os
import json
import math
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import gradio as gr

from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "Clinical ML Tutor"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DEMO_DISCLAIMER = (
    "Demo stance: Synthetic demo only. No patient-identifiable input. "
    "Not clinical advice. This tool supports learning and analytics capability uplift."
)

SYSTEM_PROMPT = (
    "You are the AI Analyst Commentary panel for a synthetic-only clinical ML demo.\n"
    "You must ONLY use the provided CURRENT ANALYSIS SNAPSHOT to answer.\n"
    "If the snapshot does not contain something, say you don't have it.\n"
    "Do NOT give medical advice. Do NOT interpret individual rows as patients.\n"
    "Stay focused: explain the current model output, metrics, drivers, caveats, and next steps.\n"
    "Use plain English. Short sections. Define any technical term immediately.\n"
    "Never claim real-world readiness or clinical deployment.\n"
)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ============================================================
# STYLING (NHS Blue, accessible contrast)
# ============================================================
CSS = """
:root{
  --nhs-blue:#005EB8;
  --nhs-blue-dark:#003B75;
  --bg:#EAF2FF;
  --panel:#FFFFFF;
  --muted:#5A6B7A;
  --text:#0B1F33;
  --border:rgba(0,0,0,0.10);
  --shadow:0 10px 30px rgba(0,0,0,0.10);
  --radius:16px;
}

.gradio-container{
  background: radial-gradient(1200px 600px at 20% 0%, #F3F8FF 0%, var(--bg) 45%, #E6F0FF 100%) !important;
  color: var(--text) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Helvetica Neue", sans-serif;
}

#topbar{
  background: linear-gradient(90deg, rgba(0,94,184,0.12), rgba(0,94,184,0.04));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 18px;
  box-shadow: var(--shadow);
}

#topbar h1{
  margin: 0;
  font-size: 28px;
  line-height: 1.1;
  color: var(--nhs-blue-dark);
}

#topbar .sub{
  margin-top: 6px;
  color: var(--muted);
  font-size: 13px;
}

.badges{
  margin-top: 10px;
  display:flex;
  gap:10px;
  flex-wrap:wrap;
}

.badge{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.65);
  color: var(--text);
  font-size: 12px;
}

.dot{
  width:10px;
  height:10px;
  border-radius:999px;
  background: var(--nhs-blue);
}

.section-card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 14px;
  box-shadow: var(--shadow);
}

.section-title{
  font-weight: 700;
  color: var(--nhs-blue-dark);
  margin: 2px 0 10px 0;
}

.small-muted{
  color: var(--muted);
  font-size: 12px;
}

#preview_wrap{
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid var(--border);
}

#preview_wrap .wrap{
  border: none !important;
}

#preview_df{
  max-height: 340px;
  overflow: auto;
}

#kpi_grid{
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.kpi{
  background: rgba(0,94,184,0.06);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 12px;
}

.kpi .k{
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 2px;
}

.kpi .v{
  font-size: 15px;
  font-weight: 800;
  color: var(--text);
}

button.primary, .primary button{
  background: var(--nhs-blue) !important;
  color: #fff !important;
  border: none !important;
}

button.primary:hover, .primary button:hover{
  filter: brightness(0.95);
}

#explain_btn button{
  background: var(--nhs-blue) !important;
  color: #fff !important;
  border: none !important;
  padding: 12px 14px !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
}

#quick_btns button{
  border-radius: 999px !important;
  border: 1px solid var(--border) !important;
}

#footer_note{
  color: var(--muted);
  font-size: 12px;
  margin-top: 10px;
}
"""


# ============================================================
# UTIL: OpenAI Responses extraction (keeps your working behaviour)
# ============================================================
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


def ask_llm_with_snapshot(
    user_message: str,
    mode: str,
    snapshot: Dict[str, Any],
    history_messages: List[Dict[str, str]],
) -> str:
    """
    Locked AI analyst: must only use snapshot.
    mode: "executive" | "risks" | "next" | "qa"
    """
    if not client:
        return "OpenAI is not configured (missing OPENAI_API_KEY). The modelling side still works without AI."

    if not user_message.strip() and mode == "qa":
        return "Type a question about the current model results (e.g., 'What does AUC mean here?')."

    # Build a constrained instruction per mode
    mode_instructions = {
        "executive": (
            "Write an executive-ready briefing with EXACTLY these sections:\n"
            "1) What this model is doing (1–2 lines)\n"
            "2) Headline results (AUC, accuracy) — define AUC\n"
            "3) Main drivers (top positives/negatives) — explain direction\n"
            "4) Risks/caveats (synthetic, leakage risk, validation)\n"
            "5) One recommended next step (single step)\n"
            "Keep it plain English. No medical advice."
        ),
        "risks": (
            "List the key risks/caveats for this specific snapshot.\n"
            "Cover: synthetic vs real-world gap, leakage suspicion, no external validation, governance stance.\n"
            "Be concise and practical. No medical advice."
        ),
        "next": (
            "Recommend ONE next improvement step, chosen from:\n"
            "- feature review (pre vs post-event)\n"
            "- data quality checks\n"
            "- external validation\n"
            "- calibration\n"
            "- cohort expansion\n"
            "Explain why that one step matters for THIS snapshot. No medical advice."
        ),
        "qa": (
            "Answer the user’s question using ONLY the snapshot.\n"
            "Define any technical term immediately.\n"
            "If the question is unrelated to the snapshot, say so and redirect back to the model context."
        ),
    }

    snapshot_json = json.dumps(snapshot or {}, indent=2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CURRENT ANALYSIS SNAPSHOT (JSON):\n{snapshot_json}"},
        {"role": "user", "content": f"MODE INSTRUCTIONS:\n{mode_instructions.get(mode, mode_instructions['qa'])}"},
    ]

    # Keep minimal history, but still in locked scope
    for m in (history_messages or [])[-8:]:
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": str(m["content"])})

    if mode == "qa":
        messages.append({"role": "user", "content": user_message.strip()})
    else:
        messages.append({"role": "user", "content": "Generate the requested output now."})

    t0 = time.time()
    resp = client.responses.create(model=MODEL_NAME, input=messages)
    out = _extract_output_text(resp)
    ms = int((time.time() - t0) * 1000)

    # Add a tiny footer line for demo traceability (execs love this)
    out += f"\n\n—\nResponse time: {ms} ms • Model: {MODEL_NAME} • Synthetic demo"
    return out


# ============================================================
# DATA + MODELLING
# ============================================================
def _safe_read_csv(path: str) -> pd.DataFrame:
    # robust defaults; you can extend later
    return pd.read_csv(path)


def _suggest_binary_target(df: pd.DataFrame) -> Optional[str]:
    # Prefer typical demo target names first
    for c in ["event_30d", "event30d", "outcome", "label", "target", "y"]:
        if c in df.columns:
            return c
    # Otherwise find first binary-ish column
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        uniq = set(s.unique().tolist())
        if uniq.issubset({0, 1}) and len(uniq) == 2:
            return c
    return None


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Convert booleans and numeric-like strings; leave non-numeric columns untouched
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)
        if out[c].dtype == object:
            # attempt numeric conversion where possible
            converted = pd.to_numeric(out[c], errors="ignore")
            out[c] = converted
    return out


def load_csv(file_obj) -> Tuple[pd.DataFrame, str, List[str], str, Dict[str, Any]]:
    """
    Returns:
      - preview dataframe (first 25 rows)
      - dataset summary markdown
      - target dropdown choices
      - default selected target
      - snapshot state (partial)
    """
    if file_obj is None:
        empty_snapshot = {"status": "no_dataset_loaded"}
        return pd.DataFrame(), "Upload a synthetic CSV to begin.", [], "", empty_snapshot

    try:
        df = _safe_read_csv(file_obj.name)
        df = _coerce_numeric(df)
    except Exception as e:
        empty_snapshot = {"status": "csv_load_failed", "error": str(e)}
        return pd.DataFrame(), f"**CSV load failed:** {e}", [], "", empty_snapshot

    preview = df.head(25)

    suggested = _suggest_binary_target(df)
    cols = df.columns.tolist()

    summary_lines = [
        "**Dataset summary**",
        f"- Rows: **{len(df):,}**",
        f"- Columns: **{len(cols)}**",
        f"- Suggested target: **{suggested or 'None detected'}**",
        f"- {DEMO_DISCLAIMER}",
    ]
    summary_md = "\n".join(summary_lines)

    snapshot = {
        "status": "dataset_loaded",
        "demo_disclaimer": DEMO_DISCLAIMER,
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(cols)),
            "column_names": cols,
            "suggested_target": suggested,
        },
        "model": None,
        "metrics": None,
        "drivers": None,
        "flags": [],
        "warnings": [],
    }

    return preview, summary_md, cols, (suggested or (cols[0] if cols else "")), snapshot


def train_model(
    file_obj,
    target_col: str,
    feature_cols: List[str],
) -> Tuple[str, str, pd.DataFrame, Any, Dict[str, Any]]:
    """
    Returns:
      - kpi markdown
      - metrics markdown
      - coefficients dataframe
      - roc image (matplotlib figure)
      - updated snapshot state
    """
    # Default outputs on failure
    empty_coef = pd.DataFrame(columns=["feature", "coef"])
    empty_fig = None

    if file_obj is None:
        snap = {"status": "no_dataset_loaded", "warnings": ["No CSV uploaded."]}
        return "Upload a synthetic CSV first.", "", empty_coef, empty_fig, snap

    try:
        df = _coerce_numeric(_safe_read_csv(file_obj.name))
    except Exception as e:
        snap = {"status": "csv_load_failed", "error": str(e)}
        return f"CSV load failed: {e}", "", empty_coef, empty_fig, snap

    if not target_col or target_col not in df.columns:
        snap = {"status": "invalid_target", "warnings": ["Target column not selected or not found."]}
        return "Select a valid target column.", "", empty_coef, empty_fig, snap

    if not feature_cols:
        snap = {"status": "no_features", "warnings": ["No features selected."]}
        return "Select at least one feature.", "", empty_coef, empty_fig, snap

    for c in feature_cols:
        if c not in df.columns:
            snap = {"status": "invalid_features", "warnings": [f"Feature not found: {c}"]}
            return f"Feature not found: {c}", "", empty_coef, empty_fig, snap

    # Prepare X/y
    y_raw = df[target_col]

    # enforce binary target
    y_vals = pd.to_numeric(y_raw, errors="coerce").dropna()
    if y_vals.empty:
        snap = {"status": "invalid_target_values", "warnings": ["Target is not numeric/binary."]}
        return "Target column must be binary (0/1).", "", empty_coef, empty_fig, snap

    # Align X/y after dropping NaNs in target
    df2 = df.loc[y_vals.index, :].copy()
    y = pd.to_numeric(df2[target_col], errors="coerce").astype(float)

    # Ensure binary 0/1
    uniq = set(pd.unique(y.dropna()).tolist())
    if not uniq.issubset({0.0, 1.0}) or len(uniq) < 2:
        snap = {"status": "non_binary_target", "warnings": [f"Target must contain both 0 and 1; got: {sorted(list(uniq))}"]}
        return "Target must be binary with both classes present (0 and 1).", "", empty_coef, empty_fig, snap

    X = df2[feature_cols].copy()

    # Coerce numeric for features (drop non-numeric)
    non_numeric = []
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            # try conversion
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().all():
            non_numeric.append(c)

    if non_numeric:
        # drop fully-non-numeric features
        X = X.drop(columns=non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    if X.shape[1] == 0:
        snap = {"status": "no_numeric_features", "warnings": ["No usable numeric features after conversion."]}
        return "No usable numeric features. Ensure features are numeric.", "", empty_coef, empty_fig, snap

    # Basic imputation: fill NaNs with column median
    X = X.apply(lambda col: col.fillna(col.median()) if np.issubdtype(col.dtype, np.number) else col)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Model pipeline
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    pipe.fit(X_train, y_train)

    # Predict probs for AUC/ROC
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, probs))
    acc = float(accuracy_score(y_test, preds))

    # Coefficients
    # Extract logistic regression coefficients mapped to feature names
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_.reshape(-1)
    coef_df = pd.DataFrame({"feature": X.columns.tolist(), "coef": coefs})
    coef_df = coef_df.sort_values("coef", ascending=False).reset_index(drop=True)

    # ROC Figure
    fpr, tpr, _thr = roc_curve(y_test, probs)
    fig = plt.figure(figsize=(6.2, 4.6), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve (Logistic Regression)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    # KPI markdown (exec friendly)
    kpi_md = (
        f"**Executive KPIs**\n\n"
        f"- Model: **Logistic Regression (explainable baseline)**\n"
        f"- AUC: **{auc:.3f}** (higher = better separation)\n"
        f"- Accuracy: **{acc:.3f}** (correct at 0.5 threshold)\n"
        f"- Rows: **{len(df2):,}** • Features used: **{X.shape[1]}**\n"
    )

    # Leakage heuristic flag (demo-friendly)
    flags = []
    warnings = []
    if auc >= 0.95:
        flags.append("possible_leakage_or_too_clean_data")
        warnings.append(
            "AUC is extremely high. In real NHS data this is unusual; "
            "it may indicate synthetic 'rules' are too clean or a feature leaks the target."
        )
    if non_numeric:
        warnings.append(f"Dropped non-numeric/empty features: {', '.join(non_numeric)}")

    # Drivers summary for snapshot
    top_pos = coef_df.head(5).to_dict(orient="records")
    top_neg = coef_df.tail(5).sort_values("coef", ascending=True).to_dict(orient="records")

    # Class balance
    pos_rate = float(y.mean())
    balance = {"positive_rate": pos_rate, "negative_rate": 1.0 - pos_rate}

    snapshot = {
        "status": "model_trained",
        "demo_disclaimer": DEMO_DISCLAIMER,
        "dataset": {
            "rows": int(len(df2)),
            "columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "target": target_col,
            "class_balance": balance,
        },
        "model": {
            "type": "Logistic Regression (baseline)",
            "threshold": 0.5,
            "features": X.columns.tolist(),
        },
        "metrics": {
            "auc": auc,
            "accuracy": acc,
        },
        "drivers": {
            "top_positive": top_pos,
            "top_negative": top_neg,
        },
        "flags": flags,
        "warnings": warnings,
    }

    # Metrics panel markdown (for analysts)
    metrics_md = (
        "**What the model is doing (simple)**\n"
        f"- Predicts the chance of **{target_col}=1** using your selected features.\n\n"
        "**How to read the numbers**\n"
        f"- **AUC ({auc:.3f})**: how well the model separates positives vs negatives across all thresholds.\n"
        f"- **Accuracy ({acc:.3f})**: how many it gets right at a **0.5** probability cut-off.\n\n"
        "**Coefficients (direction)**\n"
        "- Positive coef → higher feature value pushes predicted risk up.\n"
        "- Negative coef → higher feature value pushes predicted risk down.\n"
    )

    if warnings:
        metrics_md += "\n**Caveats detected**\n" + "\n".join([f"- {w}" for w in warnings])

    return kpi_md, metrics_md, coef_df.head(12), fig, snapshot


# ============================================================
# AI PANEL CALLBACKS
# ============================================================
def ai_chat(user_message: str, mode: str, snapshot: Dict[str, Any], history: List[Dict[str, str]]):
    history = history or []
    snapshot = snapshot or {"status": "no_snapshot"}

    # If model not trained, still allow guidance but grounded
    if snapshot.get("status") not in ["dataset_loaded", "model_trained"]:
        snapshot.setdefault("warnings", [])
        snapshot["warnings"].append("No trained model yet. Upload CSV and train to get full commentary.")

    answer = ask_llm_with_snapshot(user_message=user_message, mode=mode, snapshot=snapshot, history_messages=history)

    if mode == "qa":
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})
        return "", history
    else:
        # for button-driven modes, don't add an empty "user" line
        history.append({"role": "assistant", "content": answer})
        return "", history


def ai_clear():
    return []


# ============================================================
# BUILD UI
# ============================================================
with gr.Blocks(css=CSS, title=APP_TITLE) as demo:
    # --- Top bar
    gr.HTML(
        f"""
        <div id="topbar">
          <h1>{APP_TITLE}</h1>
          <div class="sub">
            AI partner for exploring a <b>synthetic</b> cardiac dataset — with explainable modelling + plain-English guidance (demo stance, not clinical advice).
          </div>
          <div class="badges">
            <div class="badge"><span class="dot"></span><b>Synthetic-only</b> guidance</div>
            <div class="badge"><span class="dot" style="background:#2D7D46;"></span><b>Plain-English</b> explanations</div>
            <div class="badge"><span class="dot" style="background:#FFB81C;"></span><b>Executive-ready</b> summaries</div>
          </div>
        </div>
        """
    )

    # Global state holding latest analysis snapshot
    analysis_snapshot = gr.State({"status": "no_snapshot"})
    ai_history_state = gr.State([])

    gr.Markdown("")

    with gr.Row():
        # ================= LEFT: Modelling =================
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### Cardiac regression (demo)")
                gr.Markdown(
                    "<div class='small-muted'>Upload a synthetic CSV → select target/features → train an explainable baseline model → review key signals.</div>"
                )

                file_in = gr.File(label="Upload synthetic CSV", file_types=[".csv"])

                preview_df = gr.Dataframe(
                    label="Dataset preview (first 25 rows)",
                    interactive=False,
                    elem_id="preview_df",
                    wrap=True,
                )

                with gr.Group(elem_id="preview_wrap"):
                    # This wrapper exists to let CSS control scrolling/height cleanly
                    gr.Markdown("<div class='small-muted'>Preview is scrollable. For exec demos, the point is clarity + context, not raw rows.</div>")

                summary_md = gr.Markdown("Upload a synthetic CSV to begin.")

            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### Model controls")
                target_dd = gr.Dropdown(label="Target (binary)", choices=[], value="")
                features_ms = gr.Dropdown(label="Features", choices=[], value=[], multiselect=True)
                train_btn = gr.Button("Train regression model", elem_classes=["primary"])

            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### Results")
                kpi_md = gr.Markdown("")
                metrics_md = gr.Markdown("")
                coef_df = gr.Dataframe(label="Coefficients (top signals)", interactive=False, wrap=True)
                roc_plot = gr.Plot(label="ROC Curve")

            gr.Markdown(f"<div id='footer_note'>{DEMO_DISCLAIMER}</div>")

        # ================= RIGHT: AI Analyst =================
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["section-card"]):
                gr.Markdown("### AI Analyst Commentary")
                gr.Markdown("<div class='small-muted'>This panel explains the <b>current model output</b> and its caveats. It is not a general chatbot.</div>")

                with gr.Row(elem_id="quick_btns"):
                    exec_btn = gr.Button("Generate executive briefing")
                    risks_btn = gr.Button("Risks / caveats")
                    next_btn = gr.Button("What should we improve next?")

                ai_box = gr.Chatbot(label="Commentary", height=420)
                user_q = gr.Textbox(label="Your question", placeholder="e.g., What does AUC mean? Why is troponin a top driver?")
                with gr.Row():
                    explain_btn = gr.Button("Explain current results", elem_id="explain_btn")
                    clear_btn = gr.Button("Clear commentary")

    # ============================================================
    # WIRING
    # ============================================================
    def on_file_change(file_obj):
        preview, summary, cols, default_target, snap = load_csv(file_obj)
        # update dropdowns
        return (
            preview,
            summary,
            gr.update(choices=cols, value=default_target),
            gr.update(choices=cols, value=[]),
            snap,
        )

    file_in.change(
        on_file_change,
        inputs=[file_in],
        outputs=[preview_df, summary_md, target_dd, features_ms, analysis_snapshot],
    )

    def on_train(file_obj, target, features, snap):
        kpi, metrics, coefs, fig, new_snap = train_model(file_obj, target, features)
        return kpi, metrics, coefs, fig, new_snap

    train_btn.click(
        on_train,
        inputs=[file_in, target_dd, features_ms, analysis_snapshot],
        outputs=[kpi_md, metrics_md, coef_df, roc_plot, analysis_snapshot],
    )

    # --- AI buttons / chat
    def do_exec(snapshot, history):
        return ai_chat("", "executive", snapshot, history)

    def do_risks(snapshot, history):
        return ai_chat("", "risks", snapshot, history)

    def do_next(snapshot, history):
        return ai_chat("", "next", snapshot, history)

    def do_explain(snapshot, history):
        # Explain is basically the exec briefing but slightly more analytical; reuse executive mode
        return ai_chat("", "executive", snapshot, history)

    def do_qa(user_message, snapshot, history):
        return ai_chat(user_message, "qa", snapshot, history)

    exec_btn.click(do_exec, inputs=[analysis_snapshot, ai_box], outputs=[user_q, ai_box])
    risks_btn.click(do_risks, inputs=[analysis_snapshot, ai_box], outputs=[user_q, ai_box])
    next_btn.click(do_next, inputs=[analysis_snapshot, ai_box], outputs=[user_q, ai_box])
    explain_btn.click(do_explain, inputs=[analysis_snapshot, ai_box], outputs=[user_q, ai_box])

    user_q.submit(do_qa, inputs=[user_q, analysis_snapshot, ai_box], outputs=[user_q, ai_box])

    clear_btn.click(lambda: [], inputs=None, outputs=ai_box)

if __name__ == "__main__":
    # If you're demoing to execs on a shared screen, you may prefer share=False.
    demo.launch()
