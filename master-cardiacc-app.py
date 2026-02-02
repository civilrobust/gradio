import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

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
# STYLING (NHS Blue, readable, exec-friendly)
# ============================================================
CSS = """
:root{
  --nhs-blue:#005EB8;
  --nhs-blue-dark:#003B75;
  --bg:#EAF2FF;
  --panel:#FFFFFF;
  --muted:#516273;
  --text:#0B1F33;
  --border:rgba(0,0,0,0.10);
  --shadow:0 12px 28px rgba(0,0,0,0.10);
  --radius:16px;
}

.gradio-container{
  background: radial-gradient(1200px 600px at 20% 0%, #F3F8FF 0%, var(--bg) 45%, #E6F0FF 100%) !important;
  color: var(--text) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Helvetica Neue", sans-serif;
}

#topbar{
  background: linear-gradient(90deg, rgba(0,94,184,0.14), rgba(0,94,184,0.05));
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
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
  background: rgba(255,255,255,0.72);
  color: var(--text);
  font-size: 12px;
  font-weight: 600;
}

.dot{ width:10px; height:10px; border-radius:999px; background: var(--nhs-blue); }

.card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px;
  box-shadow: var(--shadow);
}

.card h3{
  margin: 0 0 8px 0;
  color: var(--nhs-blue-dark);
}

.muted{ color: var(--muted); font-size: 12px; }

#preview_df{
  max-height: 280px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
}

#coef_df{
  max-height: 260px;
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 14px;
}

.kpi_grid{
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.kpi{
  background: rgba(0,94,184,0.06);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 10px 12px;
}
.kpi .k{ font-size: 11px; color: var(--muted); margin-bottom: 2px; }
.kpi .v{ font-size: 15px; font-weight: 800; color: var(--text); }

button.primary, .primary button{
  background: var(--nhs-blue) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
  padding: 10px 14px !important;
}

#explain_btn button{
  background: var(--nhs-blue) !important;
  color: #fff !important;
  border: none !important;
  padding: 12px 14px !important;
  border-radius: 14px !important;
  font-weight: 800 !important;
}

#right_sticky{
  position: sticky;
  top: 10px;
}

.footer_note{
  color: var(--muted);
  font-size: 12px;
  margin-top: 10px;
}
"""


# ============================================================
# OpenAI response extraction (keeps your working behaviour)
# ============================================================
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


def ask_llm_with_snapshot(
    user_message: str,
    mode: str,
    snapshot: Dict[str, Any],
    history_messages: List[Dict[str, str]],
) -> str:
    if not client:
        return "OpenAI is not configured (missing OPENAI_API_KEY). Modelling still works without AI."

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

    # Basic scope gate for QA: keep it locked to metrics/drivers/caveats/next steps
    allowed_keywords = [
        "auc", "accuracy", "roc", "coefficient", "coefficients", "driver", "feature",
        "risk", "caveat", "leak", "leakage", "validation", "threshold", "calibration",
        "next", "improve", "why", "how", "interpret"
    ]
    if mode == "qa" and user_message.strip():
        msg_lower = user_message.lower()
        if not any(k in msg_lower for k in allowed_keywords):
            return (
                "I can only answer questions about the current dataset/model results (AUC, accuracy, ROC, "
                "coefficients/drivers, risks/caveats, and next improvement steps). "
                "Ask something like: “What does AUC mean here?” or “Why is troponin a top driver?”"
            )

    snapshot_json = json.dumps(snapshot or {}, indent=2)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CURRENT ANALYSIS SNAPSHOT (JSON):\n{snapshot_json}"},
        {"role": "user", "content": f"MODE INSTRUCTIONS:\n{mode_instructions.get(mode, mode_instructions['qa'])}"},
    ]

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
    out += f"\n\n—\nResponse time: {ms} ms • Model: {MODEL_NAME} • Synthetic demo"
    return out


# ============================================================
# Data + modelling helpers
# ============================================================
def _safe_read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)
        if out[c].dtype == object:
            out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def _binary_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = df[c]
        if not np.issubdtype(s.dtype, np.number):
            continue
        vals = pd.unique(pd.to_numeric(s, errors="coerce").dropna())
        uniq = set(vals.tolist())
        if uniq.issubset({0.0, 1.0}) and len(uniq) == 2:
            cols.append(c)
    # Prefer common target name ordering
    if "event_30d" in cols:
        cols.remove("event_30d")
        cols.insert(0, "event_30d")
    return cols


def _suggest_target(df: pd.DataFrame) -> Optional[str]:
    if "event_30d" in df.columns:
        return "event_30d"
    bin_cols = _binary_columns(df)
    return bin_cols[0] if bin_cols else None


def _numeric_feature_columns(df: pd.DataFrame, exclude: Optional[str] = None) -> List[str]:
    cols = []
    for c in df.columns:
        if exclude and c == exclude:
            continue
        if np.issubdtype(df[c].dtype, np.number):
            cols.append(c)
    return cols


def load_csv(file_obj):
    if file_obj is None:
        snap = {"status": "no_dataset_loaded"}
        return (
            pd.DataFrame(),
            "<div class='muted'>Upload a synthetic CSV to begin.</div>",
            gr.update(choices=[], value=""),
            gr.update(choices=[], value=[]),
            gr.update(value=""),
            snap,
        )

    try:
        df = _coerce_numeric(_safe_read_csv(file_obj.name))
    except Exception as e:
        snap = {"status": "csv_load_failed", "error": str(e)}
        return (
            pd.DataFrame(),
            f"<div class='muted'><b>CSV load failed:</b> {str(e)}</div>",
            gr.update(choices=[], value=""),
            gr.update(choices=[], value=[]),
            gr.update(value=""),
            snap,
        )

    preview = df.head(25)

    suggested_target = _suggest_target(df)
    binary_targets = _binary_columns(df)

    # If no binary targets detected, still allow selecting any column (but training will block)
    target_choices = binary_targets if binary_targets else df.columns.tolist()

    # Features: numeric columns excluding suggested target (avoid leakage by default)
    feature_choices = _numeric_feature_columns(df, exclude=suggested_target)

    summary_html = f"""
    <div class="card" style="box-shadow:none; border-radius:14px;">
      <div style="font-weight:800; color:var(--nhs-blue-dark); margin-bottom:6px;">Dataset summary</div>
      <div class="muted">Rows: <b>{len(df):,}</b> • Columns: <b>{len(df.columns)}</b></div>
      <div class="muted">Suggested binary target: <b>{suggested_target or "None detected"}</b></div>
      <div class="muted" style="margin-top:8px;">{DEMO_DISCLAIMER}</div>
    </div>
    """

    snap = {
        "status": "dataset_loaded",
        "demo_disclaimer": DEMO_DISCLAIMER,
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "suggested_target": suggested_target,
        },
        "model": None,
        "metrics": None,
        "drivers": None,
        "flags": [],
        "warnings": [],
    }

    return (
        preview,
        summary_html,
        gr.update(choices=target_choices, value=(suggested_target or "")),
        gr.update(choices=feature_choices, value=[]),
        gr.update(value=""),  # clear status bar
        snap,
    )


def _build_kpi_html(model_type: str, auc: Optional[float], acc: Optional[float], rows: int, nfeat: int) -> str:
    def fmt(x):
        return "—" if x is None else f"{x:.3f}"

    return f"""
    <div class="kpi_grid">
      <div class="kpi"><div class="k">Model</div><div class="v">{model_type}</div></div>
      <div class="kpi"><div class="k">AUC</div><div class="v">{fmt(auc)}</div></div>
      <div class="kpi"><div class="k">Accuracy</div><div class="v">{fmt(acc)}</div></div>
      <div class="kpi"><div class="k">Rows / Features</div><div class="v">{rows:,} / {nfeat}</div></div>
    </div>
    """


def train_baseline(file_obj, target_col: str, feature_cols: List[str]):
    empty_df = pd.DataFrame(columns=["feature", "coef"])
    empty_fig = None

    if file_obj is None:
        snap = {"status": "no_dataset_loaded", "warnings": ["No CSV uploaded."]}
        return "", "Upload a CSV first.", empty_df, empty_fig, snap

    df = _coerce_numeric(_safe_read_csv(file_obj.name))

    if not target_col or target_col not in df.columns:
        snap = {"status": "invalid_target", "warnings": ["Select a valid binary target column."]}
        return "", "Select a valid binary target column.", empty_df, empty_fig, snap

    if not feature_cols:
        snap = {"status": "no_features", "warnings": ["Select at least one feature."]}
        return "", "Select at least one feature.", empty_df, empty_fig, snap

    # HARD BLOCK: target cannot be in features (prevents the exact screenshot issue)
    if target_col in feature_cols:
        snap = {
            "status": "leakage_blocked",
            "warnings": [f"Leakage blocked: target '{target_col}' cannot be used as a feature."],
            "flags": ["target_in_features"],
        }
        return "", f"Leakage blocked: remove **{target_col}** from features.", empty_df, empty_fig, snap

    # Prepare X/y and ensure binary
    y = pd.to_numeric(df[target_col], errors="coerce")
    y = y.dropna()
    df2 = df.loc[y.index].copy()
    y = pd.to_numeric(df2[target_col], errors="coerce")

    uniq = set(pd.unique(y.dropna()).tolist())
    if not uniq.issubset({0.0, 1.0}) or len(uniq) < 2:
        snap = {
            "status": "non_binary_target",
            "warnings": [f"Target must be binary with both classes present. Found: {sorted(list(uniq))}"],
        }
        return "", "Target must be binary (0/1) with both classes present.", empty_df, empty_fig, snap

    X = df2[feature_cols].copy()

    # Coerce numeric features
    dropped = []
    for c in list(X.columns):
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if X[c].isna().all():
            dropped.append(c)

    if dropped:
        X = X.drop(columns=dropped)
        feature_cols = [c for c in feature_cols if c not in dropped]

    if X.shape[1] == 0:
        snap = {"status": "no_numeric_features", "warnings": ["No usable numeric features."]}
        return "", "No usable numeric features after conversion.", empty_df, empty_fig, snap

    # Simple median fill
    X = X.apply(lambda col: col.fillna(col.median()) if np.issubdtype(col.dtype, np.number) else col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, probs))
    acc = float(accuracy_score(y_test, preds))

    clf = pipe.named_steps["clf"]
    coefs = clf.coef_.reshape(-1)
    coef_df = pd.DataFrame({"feature": X.columns.tolist(), "coef": coefs}).sort_values("coef", ascending=False).reset_index(drop=True)

    # ROC plot (smaller, exec-friendly)
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig = plt.figure(figsize=(5.6, 3.8), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("ROC Curve (Logistic Regression)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    # Leak/too-clean heuristic warning
    flags = []
    warnings = []
    if auc >= 0.95:
        flags.append("possible_leakage_or_too_clean_data")
        warnings.append(
            "AUC is extremely high. In real NHS data this is unusual; it may indicate synthetic rules are too clean "
            "or a feature leaks the target."
        )
    if dropped:
        warnings.append(f"Dropped unusable features: {', '.join(dropped)}")

    # Snapshot for AI commentary
    top_pos = coef_df.head(5).to_dict(orient="records")
    top_neg = coef_df.tail(5).sort_values("coef", ascending=True).to_dict(orient="records")
    pos_rate = float(y.mean())

    snap = {
        "status": "model_trained",
        "demo_disclaimer": DEMO_DISCLAIMER,
        "dataset": {
            "rows": int(len(df2)),
            "columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "target": target_col,
            "class_balance": {"positive_rate": pos_rate, "negative_rate": 1.0 - pos_rate},
        },
        "model": {
            "type": "Logistic Regression (explainable baseline)",
            "threshold": 0.5,
            "features": X.columns.tolist(),
        },
        "metrics": {"auc": auc, "accuracy": acc},
        "drivers": {"top_positive": top_pos, "top_negative": top_neg},
        "flags": flags,
        "warnings": warnings,
    }

    kpi_html = _build_kpi_html("Logistic Regression (baseline)", auc, acc, len(df2), X.shape[1])

    explanation_md = (
        "**What this model is doing (plain English)**\n"
        f"- Predicts the chance of **{target_col}=1** using the selected features.\n\n"
        "**How to read the metrics**\n"
        f"- **AUC ({auc:.3f})**: how well the model separates positives vs negatives across all cut-offs.\n"
        f"- **Accuracy ({acc:.3f})**: how often it gets it right at a 0.5 cut-off.\n\n"
        "**Coefficients (direction)**\n"
        "- Positive coefficient → higher value pushes predicted chance up.\n"
        "- Negative coefficient → higher value pushes predicted chance down.\n"
    )
    if warnings:
        explanation_md += "\n**Caveats detected**\n" + "\n".join([f"- {w}" for w in warnings])

    return kpi_html, explanation_md, coef_df.head(12), fig, snap


# ============================================================
# AI CALLBACKS
# ============================================================
def ai_run(mode: str, user_message: str, snapshot: Dict[str, Any], history: List[Dict[str, str]]):
    history = history or []
    snapshot = snapshot or {"status": "no_snapshot"}

    # If no trained model, still allow but it will say “not trained”
    if snapshot.get("status") != "model_trained":
        snapshot = dict(snapshot)
        snapshot.setdefault("warnings", [])
        snapshot["warnings"].append("No trained model yet. Train the baseline model to get full commentary.")

    answer = ask_llm_with_snapshot(user_message=user_message, mode=mode, snapshot=snapshot, history_messages=history)

    if mode == "qa":
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})
    else:
        history.append({"role": "assistant", "content": answer})

    return "", history


def ai_clear():
    return []


# ============================================================
# UI
# ============================================================
with gr.Blocks(css=CSS, title=APP_TITLE) as demo:
    gr.HTML(
        f"""
        <div id="topbar">
          <h1>{APP_TITLE}</h1>
          <div class="sub">
            AI partner for exploring a <b>synthetic</b> cardiac dataset — explainable modelling + plain-English commentary (demo stance, not clinical advice).
          </div>
          <div class="badges">
            <div class="badge"><span class="dot"></span>Synthetic-only guidance</div>
            <div class="badge"><span class="dot" style="background:#2D7D46;"></span>Plain-English explanations</div>
            <div class="badge"><span class="dot" style="background:#FFB81C;"></span>Executive-ready summaries</div>
          </div>
        </div>
        """
    )

    analysis_snapshot = gr.State({"status": "no_snapshot"})

    gr.Markdown("")

    with gr.Row():
        # LEFT
        with gr.Column(scale=6):
            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### Data (synthetic CSV)")
                gr.Markdown("<div class='muted'>Upload → preview → confirm target/features. No patient-identifiable data.</div>")

                file_in = gr.File(label="Upload synthetic CSV", file_types=[".csv"])

                preview_df = gr.Dataframe(
                    label="Preview (first 25 rows)",
                    interactive=False,
                    wrap=True,
                    elem_id="preview_df",
                )

                summary_html = gr.HTML("<div class='muted'>Upload a synthetic CSV to begin.</div>")
                status_bar = gr.Textbox(label="Status", interactive=False, placeholder="", value="")

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### Model controls (binary classification)")
                target_dd = gr.Dropdown(label="Target (binary 0/1)", choices=[], value="")
                features_ms = gr.Dropdown(label="Features (numeric)", choices=[], value=[], multiselect=True)
                train_btn = gr.Button("Train baseline model (Logistic Regression)", elem_classes=["primary"])

            with gr.Group(elem_classes=["card"]):
                gr.Markdown("### Results (exec view)")
                kpi_html = gr.HTML("")
                metrics_md = gr.Markdown("")

                with gr.Row():
                    coef_df = gr.Dataframe(label="Top drivers (coefficients)", interactive=False, wrap=True, elem_id="coef_df")
                    roc_plot = gr.Plot(label="ROC Curve")

            gr.HTML(f"<div class='footer_note'>{DEMO_DISCLAIMER}</div>")

        # RIGHT
        with gr.Column(scale=5):
            with gr.Group(elem_classes=["card"], elem_id="right_sticky"):
                gr.Markdown("### AI Analyst Commentary")
                gr.Markdown("<div class='muted'>Locked scope: explains the <b>current</b> model output, caveats, and next steps — not a general chatbot.</div>")

                with gr.Row():
                    exec_btn = gr.Button("Generate executive briefing")
                    risks_btn = gr.Button("Risks / caveats")
                    next_btn = gr.Button("What should we improve next?")

                ai_box = gr.Chatbot(label="Commentary", height=360)

                explain_btn = gr.Button("Explain current results", elem_id="explain_btn")
                user_q = gr.Textbox(label="Your question", placeholder="e.g., What does AUC mean? Why is troponin a driver?")
                send_btn = gr.Button("Send", elem_classes=["primary"])
                clear_btn = gr.Button("Clear commentary")

    # Wiring: CSV load
    file_in.change(
        load_csv,
        inputs=[file_in],
        outputs=[preview_df, summary_html, target_dd, features_ms, status_bar, analysis_snapshot],
    )

    # Wiring: Train model
    def do_train(file_obj, target, features):
        return train_baseline(file_obj, target, features)

    train_btn.click(
        do_train,
        inputs=[file_in, target_dd, features_ms],
        outputs=[kpi_html, metrics_md, coef_df, roc_plot, analysis_snapshot],
    )

    # AI wiring
    exec_btn.click(lambda snap, hist: ai_run("executive", "", snap, hist), inputs=[analysis_snapshot, ai_box], outputs=[user_q, ai_box])
    risks_btn.click(lambda snap, hist: ai_run("risks", "", snap, hist), inputs=[analysis_snapshot, ai_box], outputs=[user_q, ai_box])
    next_btn.click(lambda snap, hist: ai_run("next", "", snap, hist), inputs=[analysis_snapshot, ai_box], outputs=[user_q, ai_box])
    explain_btn.click(lambda snap, hist: ai_run("executive", "", snap, hist), inputs=[analysis_snapshot, ai_box], outputs=[user_q, ai_box])

    send_btn.click(lambda q, snap, hist: ai_run("qa", q, snap, hist), inputs=[user_q, analysis_snapshot, ai_box], outputs=[user_q, ai_box])
    user_q.submit(lambda q, snap, hist: ai_run("qa", q, snap, hist), inputs=[user_q, analysis_snapshot, ai_box], outputs=[user_q, ai_box])

    clear_btn.click(ai_clear, inputs=None, outputs=ai_box)

if __name__ == "__main__":
    demo.launch()
