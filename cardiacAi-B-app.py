import os
import io
import base64
import numpy as np
import pandas as pd
import gradio as gr

from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


# =========================================================
# Config (KEEP WORKING PARTS)
# =========================================================
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")

SYSTEM_PROMPT = (
    "You are a calm, friendly clinical data tutor.\n"
    "Explain in plain English with short steps.\n"
    "Avoid jargon. If you use a technical term, define it immediately.\n"
    "Assume the dataset is synthetic and for learning.\n"
    "If the user asks what to do next, suggest a single sensible next step.\n"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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


def ask_llm(user_message: str, context: str, history_messages: list[dict]) -> str:
    if not user_message or not user_message.strip():
        return "Type a question (e.g., 'Explain troponin in one line')."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context and context.strip():
        messages.append({"role": "user", "content": "CONTEXT (what I'm doing right now):\n" + context.strip()})

    for m in history_messages or []:
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": str(m["content"])})

    messages.append({"role": "user", "content": user_message.strip()})

    resp = client.responses.create(
        model=MODEL_NAME,
        input=messages,
    )
    return _extract_output_text(resp)


# =========================================================
# Helpers for visuals + formatting
# =========================================================
def _fig_to_html_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:12px; border:1px solid #d9e2f1;" />'


def _safe_markdown(s: str) -> str:
    return s if s else ""


# =========================================================
# Regression pipeline builder
# =========================================================
def build_logreg_pipeline(df: pd.DataFrame, target: str, feature_cols: list[str]):
    X = df[feature_cols].copy()
    y = df[target].copy()

    # Identify numeric vs categorical
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Note: keep it stable + fast for demo
    model = LogisticRegression(max_iter=2000, n_jobs=None)

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    return pipe, numeric_cols, categorical_cols


def get_feature_names_from_pipeline(pipe: Pipeline, numeric_cols, categorical_cols):
    # After fitting, pull out one-hot feature names
    prep = pipe.named_steps["prep"]
    feature_names = []

    if numeric_cols:
        feature_names.extend(numeric_cols)

    if categorical_cols:
        # Find the categorical transformer and its onehot encoder
        for name, transformer, cols in prep.transformers_:
            if name == "cat":
                ohe = transformer.named_steps["onehot"]
                ohe_names = list(ohe.get_feature_names_out(cols))
                feature_names.extend(ohe_names)

    return feature_names


# =========================================================
# Gradio callbacks: data loading + training
# =========================================================
def load_csv(file):
    if file is None:
        return (
            None,
            gr.update(value=None, choices=[]),
            gr.update(value=[], choices=[]),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
        )

    df = pd.read_csv(file.name)

    # Basic cleaning: strip column names
    df.columns = [c.strip() for c in df.columns]

    # Preview
    preview = df.head(25).copy()
    preview_md = "### Preview (first 25 rows)\n\n" + preview.to_markdown(index=False)

    # Suggest default target if present
    cols = list(df.columns)
    default_target = "event_30d" if "event_30d" in cols else cols[-1]

    # Suggest default features: everything except target (limit for UI sanity)
    suggested_features = [c for c in cols if c != default_target]
    suggested_features = suggested_features[:12]  # keep UI tidy

    # Data summary
    summary = [
        f"**Rows:** {len(df):,}",
        f"**Columns:** {len(df.columns):,}",
        f"**Suggested target:** `{default_target}`",
    ]
    summary_md = "### Dataset summary\n\n" + "\n\n".join(summary)

    return (
        df,
        gr.update(value=default_target, choices=cols),
        gr.update(value=suggested_features, choices=cols),
        preview_md,
        summary_md,
        "",
        "",
    )


def train_model(df: pd.DataFrame, target: str, features: list[str]):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None, "### Model\n\nUpload a CSV first.", "### Metrics\n\n—", "### Coefficients (top signals)\n\n—", "—", ""

    if not target or target not in df.columns:
        return None, "### Model\n\nPick a valid target.", "### Metrics\n\n—", "### Coefficients (top signals)\n\n—", "—", ""

    if not features:
        return None, "### Model\n\nPick at least 1 feature.", "### Metrics\n\n—", "### Coefficients (top signals)\n\n—", "—", ""

    if target in features:
        features = [c for c in features if c != target]

    # Ensure target is binary-ish
    y = df[target]
    unique_vals = pd.Series(y.dropna().unique()).tolist()
    if len(unique_vals) > 2:
        return None, (
            "### Model\n\nTarget doesn't look binary.\n\n"
            f"Target `{target}` has **{len(unique_vals)}** unique values.\n\n"
            "For this demo we expect 0/1 (e.g., `event_30d`)."
        ), "### Metrics\n\n—", "### Coefficients (top signals)\n\n—", "—", ""

    # Split
    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(pd.Series(y).value_counts()) > 1 else None
    )

    pipe, num_cols, cat_cols = build_logreg_pipeline(df, target, features)

    # Fit
    pipe.fit(X_train, y_train)

    # Predict
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, proba) if len(pd.Series(y_test).unique()) > 1 else float("nan")
    acc = accuracy_score(y_test, pred)

    metrics_md = (
        "### Metrics\n\n"
        f"- **AUC (how well it separates 0 vs 1):** `{auc:.3f}`\n"
        f"- **Accuracy (at 0.5 threshold):** `{acc:.3f}`\n"
        f"- **Test set size:** `{len(y_test):,}` rows\n"
    )

    # Coefficients table
    model = pipe.named_steps["model"]
    feature_names = get_feature_names_from_pipeline(pipe, num_cols, cat_cols)

    coefs = model.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df["abs"] = coef_df["coef"].abs()
    top = coef_df.sort_values("abs", ascending=False).head(12).drop(columns=["abs"])

    # Friendly explanation
    coef_md = "### Coefficients (top signals)\n\n"
    coef_md += (
        "Bigger **absolute** values mean the model leans on that feature more.\n\n"
        "Positive = pushes risk **up** (towards event=1), negative = pushes risk **down**.\n\n"
    )
    coef_md += top.to_markdown(index=False)

    # ROC plot
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if len(pd.Series(y_test).unique()) > 1:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title("ROC Curve (Logistic Regression)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
    else:
        ax.text(0.5, 0.5, "ROC not available (only 1 class in test set)", ha="center", va="center")
        ax.set_axis_off()

    roc_html = _fig_to_html_img(fig)
    plt.close(fig)

    # Model summary for the assistant (auto-context)
    model_context = (
        "MODEL CONTEXT (auto):\n"
        f"- Target: {target}\n"
        f"- Features: {', '.join(features)}\n"
        f"- AUC: {auc:.3f}\n"
        f"- Accuracy: {acc:.3f}\n"
        f"- Top coefficients:\n"
    )
    for _, r in top.iterrows():
        model_context += f"  - {r['feature']}: {r['coef']:.4f}\n"

    model_md = (
        "### Model\n\n"
        "**Logistic Regression** trained on your selected features.\n\n"
        "This is a simple, explainable baseline model that’s good for demonstrations and learning.\n"
    )

    return pipe, model_md, metrics_md, coef_md, roc_html, model_context


# =========================================================
# Assistant chat (right panel) – shares context
# =========================================================
def chat(user_message, user_context, history, auto_context):
    history = history or []

    combined_context_parts = []
    if auto_context and auto_context.strip():
        combined_context_parts.append(auto_context.strip())
    if user_context and user_context.strip():
        combined_context_parts.append("USER CONTEXT:\n" + user_context.strip())

    combined_context = "\n\n".join(combined_context_parts).strip()

    answer = ask_llm(user_message, combined_context, history)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})

    return "", history


def clear_chat():
    return []


# =========================================================
# Executive-ready look (NHS Blue, high contrast)
# =========================================================
CSS = """
:root {
  --nhs-blue: #005EB8;
  --nhs-dark: #003087;
  --bg: #EAF2FB;
  --card: #FFFFFF;
  --text: #0B1F33;
  --muted: #5B6B7C;
  --border: #D6E2F2;
}

body, .gradio-container {
  background: radial-gradient(1200px 700px at 15% 0%, #F4FAFF 0%, var(--bg) 55%, #E6F0FF 100%) !important;
  color: var(--text) !important;
}

#hero {
  background: linear-gradient(135deg, rgba(0,94,184,0.10), rgba(0,48,135,0.06));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  border-radius: 999px;
  border: 1px solid var(--border);
  padding: 6px 10px;
  background: #fff;
  color: var(--text);
  font-size: 12px;
  margin-right: 8px;
}

.badge-dot {
  width: 9px;
  height: 9px;
  border-radius: 999px;
  background: var(--nhs-blue);
  display: inline-block;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.06);
}

.section-title {
  font-weight: 800;
  color: var(--nhs-dark);
  margin: 0 0 8px 0;
}

.small-muted {
  color: var(--muted);
  font-size: 12px;
}

button.primary {
  background: var(--nhs-blue) !important;
  color: #fff !important;
  border-radius: 12px !important;
  border: 0 !important;
  font-weight: 700 !important;
}

button.secondary {
  background: #fff !important;
  color: var(--nhs-blue) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}
"""


# =========================================================
# UI: Single page, side-by-side
# =========================================================
with gr.Blocks(title="Clinical ML Tutor + Cardiac Regression Demo", css=CSS) as demo:
    gr.HTML(
        """
        <div id="hero">
          <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap;">
            <div>
              <div style="font-size:28px; font-weight:900; color:#003087; line-height:1.1;">
                Clinical ML Tutor
              </div>
              <div style="margin-top:6px; color:#0B1F33; font-weight:600;">
                AI partner for exploring a <b>synthetic cardiac dataset</b> — with explainable regression + plain-English guidance.
              </div>
              <div style="margin-top:10px;">
                <span class="badge"><span class="badge-dot"></span> Synthetic-only guidance</span>
                <span class="badge"><span class="badge-dot" style="background:#003087;"></span> Plain-English explanations</span>
                <span class="badge"><span class="badge-dot" style="background:#FFB81C;"></span> Executive-ready summaries</span>
              </div>
            </div>
            <div style="text-align:right; min-width:280px;">
              <div style="color:#5B6B7C; font-size:12px;">Demo stance</div>
              <div style="font-weight:800; color:#003087;">Learning & capability uplift (not clinical advice)</div>
              <div class="small-muted" style="margin-top:6px;">Model: <b>Logistic Regression</b> + LLM tutor</div>
            </div>
          </div>
        </div>
        """
    )

    # Shared state
    df_state = gr.State(value=None)
    model_state = gr.State(value=None)
    auto_context_state = gr.State(value="")

    with gr.Row():
        # LEFT: Regression app
        with gr.Column(scale=1):
            gr.HTML('<div class="card">')
            gr.Markdown("### Cardiac regression (demo)\nUpload a synthetic CSV, train a simple explainable model, and review key signals.")
            upload = gr.File(label="Upload synthetic CSV", file_types=[".csv"])

            dataset_summary = gr.Markdown("")
            preview_md = gr.Markdown("")

            target = gr.Dropdown(label="Target (binary, e.g. event_30d)", choices=[], value=None)
            features = gr.Dropdown(label="Features", choices=[], multiselect=True, value=[])

            train_btn = gr.Button("Train regression model", elem_classes=["primary"])

            model_md = gr.Markdown("")
            metrics_md = gr.Markdown("")
            coef_md = gr.Markdown("")

            gr.Markdown("### ROC Curve")
            roc_plot_html = gr.HTML("—")

            gr.HTML("</div>")

        # RIGHT: Assistant
        with gr.Column(scale=1):
            gr.HTML('<div class="card">')
            gr.Markdown("### AI assistant\nAsk questions while you explore the dataset and the regression output.")
            user_context = gr.Textbox(
                label="Context (optional)",
                placeholder="Example: I'm learning what each header means, and I just trained a regression model. Explain results simply.",
                lines=4,
            )

            chatbot = gr.Chatbot(label="Assistant", height=520)
            msg = gr.Textbox(label="Your question", placeholder="e.g., What does AUC mean in simple terms?")
            with gr.Row():
                send = gr.Button("Send", elem_classes=["primary"])
                clear = gr.Button("Clear chat", elem_classes=["secondary"])

            gr.Markdown(
                "<div class='small-muted'>Tip: the assistant automatically receives your model’s target, features, and metrics as context.</div>"
            )
            gr.HTML("</div>")

    # Wire: load CSV
    upload.change(
        fn=load_csv,
        inputs=[upload],
        outputs=[df_state, target, features, preview_md, dataset_summary, model_md, metrics_md],
    )

    # Wire: train model
    train_btn.click(
        fn=train_model,
        inputs=[df_state, target, features],
        outputs=[model_state, model_md, metrics_md, coef_md, roc_plot_html, auto_context_state],
    )

    # Wire: assistant chat (uses auto_context_state)
    send.click(
        fn=chat,
        inputs=[msg, user_context, chatbot, auto_context_state],
        outputs=[msg, chatbot],
    )
    msg.submit(
        fn=chat,
        inputs=[msg, user_context, chatbot, auto_context_state],
        outputs=[msg, chatbot],
    )
    clear.click(
        fn=clear_chat,
        inputs=None,
        outputs=chatbot,
    )

    gr.Markdown(
        "<div class='small-muted' style='margin-top:10px;'>"
        "Safety: Synthetic demo only. No patient-identifiable input. This tool supports learning and analytics capability uplift.</div>"
    )

if __name__ == "__main__":
    demo.launch()
