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
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")

SYSTEM_PROMPT = (
    "You are a calm, senior clinical analytics tutor.\n"
    "Explain in plain English with short steps.\n"
    "Avoid jargon. If you use a technical term, define it immediately.\n"
    "Assume the dataset is synthetic and for learning.\n"
    "Write answers suitable for an NHS executive demo.\n"
    "If results look unrealistically strong, explicitly warn about leakage or synthetic bias.\n"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# =========================================================
# OPENAI HELPERS
# =========================================================
def extract_output(resp) -> str:
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


def ask_llm(message: str, auto_context: str, history: list[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if auto_context and auto_context.strip():
        messages.append({"role": "user", "content": auto_context.strip()})

    for h in history or []:
        if isinstance(h, dict) and "role" in h and "content" in h:
            messages.append({"role": h["role"], "content": str(h["content"])})

    messages.append({"role": "user", "content": message.strip()})

    resp = client.responses.create(model=MODEL_NAME, input=messages)
    return extract_output(resp)


# =========================================================
# MODEL PIPELINE
# =========================================================
def build_pipeline(df: pd.DataFrame, features: list[str]):
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline([
        ("prep", pre),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    return pipe, num_cols, cat_cols


def fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"<img src='data:image/png;base64,{b64}' style='max-width:100%; height:auto; border-radius:14px; border:1px solid #D6E3F3;' />"


# =========================================================
# CSV LOADING
# =========================================================
def load_csv(file):
    if file is None:
        return (
            None,
            pd.DataFrame(),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]),
            "Upload a **synthetic CSV** to begin.",
            "",
            ""
        )

    df = pd.read_csv(file.name)
    df.columns = [c.strip() for c in df.columns]

    suggested_target = "event_30d" if "event_30d" in df.columns else df.columns[-1]
    cols = list(df.columns)

    # default features = everything except target (cap to first 10 to keep UI neat)
    default_features = [c for c in cols if c != suggested_target][:10]

    summary = (
        f"### Dataset summary\n"
        f"- Rows: **{len(df)}**\n"
        f"- Columns: **{len(cols)}**\n"
        f"- Suggested target: **{suggested_target}**\n"
        f"\n"
        f"**Demo stance:** synthetic-only (no patient-identifiable input)."
    )

    preview = df.head(25)

    auto_context = (
        "DEMO CONTEXT (synthetic dataset):\n"
        f"- Rows: {len(df)}\n"
        f"- Columns: {len(cols)}\n"
        f"- Column names: {', '.join(cols)}\n"
        "\n"
        "You are assisting an NHS exec demo. Keep answers short, clear, and focused on interpretability.\n"
    )

    return (
        df,
        preview,
        gr.update(choices=cols, value=suggested_target),
        gr.update(choices=[c for c in cols if c != suggested_target], value=default_features),
        summary,
        auto_context,
        ""
    )


# =========================================================
# TRAIN MODEL
# =========================================================
def train_model(df, target, features):
    if df is None or df.empty:
        return None, "### Executive KPIs\nUpload a CSV first.", "—", "", "No data loaded.", ""

    if not target or target not in df.columns:
        return None, "### Executive KPIs\nSelect a valid target.", "—", "", "Invalid target.", ""

    features = features or []
    features = [f for f in features if f in df.columns and f != target]

    if not features:
        return None, "### Executive KPIs\nSelect at least one feature.", "—", "", "No features selected.", ""

    y = df[target]
    uniq = pd.Series(y).dropna().unique()
    if len(uniq) != 2:
        return None, "### Executive KPIs\nTarget must be binary (0/1).", "—", "", f"Target has {len(uniq)} unique values.", ""

    X = df[features].copy()
    y = df[target].astype(int).copy()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe, num_cols, cat_cols = build_pipeline(df, features)
    pipe.fit(Xtr, ytr)

    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(yte, proba))
    acc = float(accuracy_score(yte, pred))

    # feature names for coefficients
    feat_names = []
    feat_names.extend(num_cols)

    if cat_cols:
        ohe = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        feat_names.extend(list(ohe.get_feature_names_out(cat_cols)))

    coefs = pipe.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    coef_df["abs"] = coef_df["coef"].abs()
    top = coef_df.sort_values("abs", ascending=False).head(12).drop(columns=["abs"])

    coef_md = top.to_markdown(index=False)

    # ROC
    fpr, tpr, _ = roc_curve(yte, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Logistic Regression)")
    roc_html = fig_to_html(fig)

    kpis = (
        "### Executive KPIs\n"
        f"- **Model:** Logistic Regression (explainable baseline)\n"
        f"- **AUC:** `{auc:.3f}` (higher = better separation)\n"
        f"- **Accuracy:** `{acc:.3f}` (correct at 0.5 threshold)\n"
        f"- **Rows:** `{len(df)}`\n"
        f"- **Features:** `{len(features)}`\n"
    )

    auto_context = (
        "CURRENT RESULTS (synthetic demo):\n"
        f"- Target: {target}\n"
        f"- Features: {', '.join(features)}\n"
        f"- AUC: {auc:.3f}\n"
        f"- Accuracy: {acc:.3f}\n"
        "- Top coefficients (log-odds):\n"
        + "\n".join([f"  - {r.feature}: {float(r.coef):.3f}" for r in top.itertuples(index=False)])
        + "\n"
    )

    if auc > 0.95:
        auto_context += (
            "\n⚠️ NOTE: AUC is very high. On real data this can indicate leakage or overly-easy signal.\n"
            "For synthetic datasets, it may simply be generated with strong separability.\n"
        )

    return pipe, kpis, coef_md, roc_html, auto_context, ""


# =========================================================
# ASSISTANT CALLBACKS
# =========================================================
def chat(user_message, history, auto_context):
    history = history or []
    if not user_message or not user_message.strip():
        return "", history

    answer = ask_llm(user_message.strip(), auto_context or "", history)
    history.append({"role": "user", "content": user_message.strip()})
    history.append({"role": "assistant", "content": answer})
    return "", history


def explain_current_results(history, auto_context):
    history = history or []
    prompt = (
        "Explain the current results for an executive demo.\n"
        "Format:\n"
        "1) What the model is doing (1–2 lines)\n"
        "2) What AUC and accuracy mean (plain English)\n"
        "3) Top 3 drivers (coefficients) and what they suggest\n"
        "4) Risks/caveats (synthetic limits, leakage warning)\n"
        "5) Slide-ready summary paragraph\n"
    )
    answer = ask_llm(prompt, auto_context or "", history)
    history.append({"role": "assistant", "content": answer})
    return history


def clear_chat():
    return []


# =========================================================
# UI THEME (NHS BLUE, WHITE CARDS, FIXED TABLE HEIGHT)
# =========================================================
CSS = """
:root{
  --nhs-blue:#005EB8;
  --nhs-blue-dark:#003B73;
  --nhs-bg:#EAF2FF;
  --card:#FFFFFF;
  --ink:#0B1F33;
  --muted:#516173;
  --border:#D6E3F3;
}

.gradio-container{
  background: linear-gradient(180deg, var(--nhs-bg), #ffffff 55%);
}

#hero{
  background: linear-gradient(90deg, rgba(0,94,184,0.10), rgba(0,94,184,0.03));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 10px;
}

#hero h2{ margin:0; color:var(--nhs-blue); font-weight:800; }
#hero p{ margin:6px 0 0 0; color:var(--muted); }

.card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px;
}

.small-muted{
  color: var(--muted);
  font-size: 12px;
}

button.primary{
  background: var(--nhs-blue) !important;
  border: 1px solid var(--nhs-blue) !important;
}
button.primary:hover{
  background: var(--nhs-blue-dark) !important;
  border: 1px solid var(--nhs-blue-dark) !important;
}

/* ✅ REAL fix: control Dataframe scroll/height via elem_id */
#df_preview .table-wrap{
  max-height: 320px;
  overflow: auto;
  border-radius: 12px;
  border: 1px solid var(--border);
}

/* tighten excessive spacing */
.block.svelte-1gfkn6j{ padding-top: 6px; }
"""

with gr.Blocks(title="Clinical ML Tutor – Exec Demo", css=CSS) as demo:
    gr.HTML(
        """
        <div id="hero">
          <h2>Clinical ML Tutor</h2>
          <p><b>AI partner</b> for exploring a <b>synthetic cardiac dataset</b> with explainable modelling + plain-English guidance (demo stance, not clinical advice).</p>
        </div>
        """
    )

    df_state = gr.State(None)
    model_state = gr.State(None)
    auto_ctx_state = gr.State("")

    with gr.Row(equal_height=True):
        # LEFT: Regression demo
        with gr.Column(scale=1, min_width=520):
            gr.HTML("<div class='card'>")
            gr.Markdown("### Cardiac regression (demo)")
            gr.Markdown("<span class='small-muted'>Upload a synthetic CSV → select target/features → train an explainable baseline model.</span>")
            upload = gr.File(label="Upload synthetic CSV")

            with gr.Accordion("Dataset preview (first 25 rows)", open=True):
                df_preview = gr.Dataframe(elem_id="df_preview")

            summary_md = gr.Markdown("Upload a **synthetic CSV** to begin.")
            gr.HTML("</div>")

            gr.HTML("<div class='card' style='margin-top:12px;'>")
            gr.Markdown("### Model controls")
            target = gr.Dropdown(label="Target (binary)", choices=[], value=None, interactive=True)
            features = gr.Dropdown(label="Features", choices=[], value=[], multiselect=True, interactive=True)
            train_btn = gr.Button("Train regression model", variant="primary")
            error_md = gr.Markdown("")
            gr.HTML("</div>")

            gr.HTML("<div class='card' style='margin-top:12px;'>")
            kpis_md = gr.Markdown("### Executive KPIs\nTrain the model to populate KPIs.")
            gr.Markdown("### Coefficients (top signals)")
            coef_md = gr.Markdown("—")
            gr.Markdown("### ROC Curve")
            roc_html = gr.HTML("")
            gr.HTML("</div>")

        # RIGHT: Assistant
        with gr.Column(scale=1, min_width=520):
            gr.HTML("<div class='card'>")
            gr.Markdown("### AI assistant")
            gr.Markdown("<span class='small-muted'>Ask questions and use <b>Explain current results</b> to generate exec-ready interpretation of the model output.</span>")
            chatbot = gr.Chatbot(height=520)
            explain_btn = gr.Button("Explain current results", variant="primary")
            msg = gr.Textbox(label="Your question", placeholder="e.g., What does AUC mean? Which variables drive risk?")
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear chat")
            gr.HTML("</div>")

    # events
    upload.change(
        load_csv,
        inputs=[upload],
        outputs=[df_state, df_preview, target, features, summary_md, auto_ctx_state, error_md],
    )

    train_btn.click(
        train_model,
        inputs=[df_state, target, features],
        outputs=[model_state, kpis_md, coef_md, roc_html, auto_ctx_state, error_md],
    )

    send_btn.click(
        chat,
        inputs=[msg, chatbot, auto_ctx_state],
        outputs=[msg, chatbot],
    )
    msg.submit(
        chat,
        inputs=[msg, chatbot, auto_ctx_state],
        outputs=[msg, chatbot],
    )

    explain_btn.click(
        explain_current_results,
        inputs=[chatbot, auto_ctx_state],
        outputs=[chatbot],
    )

    clear_btn.click(
        clear_chat,
        inputs=None,
        outputs=chatbot,
    )

if __name__ == "__main__":
    demo.launch()
