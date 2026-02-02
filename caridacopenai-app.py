import os
import time
import gradio as gr
from openai import OpenAI

# =============================================================
# Clinical ML Tutor (Executive Demo - NHS Blue theme)
# - Synthetic-only guidance (no patient-identifiable info)
# - Plain-English explanations
# - WORKING Gradio 6.x Chatbot messages format preserved
# =============================================================

# -----------------------------
# Config
# -----------------------------
APP_TITLE = "Clinical ML Tutor"
APP_SUBTITLE = "AI partner for exploring a synthetic cardiac dataset"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")

SYSTEM_PROMPT = (
    "You are a calm, friendly clinical data tutor.\n"
    "Explain in plain English with short steps.\n"
    "Avoid jargon. If you use a technical term, define it immediately.\n"
    "Assume the dataset is synthetic and for learning.\n"
    "Keep answers concise and practical.\n"
)

# Uses OPENAI_API_KEY from environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# NHS-style CSS (high contrast, executive readable)
# -----------------------------
NHS_CSS = """
:root{
  /* NHS Blue palette */
  --nhs-blue: #005EB8;
  --nhs-dark: #003087;
  --nhs-bright: #0072CE;
  --nhs-bg: #F0F4F5;          /* light NHS-like background */
  --panel: #FFFFFF;
  --panel2: #F7F9FA;
  --border: rgba(0,0,0,0.10);

  /* Text */
  --text: #0B0C0C;            /* NHS-ish near-black */
  --muted: rgba(11,12,12,0.70);
  --muted2: rgba(11,12,12,0.55);

  /* Accents */
  --accent: #005EB8;
  --accent2: #0072CE;
  --good: #007F3B;
  --warn: #FFB81C;

  --radius: 16px;
  --shadow: 0 14px 40px rgba(0,0,0,0.12);
}

/* Whole page background */
.gradio-container{
  background:
    radial-gradient(900px 600px at 20% 0%, rgba(0,94,184,0.22), transparent 55%),
    radial-gradient(900px 600px at 90% 10%, rgba(0,114,206,0.18), transparent 60%),
    linear-gradient(180deg, #E9F2FB, var(--nhs-bg)) !important;
  color: var(--text) !important;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
}

/* Header card */
#header-card{
  border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(0,94,184,0.10), rgba(0,114,206,0.08));
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 18px 18px;
  margin-bottom: 14px;
}

#header-title{
  font-size: 28px;
  font-weight: 900;
  letter-spacing: 0.2px;
  margin: 0 0 6px 0;
  color: var(--nhs-dark);
}

#header-subtitle{
  font-size: 14px;
  color: var(--muted);
  margin: 0 0 10px 0;
  line-height: 1.4;
}

/* Badges */
.badge-row{
  display:flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 10px;
}

.badge{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.80);
  font-size: 12px;
  color: var(--muted);
}

.badge-dot{
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--accent);
  box-shadow: 0 0 12px rgba(0,94,184,0.25);
}

/* Panels */
.panel{
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--panel);
  box-shadow: var(--shadow);
  padding: 14px;
}

.panel h3{
  margin: 0 0 6px 0;
  font-size: 14px;
  font-weight: 800;
  color: var(--nhs-dark);
}

.small-muted{
  color: var(--muted2);
  font-size: 12px;
}

/* KPI grid */
.kpi-grid{
  display:grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.kpi{
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--panel2);
  padding: 10px 12px;
}

.kpi .label{
  font-size: 11px;
  color: var(--muted2);
  margin-bottom: 4px;
}

.kpi .value{
  font-size: 15px;
  font-weight: 800;
  color: var(--text);
}

.kpi .hint{
  font-size: 11px;
  color: var(--muted);
  margin-top: 4px;
}

/* Inputs: LIGHT backgrounds, DARK text */
.gr-textbox textarea,
.gr-textbox input,
.gr-dropdown select{
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  background: #FFFFFF !important;
  color: var(--text) !important;
}

/* Fix placeholder contrast */
.gr-textbox textarea::placeholder,
.gr-textbox input::placeholder{
  color: rgba(11,12,12,0.45) !important;
}

/* Buttons */
.gr-button{
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  background: #FFFFFF !important;
  color: var(--nhs-dark) !important;
  font-weight: 900 !important;
  padding: 10px 12px !important;
}

.gr-button:hover{
  background: #F4F8FB !important;
}

/* Primary NHS Blue button */
#primary-btn{
  background: var(--nhs-blue) !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(0,0,0,0.12) !important;
}

#primary-btn:hover{
  background: var(--nhs-dark) !important;
}

/* Chatbot container: LIGHT */
.gr-chatbot{
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: #FFFFFF !important;
  box-shadow: var(--shadow) !important;
}

/* Chat bubbles - readable */
.gr-chatbot .message{
  border-radius: 14px !important;
}

/* Make assistant/user content readable even if Gradio injects odd defaults */
.gr-chatbot *{
  color: var(--text) !important;
}

/* Footer note */
.footer-note{
  margin-top: 10px;
  color: var(--muted2);
  font-size: 12px;
}

@media (max-width: 1100px){
  .kpi-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""

# -----------------------------
# OpenAI response extraction
# -----------------------------
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


# -----------------------------
# LLM call (expects messages history)
# -----------------------------
def ask_llm(user_message: str, context: str, history_messages: list[dict]) -> str:
    if not user_message or not user_message.strip():
        return "Type a question (e.g., 'Explain troponin in one line')."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context and context.strip():
        messages.append(
            {
                "role": "user",
                "content": "CONTEXT (what I'm doing right now):\n" + context.strip(),
            }
        )

    # history_messages MUST be: [{"role": "...", "content": "..."}, ...]
    for m in history_messages or []:
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": str(m["content"])})

    messages.append({"role": "user", "content": user_message.strip()})

    resp = client.responses.create(
        model=MODEL_NAME,
        input=messages,
    )
    return _extract_output_text(resp)


# -----------------------------
# Gradio callbacks (messages format)
# -----------------------------
def chat(user_message, context, history):
    history = history or []

    t0 = time.time()
    answer = ask_llm(user_message, context, history)
    elapsed_ms = int((time.time() - t0) * 1000)

    # Keep this subtle and readable (exec demo)
    answer = f"{answer}\n\n—\n*Response time: {elapsed_ms} ms • Model: {MODEL_NAME} • Synthetic demo*"

    # Append in MESSAGES FORMAT (Gradio 6.x)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})

    return "", history


def clear_chat():
    return []


def set_preset(preset_name: str):
    presets = {
        "Executive overview (what this is)": (
            "Goal: demonstrate an AI assistant supporting clinical analytics learning using synthetic data.\n"
            "Audience: executive / leadership team.\n"
            "Rule: be concise and focus on value, safety, governance.\n"
            "Ask: explain what this tool does, why it’s safe, and how it scales in NHS contexts."
        ),
        "Learn the dataset headers": (
            "Headers: age, sex, systolic_bp, heart_rate, troponin, ldl, egfr, crp, chest_pain, ecg_st, diabetes, smoking, event_30d, ntprobnp\n"
            "Current step: learning what each header means.\n"
            "Rule: one line per header + a plain-English note about why it matters."
        ),
        "Chart interpretation": (
            "Current step: I have a chart (histogram / boxplot / scatter) and I want help interpreting it.\n"
            "Rule: explain what the chart shows, what to check next, and what pitfalls exist."
        ),
        "Model thinking (beginner)": (
            "Current step: I want to predict event_30d from a few variables.\n"
            "Rule: explain simply what a model is, what target means, and how to evaluate it without jargon."
        ),
    }
    return presets.get(preset_name, "")


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title=APP_TITLE, css=NHS_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        f"""
        <div id="header-card">
          <div id="header-title">{APP_TITLE}</div>
          <div id="header-subtitle">
            <b>{APP_SUBTITLE}</b>. Designed for <b>synthetic demo data</b>, with clear explanations for non-ML audiences.
          </div>
          <div class="badge-row">
            <span class="badge"><span class="badge-dot"></span> Synthetic-only guidance</span>
            <span class="badge"><span class="badge-dot" style="background: var(--accent2)"></span> Plain-English explanations</span>
            <span class="badge"><span class="badge-dot" style="background: var(--warn)"></span> Executive-ready summaries</span>
          </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### Context (what you’re doing)")
                preset = gr.Dropdown(
                    label="Quick presets",
                    choices=[
                        "Executive overview (what this is)",
                        "Learn the dataset headers",
                        "Chart interpretation",
                        "Model thinking (beginner)",
                    ],
                    value="Learn the dataset headers",
                )

                context = gr.Textbox(
                    label="Paste what you're doing / headers / plot description",
                    placeholder=(
                        "Headers: age, sex, systolic_bp, heart_rate, troponin, ldl, egfr, crp, chest_pain, ecg_st, "
                        "diabetes, smoking, event_30d, ntprobnp\n"
                        "Current step: learning what each header means + basic charts\n"
                        "Goal: explain things simply as I go\n"
                        "Rule: keep answers short; define any technical term immediately"
                    ),
                    lines=10,
                )

                preset.change(set_preset, inputs=preset, outputs=context)

                gr.Markdown(
                    "<div class='small-muted'>Tip: Keep this context synthetic (no patient-identifiable info).</div>"
                )

            gr.HTML(
                """
                <div class="panel">
                  <h3>Demo value (for executives)</h3>
                  <div class="kpi-grid">
                    <div class="kpi">
                      <div class="label">Use-case</div>
                      <div class="value">Clinical analytics learning</div>
                      <div class="hint">Explains fields, charts, & model basics</div>
                    </div>
                    <div class="kpi">
                      <div class="label">Safety stance</div>
                      <div class="value">Synthetic data</div>
                      <div class="hint">No patient-identifiable inputs</div>
                    </div>
                    <div class="kpi">
                      <div class="label">Outcome</div>
                      <div class="value">Faster capability uplift</div>
                      <div class="hint">Less reliance on specialist time</div>
                    </div>
                    <div class="kpi">
                      <div class="label">Scalability</div>
                      <div class="value">Reusable pattern</div>
                      <div class="hint">Can adapt to other datasets & teams</div>
                    </div>
                  </div>
                  <div class="footer-note">
                    This is a demo assistant. It does not provide clinical advice; it supports learning with synthetic data.
                  </div>
                </div>
                """
            )

        with gr.Column(scale=1):
            # Keep the WORKING Chatbot history format (messages list of dicts)
            chatbot = gr.Chatbot(label="Assistant", height=520)

            msg = gr.Textbox(
                label="Your question",
                placeholder="e.g., Explain troponin in one line",
            )

            with gr.Row():
                send = gr.Button("Send", elem_id="primary-btn")
                clear = gr.Button("Clear chat")

            gr.Markdown(
                "<div class='small-muted'>Tip: Ask one question at a time. "
                "Examples: “Explain egfr simply”, “What does event_30d mean?”, “What chart should I make next?”</div>"
            )

    # Wiring (keep working behavior)
    send.click(chat, inputs=[msg, context, chatbot], outputs=[msg, chatbot])
    msg.submit(chat, inputs=[msg, context, chatbot], outputs=[msg, chatbot])
    clear.click(clear_chat, inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
