import os
import io
import base64
import math
from datetime import datetime

import numpy as np
import pandas as pd
import gradio as gr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader


APP_TITLE = "Cardiac Risk & Regression Lab (synthetic demo)"
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def bytes_to_data_uri_png(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def now_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ----------------------------
# Synthetic data
# ----------------------------
def generate_synthetic(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))

    age = rng.normal(65, 12, rows).clip(18, 95)
    sex = rng.choice(["male", "female"], size=rows, p=[0.52, 0.48])
    systolic_bp = rng.normal(140, 20, rows).clip(85, 220)
    heart_rate = rng.normal(82, 14, rows).clip(40, 160)
    smoker = rng.choice(["no", "yes"], size=rows, p=[0.72, 0.28])
    diabetes = rng.choice(["no", "yes"], size=rows, p=[0.85, 0.15])
    chest_pain = rng.choice(["none", "atypical", "typical"], size=rows, p=[0.62, 0.25, 0.13])
    ecg = rng.choice(["normal", "st", "lvh"], size=rows, p=[0.70, 0.18, 0.12])

    # Latent risk score -> event probability
    # (purely synthetic, designed to be learnable)
    z = (
        0.035 * (age - 60)
        + 0.018 * (systolic_bp - 130)
        + 0.022 * (heart_rate - 75)
        + 0.65 * (smoker == "yes")
        + 0.55 * (diabetes == "yes")
        + 0.40 * (chest_pain == "typical")
        + 0.18 * (chest_pain == "atypical")
        + 0.22 * (ecg == "st")
        + 0.30 * (ecg == "lvh")
        + 0.10 * (sex == "male")
        - 1.6
    )

    p_event = 1 / (1 + np.exp(-z))
    event_30d = rng.binomial(1, p_event)

    # Synthetic NT-proBNP (skewed, higher with risk + age)
    ntprobnp = (
        80
        + 7.5 * (age)
        + 2.8 * (systolic_bp - 120)
        + 3.0 * (heart_rate - 70)
        + 190 * (event_30d == 1)
        + 160 * (diabetes == "yes")
        + 140 * (smoker == "yes")
        + rng.lognormal(mean=4.7, sigma=0.35, size=rows)  # heavy tail
    )
    ntprobnp = np.clip(ntprobnp, 20, 6000)

    df = pd.DataFrame(
        {
            "age": np.round(age, 1),
            "sex": sex,
            "systolic_bp": np.round(systolic_bp, 1),
            "heart_rate": np.round(heart_rate, 1),
            "smoker": smoker,
            "diabetes": diabetes,
            "chest_pain": chest_pain,
            "ecg": ecg,
            "event_30d": event_30d.astype(int),
            "ntprobnp": np.round(ntprobnp, 1),
        }
    )
    return df


# ----------------------------
# Training + outputs
# ----------------------------
def train_models(rows, seed, test_fraction, threshold):
    """
    Returns:
      preview_df (pd.DataFrame)
      training_summary (str)
      roc_png (bytes) or None
      cm_png (bytes) or None
      reg_png (bytes) or None
      state (dict)
    """
    try:
        rows = int(rows)
        seed = int(seed)
        test_fraction = float(test_fraction)
        threshold = float(threshold)

        if rows < 200:
            raise ValueError("Rows too low. Set at least 200.")
        if not (0.1 <= test_fraction <= 0.5):
            raise ValueError("Test fraction must be between 0.1 and 0.5.")
        if not (0.1 <= threshold <= 0.9):
            raise ValueError("Threshold must be between 0.1 and 0.9.")

        df = generate_synthetic(rows, seed)
        preview_df = df.head(25)

        # Features/targets
        feature_cols = ["age", "sex", "systolic_bp", "heart_rate", "smoker", "diabetes", "chest_pain", "ecg"]
        X = df[feature_cols]
        y_class = df["event_30d"].astype(int)
        y_reg = df["ntprobnp"].astype(float)

        X_train, X_test, y_train, y_test, yreg_train, yreg_test = train_test_split(
            X, y_class, y_reg, test_size=test_fraction, random_state=seed, stratify=y_class
        )

        cat_cols = ["sex", "smoker", "diabetes", "chest_pain", "ecg"]
        num_cols = ["age", "systolic_bp", "heart_rate"]

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("num", "passthrough", num_cols),
            ]
        )

        # Logistic regression
        clf = Pipeline(
            steps=[
                ("pre", pre),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        )
        clf.fit(X_train, y_train)

        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # ROC plot
        fig1 = plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        plt.title(f"ROC Curve (AUC={roc_auc:.3f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        roc_png = fig_to_png_bytes(fig1)

        # Confusion matrix plot
        fig2 = plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        cm_png = fig_to_png_bytes(fig2)

        # Ridge regression for NT-proBNP
        reg = Pipeline(
            steps=[
                ("pre", pre),
                ("model", Ridge(alpha=5.0)),
            ]
        )
        reg.fit(X_train, yreg_train)
        yreg_pred = reg.predict(X_test)

        mae = mean_absolute_error(yreg_test, yreg_pred)
        rmse = math.sqrt(mean_squared_error(yreg_test, yreg_pred))
        r2 = r2_score(yreg_test, yreg_pred)

        # Regression scatter plot
        fig3 = plt.figure()
        plt.scatter(yreg_test, yreg_pred, s=10)
        reminder = max(1.0, float(np.max(yreg_test)))
        plt.plot([0, reminder], [0, reminder])
        plt.title("Ridge Regression: Predicted vs Actual NT-proBNP")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        reg_png = fig_to_png_bytes(fig3)

        summary = []
        summary.append(f"Rows: {rows:,} | Seed: {seed} | Test fraction: {test_fraction:.2f}")
        summary.append("")
        summary.append("Logistic Regression (event_30d):")
        summary.append(f"  Threshold: {threshold:.2f}")
        summary.append(f"  Accuracy:  {acc:.3f}")
        summary.append(f"  ROC AUC:   {roc_auc:.3f}")
        summary.append(f"  Confusion matrix: {cm.tolist()}")
        summary.append("")
        summary.append("Ridge Regression (ntprobnp):")
        summary.append(f"  MAE:   {mae:.1f}")
        summary.append(f"  RMSE:  {rmse:.1f}")
        summary.append(f"  R²:    {r2:.3f}")

        training_summary = "\n".join(summary)

        # Pack state for export
        state = {
            "meta": {
                "title": APP_TITLE,
                "created": datetime.now().isoformat(timespec="seconds"),
                "rows": rows,
                "seed": seed,
                "test_fraction": test_fraction,
                "threshold": threshold,
            },
            "preview": preview_df.to_dict(orient="records"),
            "summary_text": training_summary,
            "plots": {
                "roc_png": roc_png,
                "cm_png": cm_png,
                "reg_png": reg_png,
            },
        }

        return preview_df, training_summary, roc_png, cm_png, reg_png, state

    except Exception as e:
        # IMPORTANT: Return something instead of crashing the UI.
        err = f"ERROR during training:\n{type(e).__name__}: {e}"
        empty = pd.DataFrame()
        return empty, err, None, None, None, {}


# ----------------------------
# Export HTML/PDF
# ----------------------------
def export_html(state: dict):
    try:
        if not state or "meta" not in state:
            return None, "Nothing to export yet. Train models first."

        meta = state["meta"]
        plots = state.get("plots", {})

        roc_uri = bytes_to_data_uri_png(plots["roc_png"]) if plots.get("roc_png") else ""
        cm_uri = bytes_to_data_uri_png(plots["cm_png"]) if plots.get("cm_png") else ""
        reg_uri = bytes_to_data_uri_png(plots["reg_png"]) if plots.get("reg_png") else ""

        preview_rows = state.get("preview", [])
        preview_df = pd.DataFrame(preview_rows)

        # Simple HTML report (self-contained)
        html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{meta.get("title","Report")}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 0; }}
    .muted {{ color: #666; margin-top: 4px; }}
    .card {{ border: 1px solid #ddd; padding: 16px; border-radius: 10px; margin: 16px 0; }}
    pre {{ background: #f7f7f7; padding: 12px; border-radius: 10px; overflow-x: auto; }}
    img {{ max-width: 100%; border: 1px solid #eee; border-radius: 10px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
    th {{ background: #fafafa; text-align: left; }}
  </style>
</head>
<body>
  <h1>{meta.get("title", "Report")}</h1>
  <div class="muted">
    Generated: {meta.get("created","")} &nbsp; | &nbsp;
    Rows: {meta.get("rows","")} &nbsp; | &nbsp;
    Seed: {meta.get("seed","")} &nbsp; | &nbsp;
    Test fraction: {meta.get("test_fraction","")} &nbsp; | &nbsp;
    Threshold: {meta.get("threshold","")}
  </div>

  <div class="card">
    <h2>Training summary</h2>
    <pre>{state.get("summary_text","")}</pre>
  </div>

  <div class="card">
    <h2>Plots</h2>
    <h3>ROC Curve</h3>
    {f'<img src="{roc_uri}"/>' if roc_uri else "<p>(No ROC plot)</p>"}
    <h3>Confusion Matrix</h3>
    {f'<img src="{cm_uri}"/>' if cm_uri else "<p>(No confusion matrix plot)</p>"}
    <h3>Regression: Predicted vs Actual NT-proBNP</h3>
    {f'<img src="{reg_uri}"/>' if reg_uri else "<p>(No regression plot)</p>"}
  </div>

  <div class="card">
    <h2>Preview (first 25 rows)</h2>
    {preview_df.to_html(index=False, escape=True) if not preview_df.empty else "<p>(No preview)</p>"}
  </div>
</body>
</html>
""".strip()

        filename = f"cardiac_report_{now_stamp()}.html"
        path = os.path.join(REPORT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        return path, f"HTML report created: {filename}"

    except Exception as e:
        return None, f"Export HTML failed: {type(e).__name__}: {e}"


def export_pdf(state: dict):
    try:
        if not state or "meta" not in state:
            return None, "Nothing to export yet. Train models first."

        meta = state["meta"]
        plots = state.get("plots", {})

        filename = f"cardiac_report_{now_stamp()}.pdf"
        path = os.path.join(REPORT_DIR, filename)

        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(2 * cm, h - 2 * cm, meta.get("title", "Report"))

        c.setFont("Helvetica", 10)
        c.drawString(2 * cm, h - 2.7 * cm, f"Generated: {meta.get('created','')}")
        c.drawString(2 * cm, h - 3.2 * cm, f"Rows: {meta.get('rows','')}   Seed: {meta.get('seed','')}")
        c.drawString(2 * cm, h - 3.7 * cm, f"Test fraction: {meta.get('test_fraction','')}   Threshold: {meta.get('threshold','')}")

        # Summary
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, h - 5.0 * cm, "Training summary")
        c.setFont("Helvetica", 9)

        summary_text = state.get("summary_text", "")
        y = h - 5.6 * cm
        for line in summary_text.splitlines():
            if y < 2.2 * cm:
                c.showPage()
                y = h - 2.0 * cm
                c.setFont("Helvetica", 9)
            c.drawString(2 * cm, y, line[:120])
            y -= 0.45 * cm

        # New page for plots
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2 * cm, h - 2 * cm, "Plots")

        def draw_png(png_bytes, title, top_y):
            c.setFont("Helvetica-Bold", 10)
            c.drawString(2 * cm, top_y, title)
            if not png_bytes:
                c.setFont("Helvetica", 9)
                c.drawString(2 * cm, top_y - 0.6 * cm, "(missing)")
                return top_y - 1.6 * cm

            img = ImageReader(io.BytesIO(png_bytes))
            # Fit into page width with margin
            max_w = w - 4 * cm
            # Try a fixed height area
            img_h = 7.5 * cm
            img_w = max_w
            c.drawImage(img, 2 * cm, top_y - (img_h + 0.8 * cm), width=img_w, height=img_h, preserveAspectRatio=True, anchor='sw')
            return top_y - (img_h + 1.6 * cm)

        y = h - 3.0 * cm
        y = draw_png(plots.get("roc_png"), "ROC Curve", y)
        y = draw_png(plots.get("cm_png"), "Confusion Matrix", y)

        # If not enough space, new page
        if y < 9 * cm:
            c.showPage()
            y = h - 2.5 * cm

        y = draw_png(plots.get("reg_png"), "Regression: Predicted vs Actual NT-proBNP", y)

        c.save()
        return path, f"PDF report created: {filename}"

    except Exception as e:
        return None, f"Export PDF failed: {type(e).__name__}: {e}"


# ----------------------------
# UI
# ----------------------------
with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}\nA clinical-style Gradio app that teaches regression while looking impressive.\n\n"
                f"**Two models:**\n"
                f"- Logistic Regression → probability of a 30-day cardiac event\n"
                f"- Linear Regression (Ridge) → predicted NT-proBNP\n\n"
                f"All data is **synthetic** for learning and demonstration.")

    state = gr.State({})

    with gr.Tab("1) Train & Evaluate"):
        with gr.Row():
            rows = gr.Slider(500, 20000, value=8000, step=100, label="Rows of synthetic data")
            seed = gr.Number(value=42, precision=0, label="Random seed")

        with gr.Row():
            test_fraction = gr.Slider(0.1, 0.5, value=0.25, step=0.01, label="Test set fraction")
            threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.01, label="Event decision threshold")

        train_btn = gr.Button("Train models", variant="primary")

        preview = gr.Dataframe(label="Preview (first 25 rows)", interactive=False, wrap=True)
        summary = gr.Textbox(label="Training summary", lines=10)

        with gr.Row():
            roc_img = gr.Image(label="ROC curve", type="numpy")
            cm_img = gr.Image(label="Confusion matrix", type="numpy")
            reg_img = gr.Image(label="Regression: predicted vs actual", type="numpy")

        gr.Markdown("## Export report")
        with gr.Row():
            export_html_btn = gr.Button("Export HTML report")
            export_pdf_btn = gr.Button("Export PDF report")

        export_file = gr.File(label="Download")
        export_msg = gr.Textbox(label="Export status", lines=2)

        def _train_and_show(rows, seed, test_fraction, threshold):
            prev_df, summ, roc_png, cm_png, reg_png, st = train_models(rows, seed, test_fraction, threshold)

            # Convert PNG bytes -> numpy arrays for gr.Image
            def png_bytes_to_numpy(png_bytes):
                if not png_bytes:
                    return None
                import PIL.Image
                img = PIL.Image.open(io.BytesIO(png_bytes)).convert("RGB")
                return np.array(img)

            return (
                prev_df,
                summ,
                png_bytes_to_numpy(roc_png),
                png_bytes_to_numpy(cm_png),
                png_bytes_to_numpy(reg_png),
                st
            )

        train_btn.click(
            _train_and_show,
            inputs=[rows, seed, test_fraction, threshold],
            outputs=[preview, summary, roc_img, cm_img, reg_img, state],
        )

        export_html_btn.click(
            export_html,
            inputs=[state],
            outputs=[export_file, export_msg],
        )

        export_pdf_btn.click(
            export_pdf,
            inputs=[state],
            outputs=[export_file, export_msg],
        )

    with gr.Tab("2) Predict (patient-style)"):
        gr.Markdown("Optional. If you want this tab fully wired into the trained models, say so and I’ll hook it up cleanly.")

# IMPORTANT: Your Gradio version choked on queue(concurrency_count=...)
# So we keep it simple and compatible.
if __name__ == "__main__":
    demo.queue()
    demo.launch()
