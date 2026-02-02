"""
Martian Care for Aliens — Gradio ML demo
- Synthetic data generator
- Train + evaluate classifier
- Confusion matrix plot
- Interactive predictions + explanation

Run:
  python app.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Synthetic "alien triage" data
# -----------------------------
SPECIES = ["Zorgon", "Velorian", "Kith", "Aurelian", "Mekanoid"]
PLANETS = ["Phobos-9", "Europa Station", "New Carthage", "Io Forge", "Luna Outpost"]
YESNO = ["no", "yes"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_synthetic_df(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        species = rng.choice(SPECIES)
        planet = rng.choice(PLANETS)

        # "Vitals"
        pulse = rng.gauss(92, 18)          # beats per minute
        temp_c = rng.gauss(38.0, 1.2)      # Celsius
        o2 = rng.gauss(93.5, 4.5)          # oxygen saturation %
        pain = max(0, min(10, rng.gauss(4.5, 2.5)))  # 0-10

        # Symptoms (categorical)
        cyanosis = rng.choice(YESNO)       # blue-ish skin
        neuro = rng.choice(YESNO)          # confusion / tremor
        bleed = rng.choice(YESNO)          # bleeding

        # Species quirks
        # (just enough structure so the model can learn patterns)
        sp_temp_bias = {
            "Zorgon": 0.4,
            "Velorian": -0.3,
            "Kith": 0.0,
            "Aurelian": 0.2,
            "Mekanoid": -0.8,  # "cooler"
        }[species]
        temp_c += sp_temp_bias

        sp_o2_bias = {
            "Zorgon": -1.0,
            "Velorian": 0.5,
            "Kith": 0.0,
            "Aurelian": -0.5,
            "Mekanoid": 1.5,
        }[species]
        o2 += sp_o2_bias

        # Risk score -> triage label
        # 0 = routine, 1 = urgent, 2 = critical
        score = 0.0
        score += 0.06 * (pulse - 90)
        score += 0.9 * (temp_c - 38.0)
        score += 0.12 * (92.0 - o2)
        score += 0.22 * (pain - 4.0)

        if cyanosis == "yes":
            score += 1.2
        if neuro == "yes":
            score += 0.9
        if bleed == "yes":
            score += 1.4

        # Planet adds subtle operational effect (noise, environment)
        planet_noise = {
            "Phobos-9": 0.2,
            "Europa Station": 0.0,
            "New Carthage": 0.1,
            "Io Forge": 0.3,
            "Luna Outpost": -0.1,
        }[planet]
        score += planet_noise

        # Convert score to probabilities for each class (soft-ish)
        p_critical = _sigmoid(score - 1.6)
        p_urgent = _sigmoid(score - 0.3) - p_critical
        p_routine = max(0.0, 1.0 - (p_critical + p_urgent))

        r = rng.random()
        if r < p_critical:
            triage = "critical"
        elif r < p_critical + p_urgent:
            triage = "urgent"
        else:
            triage = "routine"

        rows.append(
            dict(
                species=species,
                planet=planet,
                pulse=round(pulse, 1),
                temp_c=round(temp_c, 1),
                o2=round(o2, 1),
                pain=round(pain, 1),
                cyanosis=cyanosis,
                neuro=neuro,
                bleed=bleed,
                triage=triage,
            )
        )

    return pd.DataFrame(rows)


# -----------------------------
# Model training + evaluation
# -----------------------------
CAT_COLS = ["species", "planet", "cyanosis", "neuro", "bleed"]
NUM_COLS = ["pulse", "temp_c", "o2", "pain"]


@dataclass
class TrainedState:
    model: Pipeline
    feature_cols: list
    label_col: str = "triage"


STATE: Dict[str, TrainedState] = {}


def train_model(n: int, seed: int, test_size: float) -> Tuple[pd.DataFrame, str, plt.Figure]:
    df = make_synthetic_df(n=n, seed=seed)

    X = df[CAT_COLS + NUM_COLS]
    y = df["triage"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    clf = LogisticRegression(max_iter=400, n_jobs=None)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)

    labels = ["routine", "urgent", "critical"]
    cm = confusion_matrix(y_test, pred, labels=labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # store trained model in memory
    STATE["trained"] = TrainedState(model=pipe, feature_cols=CAT_COLS + NUM_COLS)

    summary = (
        f"Trained on {len(X_train)} rows, tested on {len(X_test)} rows.\n"
        f"Accuracy: {acc:.3f}\n\n"
        f"Labels: {', '.join(labels)}\n"
        f"Tip: click a few examples below, then try your own vitals."
    )

    return df.head(25), summary, fig


def predict_one(
    species: str,
    planet: str,
    pulse: float,
    temp_c: float,
    o2: float,
    pain: float,
    cyanosis: str,
    neuro: str,
    bleed: str,
) -> Tuple[str, str]:
    if "trained" not in STATE:
        return "No model yet.", "Hit **Train model** first."

    pipe = STATE["trained"].model

    row = pd.DataFrame(
        [
            dict(
                species=species,
                planet=planet,
                pulse=pulse,
                temp_c=temp_c,
                o2=o2,
                pain=pain,
                cyanosis=cyanosis,
                neuro=neuro,
                bleed=bleed,
            )
        ]
    )

    pred = pipe.predict(row)[0]

    # “Clinician-ish” explanation (simple + honest)
    flags = []
    if pulse >= 120:
        flags.append("very high pulse")
    if temp_c >= 39.5:
        flags.append("high temperature")
    if o2 <= 90:
        flags.append("low oxygen saturation")
    if pain >= 8:
        flags.append("severe pain")
    if cyanosis == "yes":
        flags.append("cyanosis present")
    if neuro == "yes":
        flags.append("neurological symptoms")
    if bleed == "yes":
        flags.append("bleeding present")

    if not flags:
        flags_txt = "No obvious red flags from the thresholds."
    else:
        flags_txt = "Red flags: " + ", ".join(flags) + "."

    expl = (
        f"Prediction: **{pred.upper()}**\n\n"
        f"{flags_txt}\n\n"
        f"Species/planet context can shift baselines (this is synthetic, but structured)."
    )

    return pred, expl


# -----------------------------
# Gradio UI (Blocks + Tabs)
# -----------------------------
with gr.Blocks(title="Martian Care for Aliens — Triage Lab") as demo:
    gr.Markdown(
        """
# Martian Care for Aliens — Triage Lab (synthetic demo)
This app generates **synthetic** alien “patient” data, trains a model, and lets you test predictions.

- Tab 1: Generate + train + evaluate
- Tab 2: Interactive triage prediction
"""
    )

    with gr.Tabs():
        with gr.TabItem("1) Train & Evaluate"):
            with gr.Row():
                n = gr.Slider(200, 10000, value=2000, step=100, label="Rows of synthetic data")
                seed = gr.Number(value=42, precision=0, label="Random seed")
                test_size = gr.Slider(0.1, 0.5, value=0.25, step=0.05, label="Test set fraction")

            train_btn = gr.Button("Train model", variant="primary")

            with gr.Row():
                df_preview = gr.Dataframe(label="Preview (first 25 rows)", interactive=False)
            summary = gr.Textbox(label="Training summary", lines=6)
            cm_plot = gr.Plot(label="Confusion matrix")

            train_btn.click(
                fn=train_model,
                inputs=[n, seed, test_size],
                outputs=[df_preview, summary, cm_plot],
            )

        with gr.TabItem("2) Predict"):
            gr.Markdown("Enter alien vitals & symptoms, then predict triage level.")

            with gr.Row():
                species_in = gr.Dropdown(SPECIES, value=SPECIES[0], label="Species")
                planet_in = gr.Dropdown(PLANETS, value=PLANETS[0], label="Planet / habitat")

            with gr.Row():
                pulse_in = gr.Slider(40, 200, value=95, step=1, label="Pulse (beats per minute)")
                temp_in = gr.Slider(34.0, 42.5, value=38.2, step=0.1, label="Temperature (Celsius)")
            with gr.Row():
                o2_in = gr.Slider(70.0, 100.0, value=93.0, step=0.1, label="Oxygen saturation (%)")
                pain_in = gr.Slider(0, 10, value=4, step=1, label="Pain (0–10)")

            with gr.Row():
                cyan_in = gr.Radio(YESNO, value="no", label="Cyanosis")
                neuro_in = gr.Radio(YESNO, value="no", label="Neurological symptoms")
                bleed_in = gr.Radio(YESNO, value="no", label="Bleeding")

            pred_btn = gr.Button("Predict triage", variant="primary")
            pred_label = gr.Label(label="Predicted class")
            explanation = gr.Markdown()

            pred_btn.click(
                fn=predict_one,
                inputs=[species_in, planet_in, pulse_in, temp_in, o2_in, pain_in, cyan_in, neuro_in, bleed_in],
                outputs=[pred_label, explanation],
            )

    gr.Markdown(
        """
### Sharing note
If you run this locally you can optionally use `share=True` to create a temporary public link.
Only do that with synthetic data / demos.  
"""
    )

if __name__ == "__main__":
    # share=True creates a public link (handy for demos) per Gradio docs.
    # Keep it False by default.
    demo.launch(share=False)
