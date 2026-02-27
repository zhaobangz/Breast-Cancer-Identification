"""Gradio demo scaffold for local model inference.

This demo loads a local Keras model (path set via MODEL_PATH env var or
models/model.h5) and exposes a minimal web UI for image upload and
prediction. It is intentionally offline and does not call any external
APIs or require payment keys.
"""

import os
import sys
from PIL import Image
import numpy as np
import gradio as gr

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.h5")

def load_model_safe(path):
    try:
        from tensorflow.keras.models import load_model
        if os.path.exists(path):
            return load_model(path)
        else:
            return None
    except Exception as e:
        print("Model load failed:", e)
        return None


model = load_model_safe(MODEL_PATH)


def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((48, 48))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(img: Image.Image):
    if model is None:
        return {"error": "Model not found. Set MODEL_PATH or place model at models/model.h5"}
    x = preprocess_image(img)
    preds = model.predict(x)
    if preds.shape[-1] == 1 or preds.shape[-1] == 2:
        # binary
        if preds.shape[-1] == 1:
            prob = float(preds[0][0])
            return {"benign": 1.0 - prob, "malignant": prob}
        else:
            prob = float(preds[0][1])
            return {"benign": 1.0 - prob, "malignant": prob}
    else:
        # multi-class
        labels = [str(i) for i in range(preds.shape[-1])]
        return dict(zip(labels, map(float, preds[0].tolist())))


def main():
    title = "Breast Cancer Classification — Demo (offline)"
    description = "Upload a microscopy image. This demo runs a local Keras model; no external API or keys used."
    iface = gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=2), title=title, description=description)
    iface.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
