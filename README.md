# Breast Cancer Classification

This repository contains a reproducible pipeline for training an image-based
breast cancer classifier (local-only, no external API keys required).

Contents
- `build_dataset.py` — produce stratified train/val/test splits (uses `cancernet/data_pipeline.py`).
- `train_model.py` — training script (Keras/TensorFlow) and evaluation.
- `demo/app.py` — Gradio demo scaffold for local inference (uses a local Keras model).
- `Dockerfile` / `Dockerfile.gpu` — images for training; `Dockerfile.demo` runs the Gradio demo.

Goals and recruiter-friendly assets
- Clear, reproducible instructions for running the pipeline locally.
- An offline demo so reviewers can try inference without credentials or cloud costs.
- A concise `README` + demo screenshots / short GIF (add to repo) to attract recruiters.

Safety & privacy
- This repo avoids any external dataset downloads or cloud API keys in code.
- Keep sensitive files out of the repo (use `.env` for any secrets; see `.env.example`).

Quickstart (local, no Docker)
1. Create and activate a Python virtual environment.
2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
3. Prepare your dataset under `datasets/original/` following the original layout (images in subfolders or named as expected).
4. Build dataset splits:
```bash
python build_dataset.py
```
5. Train the model (example):
```bash
python train_model.py
```

Run the Gradio demo locally (offline)
1. Place a trained Keras model at `models/model.h5` or set `MODEL_PATH` env var.
2. Start the demo:
```bash
python demo/app.py
```
3. Open http://localhost:7860 to interact with the demo.

Run the demo via Docker
```bash
docker build -t bcc-demo -f Dockerfile.demo .
docker run --rm -p 7860:7860 -v $(pwd):/app bcc-demo
```

Developer notes — concrete tasks completed here

- Implemented a robust local data pipeline: `cancernet/data_pipeline.py` (discover, validate, stratified split, copy).
- Replaced the original ad-hoc dataset builder with `build_dataset.py` that calls the pipeline.
- Added a Gradio offline demo scaffold at `demo/app.py` and `Dockerfile.demo`.
- Added `.env.example` and updated `requirements.txt` to include `gradio`.

Next suggested steps (pick one):
- Add a `models/` example (small, synthetic) so recruiters can try demo immediately.
- Create a short demo GIF and add it to the README root to showcase results.
- Add CI smoke test that runs `python -m demo.app --help` or a tiny unit test.

If you want, I can:
- add a small synthetic dataset and run a quick training pass locally (no GPU) to produce an example model; or
- scaffold a GitHub Actions workflow that runs smoke tests and builds the demo Docker image.

