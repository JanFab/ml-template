# MNIST FastAPI Service

This project provides a FastAPI-based web service for MNIST digit classification, including a web UI for uploading images and viewing predictions.

## Project Structure

- `api_main.py` — FastAPI server for inference and serving the web UI
- `train.py` — Script to train the MNIST model
- `predict.yaml` — Inference configuration (checkpoint path, model params)
- `static/` — Contains `index.html` (or `index_nice.html`) and any static assets for the web UI
- `mlruns/` — MLflow experiment tracking and model checkpoints

## Setup

1. **Install dependencies** (in your environment):
   ```sh
   pip install -r requirements.txt
   # or manually:
   pip install fastapi uvicorn torch torchvision pytorch-lightning pillow numpy mlflow
   ```

2. **(Optional) Activate your environment:**
   ```sh
   conda activate dp
   # or
   source venv/bin/activate
   ```

## Training

Train the MNIST model and log results/checkpoints:
```sh
python train.py --config train.yaml
```
- The best checkpoint will be saved in `mlruns/` or as configured in your script.

## Inference API

1. **Set the correct checkpoint path in `predict.yaml`:**
   ```yaml
   checkpoint_path: mlruns/<experiment_id>/<run_id>/checkpoints/<best_checkpoint>.ckpt
   model:
     learning_rate: 0.001
   ```
2. **Run the API server:**
   ```sh
   python api_main.py
   # or
   uvicorn api_main:app --host 0.0.0.0 --port 8000
   ```
3. **Open the web UI:**
   - Go to [http://localhost:8000/](http://localhost:8000/) in your browser.
   - The default page is `static/index.html` or `static/index_nice.html` (as configured in `api_main.py`).

4. **API Endpoints:**
   - `POST /predict` — Upload an image file to get a digit prediction (returns JSON).
   - `GET /` — Web UI (HTML page).

## MLflow Tracking

- To view experiment results and artifacts, run:
  ```sh
  mlflow ui
  ```
  Then open [http://localhost:5000](http://localhost:5000) in your browser.

## Notes
- You can switch the main web UI by changing the file served in the `/` route in `api_main.py`.
- All static files (HTML, CSS, JS) should be placed in the `static/` directory.
- Make sure the checkpoint path in `predict.yaml` matches the actual location of your best model. 