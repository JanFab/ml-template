# MNIST Classification Project

This project implements a PyTorch Lightning-based MNIST digit classification model with a FastAPI service for model inference and a web interface for easy interaction.

## Project Structure

```
.
├── experiment.py      # Training script
├── api.py            # FastAPI server
├── index.html        # Web interface
├── MNISTModel.py     # Model architecture
├── MnistDataModule.py # Data loading module
├── utils.py          # Utility functions
├── requirements.txt  # Project dependencies
└── logs/            # Training logs and model checkpoints
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

To train the model, run:
```bash
python experiment.py
```

This will:
- Train the model for 3 epochs
- Save checkpoints in the `logs` directory
- Generate visualizations of predictions and confusion matrix

## Running the API Server

1. Start the API server:
```bash
python api.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **URL**: `/`
- **Method**: GET
- **Response**: Welcome message and usage instructions

### 2. Prediction Endpoint
- **URL**: `/predict`
- **Method**: POST
- **Input**: Image file (any format)
- **Response**: JSON containing:
  - `prediction`: Predicted digit (0-9)
  - `probabilities`: List of probabilities for each digit

## Example Usage

### Using curl
```bash
curl -X POST -F "file=@path/to/your/image.png" http://localhost:8000/predict
```

### Using Python requests
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("path/to/your/image.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Model Details

- Architecture: PyTorch Lightning-based neural network
- Input: 28x28 grayscale images
- Output: Digit classification (0-9)
- Training: Uses early stopping and model checkpointing
- Hardware: Automatically uses available GPU/MPS if available

## Notes

- The API automatically preprocesses images to match MNIST format:
  - Converts to grayscale
  - Resizes to 28x28 pixels
  - Normalizes pixel values
  - Inverts colors (MNIST has white digits on black background)
- Make sure the model checkpoint exists at `logs/mnist-02-0.12.ckpt` before starting the API server

## Web Interface

The project includes a modern web interface for easy interaction with the model. To use it:

1. Start the API server:
```bash
python api.py
```

2. Open `index.html` in your web browser. You can do this by:
   - Double-clicking the file in your file explorer
   - Using a simple HTTP server:
     ```bash
     # Using Python's built-in HTTP server
     python -m http.server 8080
     # Then visit http://localhost:8080 in your browser
     ```

The web interface features:
- Drag and drop image upload
- Live image preview
- Real-time predictions with probability bars
- Responsive design that works on both desktop and mobile
- Error handling and loading states

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- FastAPI 0.68+
- Other dependencies listed in requirements.txt 