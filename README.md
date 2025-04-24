# ML Project

This project implements a PyTorch Lightning model for MNIST classification.

## Project Structure

```
.
├── model.py           # Model architecture
├── train.py          # Training script
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── requirements.txt  # Python dependencies
└── README.md        # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml.git
cd ml
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development with Docker

1. Build and start the development environment:
```bash
docker-compose up --build
```

2. Access VS Code in your browser at `http://localhost:8080`

## Training the Model

1. Start the training container:
```bash
docker-compose up ml
```

2. The model will train using the specified configuration in `train.py`

## GPU Support

The project is configured to use GPU acceleration when available. Make sure you have NVIDIA drivers installed and Docker configured for GPU support.

## License

[Add your license here] 