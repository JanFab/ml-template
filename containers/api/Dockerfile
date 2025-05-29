FROM python:3.10-slim

WORKDIR /app

# Copy only the necessary files
COPY src/api /app/api
COPY src/ml_service /app/ml_service
COPY src/shared /app/shared
COPY pyproject.toml /app/

# Install dependencies
RUN pip install --no-cache-dir .

# Copy model checkpoint
COPY logs/lightning_logs/version_0/checkpoints/mnist-epoch=03-val_loss=0.02.ckpt /app/model.ckpt

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 