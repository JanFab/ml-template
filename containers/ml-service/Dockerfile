FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY pyproject.toml .
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the training script
CMD ["python", "scripts/train.py"] 