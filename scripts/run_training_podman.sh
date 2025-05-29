#!/bin/bash

# Build the container
podman build -t ml-training -f containers/ml-service/Dockerfile .

# Run the container with volume mounts for data and MLflow
podman run -it --rm \
    -v ./data:/app/data \
    -v ./mlruns:/app/mlruns \
    -v ./configs:/app/configs \
    ml-training 