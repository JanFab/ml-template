# Build the container
podman build -t ml-training -f containers/ml-service/Dockerfile .

# Run the container with volume mounts for data and MLflow
podman run -it --rm `
    -v ${PWD}/data:/app/data `
    -v ${PWD}/mlruns:/app/mlruns `
    -v ${PWD}/configs:/app/configs `
    ml-training 