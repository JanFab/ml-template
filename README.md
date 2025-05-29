# MNIST Classification Service

A microservice-based MNIST classification system using PyTorch, FastAPI, and Kubernetes.

## Project Structure

```
ml/
├── .github/                      # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml               # Continuous Integration
│       └── cd.yml               # Continuous Deployment
│
├── src/                         # Source code
│   ├── api/                     # API service
│   │   └── main.py             # FastAPI application
│   │
│   ├── ml_service/             # ML service
│   │   └── models.py           # PyTorch model
│   │
│   └── shared/                 # Shared utilities
│
├── containers/                  # Container definitions
│   └── api/                    # API service container
│       └── Dockerfile
│
├── k8s/                        # Kubernetes configurations
│   ├── base/                   # Base configurations
│   └── overlays/               # Environment-specific configs
│       ├── development/
│   └── production/
│
├── configs/                    # Configuration files
│   ├── api/
│   └── ml_service/
│
├── tests/                      # Test files
├── data/                       # Data directory
└── notebooks/                  # Jupyter notebooks
```

## Setup

### Prerequisites

- Python 3.9+
- Docker
- Kubernetes cluster
- kubectl
- kustomize

### Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install .
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

### Container Build

Build the API container:
```bash
docker build -t mnist-api:latest -f containers/api/Dockerfile .
```

### Kubernetes Deployment

1. Development:
   ```bash
   kustomize build k8s/overlays/development | kubectl apply -f -
   ```

2. Production:
   ```bash
   kustomize build k8s/overlays/production | kubectl apply -f -
   ```

## API Endpoints

- `GET /`: Health check endpoint
- `POST /predict`: Upload an image for MNIST digit classification

## CI/CD Options

The project supports multiple CI/CD platforms:

### GitHub Actions
- Located in `.github/workflows/`
- `ci.yml`: Runs tests and linting
- `cd.yml`: Builds and deploys to Kubernetes

### GitLab CI/CD
- Located in `.gitlab-ci.yml`
- Stages: test, lint, build, deploy
- Uses GitLab Container Registry
- Supports manual deployment to production
- Environment-specific deployments

### Azure DevOps
- Located in `azure-pipelines.yml`
- Stages: Test, Build, Deploy
- Uses Azure Container Registry
- Environment-specific deployments
- Manual approval for production

Required Variables:
- GitLab:
  - `KUBE_CONTEXT_DEV`: Development cluster context
  - `KUBE_CONTEXT_PROD`: Production cluster context
  - `CI_REGISTRY`: Container registry URL
  - `CI_REGISTRY_USER`: Registry username
  - `CI_REGISTRY_PASSWORD`: Registry password

- Azure DevOps:
  - `ACR_NAME`: Azure Container Registry name
  - `KUBE_SERVICE_CONNECTION_DEV`: Development cluster connection
  - `KUBE_SERVICE_CONNECTION_PROD`: Production cluster connection

## Future Improvements

- Add Kafka integration for async processing
- Implement model versioning
- Add monitoring and logging
- Implement A/B testing
- Add more comprehensive testing
- Implement model retraining pipeline