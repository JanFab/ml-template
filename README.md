# MNIST Classification Project

A clean, production-ready MNIST classification project using PyTorch Lightning and MLflow.

## Project Structure

```
mnist-classification/
├── configs/                 # Configuration files
│   ├── development.yaml    # Development environment config
│   ├── production.yaml     # Production environment config
│   └── testing.yaml        # Testing environment config
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   │   └── mnist_data_module.py
│   ├── models/            # Model definitions
│   │   ├── base.py
│   │   └── mnist_model.py
│   └── utils/             # Utility functions
│       └── visualization.py
├── tests/                 # Unit tests
│   └── test_models.py
├── scripts/               # Training script
│   └── train.py
├── pyproject.toml         # Project configuration and dependencies
├── requirements-dev.txt   # Development dependencies with exact versions
└── README.md             # This file
```

## Features

- **Clean Architecture**: Modular and maintainable code structure
- **Environment Management**: Separate configurations for development, testing, and production
- **Experiment Tracking**: MLflow integration for tracking experiments
- **Type Hints**: Full type annotation support
- **Testing**: Comprehensive test suite
- **Code Quality**: Black, isort, and mypy for code quality

## Quick Start

1. Create and activate a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

2. Install the package in development mode:
```powershell
pip install -e ".[dev]"
```

3. Train the model:
```powershell
python scripts/train.py --env development
```

4. View experiment results:
```powershell
mlflow ui
```

## Development

### Code Quality

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing

Run them directly:
```powershell
# Format code
black src tests
isort src tests

# Run code quality checks
mypy src
flake8 src tests

# Run tests
pytest
```

### Environment-Specific Training

The project supports different environments:

- **Development** (default):
```powershell
python scripts/train.py --env development
```

- **Testing**:
```powershell
python scripts/train.py --env testing
```

- **Production**:
```powershell
python scripts/train.py --env production
```

### Configuration

Environment-specific configurations are in the `configs/` directory:
- `development.yaml`: Development settings
- `production.yaml`: Production settings
- `testing.yaml`: Testing settings

## Project Structure Details

- **src/**: Core package code
  - `data/`: Data loading and processing modules
  - `models/`: Model definitions and training logic
  - `utils/`: Utility functions and helpers

- **configs/**: Configuration files for different environments

- **tests/**: Unit tests and test utilities

- **scripts/**: Training script

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests and code quality checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.