import pytest
import torch
from src.models import MNISTModel

@pytest.fixture
def config():
    return {
        'model': {
            'learning_rate': 0.001,
            'hidden_size': 128
        }
    }

def test_mnist_model_initialization(config):
    model = MNISTModel(config)
    assert isinstance(model, MNISTModel)
    assert model.conv1.in_channels == 1
    assert model.conv1.out_channels == 32
    assert model.fc2.out_features == 10

def test_mnist_model_forward(config):
    model = MNISTModel(config)
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10)
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(batch_size)) 