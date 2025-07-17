import pytest
import torch
import numpy as np

from pylibxai.Explainers import ShapExplainer

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

@pytest.fixture
def model():
    return MockModel()

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_init(explainer, model, device):
    assert explainer.device == device
    assert explainer.attributions is None
    assert explainer.delta is None

def test_explain_instance(explainer):
    # Create dummy input
    audio = torch.randn(1, 1, 64, 64)  # [batch, channels, height, width]
    target = 0
    
    # Test explanation
    attributions, delta = explainer.explain_instance(audio, target)
    
    # Check shapes and types
    assert isinstance(attributions, torch.Tensor)
    assert isinstance(delta, torch.Tensor)
    assert attributions.shape == audio.shape
    assert delta.shape[0] == audio.shape[0]

def test_explain_instance_visualize(explainer):
    # Create dummy input
    audio = torch.randn(1, 1, 64, 64)
    target = 0
    
    # Test visualization
    fig, _ = explainer.explain_instance_visualize(audio, target, type="blended_heat_map")
    
    # Check that figure was created
    assert fig is not None
    assert explainer.attributions is not None
    assert explainer.delta is not None

def test_save_attributions(explainer, tmp_path):
    # Create dummy attributions
    explainer.attributions = torch.randn(1, 1, 64, 64)
    save_path = tmp_path / "test_attributions.json"
    
    # Test saving
    explainer.save_attributions(str(save_path))
    assert save_path.exists() 

@pytest.fixture
def explainer(model, device):
    return ShapExplainer(model, device)

def test_save_spectrogram(explainer, tmp_path):
    # Create dummy input
    audio = torch.randn(1, 1, 64, 64)
    save_path = tmp_path / "test_spectrogram.png"
    
    # Test saving
    explainer.save_spectrogram(audio, str(save_path))
    assert save_path.exists()


