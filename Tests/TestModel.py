# Tests/TestModel.py

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.LexaLCM import LexaLCM, LexaLCMConfig
import torch

def test_forward_pass():
    config = LexaLCMConfig()
    model = LexaLCM(config)
    dummy_input = torch.randn(1, 3, 1024)
    output = model(dummy_input)
    assert output["logits"].shape == (1, 1024)
    assert output["loss"] is None  # No labels provided
    
    dummy_labels = torch.randn(1, 1024)
    output = model(dummy_input, labels=dummy_labels)
    assert output["loss"] is not None