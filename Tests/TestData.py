# Tests/TestData.py

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
from src.LexaLCM.Data.DataHandler import LCMDataset, LCMCollator

def test_dataset_shapes():
    dataset = LCMDataset()
    sample = dataset[0]
    assert sample["embeddings"].shape == (3, 1024)
    assert sample["labels"].shape == (1024,)

def test_collator():
    collator = LCMCollator()
    batch = collator([{
        "embeddings": torch.randn(3, 1024),
        "labels": torch.randn(1024)
    }])
    assert batch["embeddings"].shape == (1, 3, 1024)
    assert batch["labels"].shape == (1, 1024)