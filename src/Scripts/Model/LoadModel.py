# src/LexaLCM/load_model.py

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.LexaLCM.Config.LCM_Config import LexaLCMConfig
from LexaLCM.LCM_Model import LexaLCM

# Define the directory where the model is saved
load_directory = "tmp_output"

# Load the configuration and model
config = LexaLCMConfig.from_pretrained(load_directory)
model = LexaLCM.from_pretrained(load_directory)

print("Model and configuration loaded successfully.")
