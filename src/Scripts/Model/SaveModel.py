# src/LexaLCM/save_model.py

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.LexaLCM.Config.LCM_Config import LexaLCMConfig
from LexaLCM.LCM_Model import LexaLCM

# Instantiate configuration and model
config = LexaLCMConfig()
model = LexaLCM(config)

# Define the directory to save the model
save_directory = "tmp_output"

# Save the configuration and model
config.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model and configuration saved to '{save_directory}'")
