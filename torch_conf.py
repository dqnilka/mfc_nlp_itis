import os

import torch


model_path = "train_labse/finetuned_labse_model.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist.")

if not os.access(model_path, os.R_OK):
    raise PermissionError(f"The model file {model_path} is not readable.")