import os

import torch


model_path = "train_labse/finetuned_labse_model.pth"

# Проверяем, существует ли файл по указанному пути
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file {model_path} does not exist.")

# Убедитесь, что права доступа к файлу корректны
if not os.access(model_path, os.R_OK):
    raise PermissionError(f"The model file {model_path} is not readable.")