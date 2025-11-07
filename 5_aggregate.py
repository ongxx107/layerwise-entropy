import os
import json
import glob

import numpy as np
import torch.nn.functional as F


model_name = "MODEL_NAME"
predictions_dir = "predictions_per_json"
internal_dir = "internal"
model_path = os.path.join(predictions_dir, internal_dir, model_name)
os.makedirs(model_path, exist_ok=True)
mean_std_path = os.path.join(predictions_dir, internal_dir, model_name, "mean_std")
os.makedirs(mean_std_path, exist_ok=True)

# Loop through each mapped validation file
for file_path in glob.glob(os.path.join(model_path, "*.json")):
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    with open(file_path, "r") as f:
        data = json.load(f)
        
    num_layers = int(data["result"][0]["total_layers"])
    
    entropy_mean = [
        float(np.mean([item["hidden_states_entropy"][i] for item in data["result"]]))
        for i in range(num_layers)
    ]
    
    entropy_std = [
        float(np.std([item["hidden_states_entropy"][i] for item in data["result"]]))
        for i in range(num_layers)
    ]
    
    varentropy_mean = [
        float(np.mean([item["hidden_states_varentropy"][i] for item in data["result"]]))
        for i in range(num_layers)
    ]
    
    varentropy_std = [
        float(np.std([item["hidden_states_varentropy"][i] for item in data["result"]]))
        for i in range(num_layers)
    ]
    
    result_json_path = os.path.join(mean_std_path, f"{filename_without_ext}.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump({"total_layers": num_layers,"entropy_mean": entropy_mean, "entropy_std": entropy_std, "varentropy_mean": varentropy_mean, "varentropy_std": varentropy_std}, f, indent=2, ensure_ascii=False)
        
        
        
        
