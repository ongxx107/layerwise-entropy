from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score
import json
import os
import re

def parse_labels(label_str):
    #return [x.strip().replace("(", "").replace(")", "").upper() for x in label_str.split(",")]
    return re.findall(r"\(([A-J])\)", label_str)
    
model_name = "MODEL_NAME"

# Directory containing all prediction result JSONs
predictions_dir = "predictions_per_json"
model_path = os.path.join(predictions_dir, model_name)

# Optional: where to store final metrics for all files
summary_file = os.path.join(model_path, "summary_scores.txt")
with open(summary_file, "w", encoding="utf-8") as summary_out:

    for filename in os.listdir(model_path):
        if not filename.endswith("_predictions.json"):
            continue

        file_path = os.path.join(model_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        y_true_raw = data["y_true"]
        y_pred_raw = data["y_pred"]

        # --- Clean and filter ---
        y_true_parsed = [parse_labels(x) for x in y_true_raw]
        y_pred_parsed = [parse_labels(x) for x in y_pred_raw]

        # Filter out samples with empty ground truth
        filtered_true = []
        filtered_pred = []

        for true, pred in zip(y_true_parsed, y_pred_parsed):
            if true:  # skip examples with no ground truth
                filtered_true.append(true)
                filtered_pred.append(pred)
        
        filename_without_ext = os.path.splitext(filename)[0]
        result_txt_path = os.path.join(model_path, f"filtered_{filename_without_ext}_comparison.txt")
        with open(result_txt_path, "w", encoding="utf-8") as f:
            for idx, (pred, truth) in enumerate(zip(filtered_pred, filtered_true), start=1):
                f.write(f"### Sample {idx}\n")
                f.write(f"Predicted: {pred}\n")
                f.write(f"Ground Truth: {truth}\n")
                f.write("=" * 80 + "\n")
                
        # --- Evaluation ---
        if "casehold" in filename.lower():
            # Special-case: accuracy only for single-label samples (casehold)
            y_true_flat = [labels[0] if labels else None for labels in filtered_true]
            y_pred_flat = [labels[0] if labels else "UNKNOWN" for labels in filtered_pred]
            acc = accuracy_score(y_true_flat, y_pred_flat)
            print(f"\n===== {filename} =====")
            print(f"Accuracy: {acc:.4f}")
            summary_out.write(f"===== {filename} =====\n")
            summary_out.write(f"Accuracy: {acc:.4f}\n")
        else:
            mlb = MultiLabelBinarizer()
            mlb.fit(filtered_true + filtered_pred)
    
            y_true_bin = mlb.transform(filtered_true)
            y_pred_bin = mlb.transform(filtered_pred)
    
            macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro")
            micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro")
            
    
            print(f"\n===== {filename} =====")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Micro F1: {micro_f1:.4f}")
    
            summary_out.write(f"===== {filename} =====\n")
            summary_out.write(f"Macro F1: {macro_f1:.4f}\n")
            summary_out.write(f"Micro F1: {micro_f1:.4f}\n")
        summary_out.write("=" * 80 + "\n")
        
print(f"\nSummary written to: {summary_file}")