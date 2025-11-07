import json
import os
import glob
from datasets import load_dataset
from tqdm import tqdm

training_test_dir_json = os.path.join("preprocessed", "new_combination", "json", "paraphrase")
output_path = os.path.join(training_test_dir_json, "filtered")
os.makedirs(output_path, exist_ok=True)

THRESH = 0.8
SAMPLE_LIMIT = 1000

def conversation_passes(conv, thresh=THRESH):
    # Include the conversation if ANY message in it has led_bertscore > thresh
    return any(
        isinstance(msg.get("led_bertscore"), (int, float)) and msg["led_bertscore"] > thresh
        for msg in conv
    )

# Loop through each mapped validation file
for file_path in glob.glob(os.path.join(training_test_dir_json, "*.json")):
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    
    # Load JSON
    with open(file_path, "r") as f:
        data = json.load(f)

    filtered_samples = []
    for conv in data.get("conversations", []):
        if conversation_passes(conv):
            # Transform each message:
            # Replace content with new_content
            # Remove new_content
            # Remove led_bertscore
            transformed = []
            for msg in conv:
                msg = dict(msg)  # copy
                if "new_content" in msg:
                    msg["content"] = msg["new_content"]
                    del msg["new_content"]
                msg.pop("led_bertscore", None)
                transformed.append(msg)
            filtered_samples.append(transformed)
            if len(filtered_samples) >= SAMPLE_LIMIT:
                break
    
    output_path_json = os.path.join(output_path, f"{filename_without_ext}_filtered.json")
    with open(output_path_json, "w", encoding="utf-8") as f:
        json.dump({"conversations": filtered_samples}, f, indent=2, ensure_ascii=False)