from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
import glob

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
max_seq_length = 4096
dtype = None
load_in_4bit = True

model_name = "MODEL_NAME" # e.g.: "kyloren1989/LFM2-1.2B-lexglue"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    torch_dtype="bfloat16", 
    trust_remote_code=True, 
    # attn_implementation="flash_attention_2"
)

training_test_dir_json = os.path.join("preprocessed", "test")

predictions_dir = "predictions_per_json"
model_path = os.path.join(predictions_dir, model_name)
os.makedirs(model_path, exist_ok=True)

# Loop through each mapped validation file
for file_path in glob.glob(os.path.join(training_test_dir_json, "*.json")):
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]

    # Load eval dataset
    eval_dataset = load_dataset("json", data_files={"validation": file_path}, split="validation")
    print(f"\nEvaluating {filename} — {len(eval_dataset)} samples")
    
    y_pred, y_true = [], []

    # Go through each datapoint
    for dp in tqdm(eval_dataset, desc=f"Evaluating {filename}"):
        for message in dp["conversations"]:
            role = message["role"]
            content = message["content"]

            if role == "user":
                input_message = [message]
                chat_sample = tokenizer.apply_chat_template(
                                                                input_message, 
                                                                add_generation_prompt=True,
                                                                return_tensors="pt",
                                                                tokenize=True,
                                                            ).to(model.device)
                
                # Tokenize input and generate output
                input_ids = chat_sample
                outputs = model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=0.1,
                    min_p=0.1,
                    repetition_penalty=1.05,
                    max_new_tokens=32,
                    num_beams=1,
                )
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False).split('<|im_start|>assistant')[-1].replace('<|im_end|>', '').replace('\n', '').replace('://', '').strip()
            
                y_pred.append(decoded_output)
            elif role == "assistant":
                y_true.append(content.strip())

    # Save to JSON
    result_json_path = os.path.join(predictions_dir, model_name, f"{filename_without_ext}_predictions.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump({"y_pred": y_pred, "y_true": y_true}, f, indent=2, ensure_ascii=False)

    # Save comparison txt
    result_txt_path = os.path.join(predictions_dir, model_name, f"{filename_without_ext}_comparison.txt")
    with open(result_txt_path, "w", encoding="utf-8") as f:
        for idx, (pred, truth) in enumerate(zip(y_pred, y_true), start=1):
            f.write(f"### Sample {idx}\n")
            f.write(f"Predicted: {pred}\n")
            f.write(f"Ground Truth: {truth}\n")
            f.write("=" * 80 + "\n")