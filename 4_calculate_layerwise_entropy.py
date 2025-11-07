from tqdm import tqdm
from datasets import load_dataset
import os
import json
import glob
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import torch.nn.functional as F

def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        # detach in case it requires grad, ensure on CPU, and convert
        if obj.dim() == 0:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, set):
        return [to_serializable(x) for x in obj]  # JSON has no sets
    else:
        return obj
        
LN_2 = 0.69314718056  # ln(2)
def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy
    
def get_final_norm(model):
    # try common names across families
    for attr in [
        "norm",                    # LLaMA
        "final_layernorm",         # some Mamba/SSM impls
        "final_layer_norm",        # GPT-NeoX style
        "embedding_norm",          # some LFM/Zamba variants
    ]:
        m = getattr(getattr(model, "model", model), attr, None)
        if m is not None:
            return m
    return None


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_in_4bit = False

model_name = "MODEL_NAME"
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit= load_in_4bit, device_map="auto", torch_dtype="bfloat16", trust_remote_code=True, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_name)

final_norm = get_final_norm(model)

folder_name = "test" # test/unknown/paraphrase folder
training_test_dir_json = os.path.join("preprocessed", folder_name) 
predictions_dir = "predictions_per_json"
internal_dir = "internal"
model_path = os.path.join(predictions_dir, internal_dir, model_name)
os.makedirs(model_path, exist_ok=True)

# Loop through each mapped test file
for file_path in glob.glob(os.path.join(training_test_dir_json, "*.json")):
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]

    # Load eval dataset
    eval_dataset = load_dataset("json", data_files={"validation": file_path}, split="validation")
    eval_dataset = eval_dataset.select(range(min(1000, len(eval_dataset))))
    
    print(f"\nEvaluating {filename} — {len(eval_dataset)} samples")
    
    y_pred, y_true = [], []

    # Go through each datapoint
    for dp in tqdm(eval_dataset, desc=f"Evaluating {filename}"):
        for message in dp["conversations"]:
            ground_truth = next(
                (m["content"] for m in dp["conversations"] if m["role"] == "assistant"), 
                None
            )
        
        
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                input_message = [message]
                
                inputs = tokenizer.apply_chat_template(
                    input_message,
                    tokenize = True,
                    add_generation_prompt = True,
                    return_tensors = "pt",
                ).to(model.device)
                
                with torch.no_grad():
                    gen_out = model.generate(
                        input_ids=inputs,
                        max_new_tokens=32,
                        use_cache=True,
                        output_scores=True,
                        return_dict_in_generate=True,
                        do_sample=True,
                        temperature=0.1,
                        min_p=0.1,
                        repetition_penalty=1.05,
                        num_beams=1
                    )
                    
                    fwd = model(
                        input_ids=gen_out.sequences,          # full prompt + generated tokens
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                        attention_mask=(gen_out.sequences != tokenizer.pad_token_id)
                                               if tokenizer.pad_token_id is not None
                                               else torch.ones_like(gen_out.sequences),
                    )
                
                # ----- DECODE -----
                # For decoder-only models: get only new tokens
                use_toklens = True
                input_len = inputs.shape[1]
                full_len = gen_out.sequences.shape[1]
                new_tokens = gen_out.sequences[:, input_len:]   # strip the prompt
                decoded_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
                
                # ----- HIDDEN STATES -----
                hidden_states = fwd.hidden_states

                layer_logits_varentropy, layer_logits_entropy = [], []
                for index, layer_output in enumerate(hidden_states):
                    h = final_norm(layer_output) if final_norm is not None else layer_output
                    logits = model.lm_head(h)                      # (B, T, V)
                    if use_toklens:
                        logits = logits[:, input_len:full_len, :] # per-layer, per-generated-token entropy and varentropy
                    
                    entropy, varentropy = calculate_varentropy_logsoftmax(logits)
                    ent_mean  = entropy.mean(dim=1).item()
                    varent_mean  = varentropy.mean(dim=1).item()
                    layer_logits_entropy.append(ent_mean)
                    layer_logits_varentropy.append(varent_mean)
                    
                
                # ----- RETURN EVERYTHING -----
                result = {
                    "decoded_texts": decoded_texts,
                    "ground_truth": ground_truth,
                    "total_layers": len(hidden_states),
                    "hidden_states_entropy": layer_logits_entropy,
                    "hidden_states_varentropy": layer_logits_varentropy
                }
                y_pred.append(to_serializable(result))

            # Save to JSON
            result_json_path = os.path.join(predictions_dir, internal_dir, model_name, f"{filename_without_ext}_check_internal_{folder_name}_qs.json")
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump({"result": y_pred}, f, indent=2, ensure_ascii=False)