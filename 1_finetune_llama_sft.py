from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from unsloth import is_bfloat16_supported
from datasets import load_dataset, DatasetDict, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback

import torch
import glob
import os
import json
import random
import wandb
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

lora_rank = 16
lora_dropout = 0.07
# add lora weight decay
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_rank,
    lora_dropout = lora_dropout,
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# Apply to both train and validation folders
training_dir_json = os.path.join("preprocessed", "training")
os.makedirs(training_dir_json, exist_ok=True)

merged_train_path_json = os.path.join(training_dir_json, "merged_train_samples.json")
merged_val_path_json = os.path.join(training_dir_json, "merged_val_samples.json")
dataset = load_dataset("json", data_files={"train": merged_train_path_json, "validation": merged_val_path_json})

dataset["train"] = dataset["train"].shuffle(seed=42)
dataset["validation"] = dataset["validation"].shuffle(seed=42)

dataset["train"] = standardize_sharegpt(dataset["train"])
dataset["validation"] = standardize_sharegpt(dataset["validation"])
dataset = dataset.map(formatting_prompts_func, batched = True,)

train_bs = 8
grad_acc = 4
epochs = 3
warmup_ratio = 0.1
lr = 2e-5
lr_scheduler_type = "cosine_with_restarts"
run_name = "llama-8b-runname"

wandb.init(project="finetune llama 3.1 8b", name=run_name, config={
    "model": "LLaMA 3.1 8B",
    "dataset": "custom-32000",
    "lr": lr,
    "batch_size": train_bs,
    "lora_r": lora_rank,
    
    "lr_scheduler_type": lr_scheduler_type,
    "gradient_accumulation_steps": grad_acc,
    "warmup_ratio": warmup_ratio,
    "lora_dropout": lora_dropout,
})

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    args = TrainingArguments(
        per_device_train_batch_size = train_bs,
        gradient_accumulation_steps = grad_acc,
        per_device_eval_batch_size = 1,
        warmup_ratio = warmup_ratio,
        num_train_epochs = epochs,
        eval_strategy = "steps",
        eval_steps = 50,
        max_steps = -1,
        learning_rate = lr,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = lr_scheduler_type,
        seed = 3407,
        output_dir = "llama3-checkpoints",
        report_to = "wandb",
        run_name = run_name,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        save_strategy = "steps",
        save_steps = 50,
        save_total_limit = 2,
    ),
)

trainer_stats = trainer.train()
trainer.save_model(run_name)
tokenizer.save_pretrained(run_name)
print("Saved model")