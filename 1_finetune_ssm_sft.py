from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
import torch
import os
import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_id = "Zyphra/Zamba2-7B-Instruct-v2"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    #attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right' # to prevent warnings

# Apply to both train and validation folders
training_dir_json = os.path.join("preprocessed", "training")
os.makedirs(training_dir_json, exist_ok=True)

merged_train_path_json = os.path.join(training_dir_json, "mapped_train_samples.json")
merged_val_path_json = os.path.join(training_dir_json, "mapped_val_samples.json")
dataset = load_dataset("json", data_files={"train": merged_train_path_json, "validation": merged_val_path_json})
dataset["train"] = dataset["train"].shuffle(seed=42)
dataset["validation"] = dataset["validation"].shuffle(seed=42)

lora_rank = 16
peft_config = LoraConfig(
        lora_alpha=lora_rank,
        lora_dropout=0.05,
        r=lora_rank,
        bias="none",
        target_modules=[
            "in_proj", "out_proj"
        ],
        task_type="CAUSAL_LM", 
)

train_bs = 2
grad_acc = 4
epochs = 3
warmup_ratio = 0.1
lr = 5e-5
lr_scheduler_type = "cosine_with_restarts"
run_name = "zamba2-runname"

wandb.init(project="finetune zamba2", name=run_name, config={
    "model": "Zamba2-7B-Instruct-v2",
    "dataset": "custom-32000",
    "lr": lr,
    "batch_size": train_bs,
    "lora_r": lora_rank,
})

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    max_seq_length = max_seq_length,
    peft_config=peft_config,
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
        eval_steps = 250,
        max_steps = -1,
        learning_rate = lr,
        gradient_checkpointing=True,
        fp16 = True,
        bf16 = False,
        logging_steps = 50,
        logging_first_step=True,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = lr_scheduler_type,
        seed = 3407,
        output_dir = "ssm-checkpoints",
        report_to = "wandb",
        run_name = run_name,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        save_strategy = "steps",
        save_steps = 250,
        save_total_limit = 2,
    ),
)

trainer_stats = trainer.train()
trainer.save_model(run_name)
tokenizer.save_pretrained(run_name)
print("Saved model")