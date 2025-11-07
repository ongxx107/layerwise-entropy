from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import os
import wandb

model_id = "LiquidAI/LFM2-1.2B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
    # attn_implementation="flash_attention_2" # uncomment on compatible GPU
)

print("Local model loaded successfully!")
print(f"Parameters: {model.num_parameters():,}")
print(f"Vocab size: {len(tokenizer)}")
print(f"Model size: ~{model.num_parameters() * 2 / 1e9:.1f} GB (bfloat16)")

#dataset
training_dir_json = os.path.join("preprocessed", "training")
os.makedirs(training_dir_json, exist_ok=True)
merged_train_path_json = os.path.join(training_dir_json, "mapped_train_samples.json")
merged_val_path_json = os.path.join(training_dir_json, "mapped_val_samples.json")
dataset = load_dataset("json", data_files={"train": merged_train_path_json, "validation": merged_val_path_json})
dataset["train"] = dataset["train"].shuffle(seed=42)
dataset["validation"] = dataset["validation"].shuffle(seed=42)

#peft
GLU_MODULES = ["w1", "w2", "w3"]
MHA_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
CONV_MODULES = ["in_proj", "out_proj"]

lora_rank = 16
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_rank,
    lora_alpha=lora_rank,
    lora_dropout=0.05,
    target_modules=GLU_MODULES + MHA_MODULES + CONV_MODULES,
    bias="none",
    modules_to_save=None,
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

print("LoRA configuration applied!")
print(f"LoRA rank: {lora_config.r}")
print(f"LoRA alpha: {lora_config.lora_alpha}")
print(f"Target modules: {lora_config.target_modules}")


#trainer
train_bs = 2
grad_acc = 5
eval_bs = 2
epochs = 3
warmup_ratio = 0.0
lr = 5e-5
lr_scheduler_type = "cosine_with_restarts"
run_name = "lfm2-1.2b-runname"

wandb.init(project="finetune lfm", name=run_name, config={
    "model": "lfm-700m",
    "dataset": "custom-32000",
    "lr": lr,
    "batch_size": train_bs,
    "lora_r": lora_rank,
})

lora_sft_config = SFTConfig(
    num_train_epochs = epochs,
    per_device_train_batch_size = train_bs,
    gradient_accumulation_steps = grad_acc,
    per_device_eval_batch_size = eval_bs,
    learning_rate = lr,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = lr_scheduler_type,
    warmup_ratio = warmup_ratio,
    logging_steps = 50,
    save_strategy = "steps",
    save_steps = 200,
    save_total_limit = 2,
    eval_strategy = "steps",
    eval_steps = 200,
    max_steps = -1,
    load_best_model_at_end=True,
    metric_for_best_model = "eval_loss",
    greater_is_better = False,
    output_dir = "lfm-checkpoints",
    report_to = "wandb",
    run_name = run_name,
    max_seq_length=4096,
    seed = 3407,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

print("Creating LoRA SFT trainer...")
lora_sft_trainer = SFTTrainer(
    model=lora_model,
    args=lora_sft_config,
    train_dataset = dataset["train"],
    eval_dataset = dataset["validation"],
    processing_class=tokenizer,
)

print("Starting LoRA + SFT training...")
lora_sft_trainer.train()

print("LoRA + SFT training completed!")

lora_sft_trainer.save_model(run_name)
tokenizer.save_pretrained(run_name)
print("Saved model")