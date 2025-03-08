import os
from unsloth import FastLanguageModel, is_bfloat16_supported, UnslothTrainer, UnslothTrainingArguments
import torch
from load_data import load_data
from transformers import TrainingArguments
from trl import SFTTrainer

OUTPUT_DIR = "/home/trainer011/training-output"
CHECKPOINT_DIR = OUTPUT_DIR

# Find the latest checkpoint if exists
last_checkpoint = None
if os.path.isdir(CHECKPOINT_DIR):
    checkpoints = [os.path.join(CHECKPOINT_DIR, d) for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"✅ Found checkpoint. Resuming from: {last_checkpoint}")
    else:
        print("ℹ️ No checkpoint found. Starting from scratch.")
else:
    print("ℹ️ No checkpoint directory found. Starting from scratch.")

# Model setup (same as before)
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
    loftq_config=None,
)

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=load_data("train"),
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=8,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,
        embedding_learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.00,
        logging_steps=1,
        report_to="none",
        seed=3407,
        output_dir=OUTPUT_DIR,
        save_strategy="steps",      # Save every N steps
        save_steps=500,             # Save every 500 steps (adjust as needed)
        save_total_limit=3,         # Keep only last 3 checkpoints
    ),
)

trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
