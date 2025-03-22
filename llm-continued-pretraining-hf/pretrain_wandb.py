import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from transformers.integrations import WandbCallback  # Use the correct WandbCallback
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset_loader import load_and_split_dataset
import wandb

# Initialize wandb
wandb.init(project="llama-3.2-fine-tuning", name="llama-3.2-1b-lora-training")

# Configuration
dataset_name = 'atekrugis/etsi-AI'
column_name = 'text'
train_size = 60
val_size = 30
eval_size = 10
model_name = 'unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit'
output_dir = '/home/trainer011/training-output'
batch_size = 2  # Reduced batch size
learning_rate = 1e-4
num_train_epochs = 2
weight_decay = 0.01

torch.cuda.empty_cache()

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load and split dataset
train_dataset, val_dataset, eval_dataset = load_and_split_dataset(
    dataset_name,
    column_name=column_name,
    train_size=train_size,
    val_size=val_size,
    eval_size=eval_size,
)

# Load tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()
model.enable_input_require_grads()  # required for peft.
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples[column_name], padding='max_length', truncation=True, max_length=256)

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Data collator for causal language modeling
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    run_name="llama-3.2-1b-lora-training",  # Explicitly set run_name
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='steps',
    eval_steps=500,
    save_strategy='epoch',
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    push_to_hub=False,
    logging_dir='/home/trainer011/logs',
    logging_steps=100,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_accumulation_steps=8,
    report_to="wandb",
)

# Trainer
trainer = Trainer(
    model=model,  
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    callbacks=[WandbCallback()],  # Correctly using the built-in WandbCallback
)

# Train the model
train_results = trainer.train()

# Log final metrics
wandb.log({
    "final_train_loss": train_results.training_loss,
    "train_runtime": train_results.metrics["train_runtime"],
    "train_samples_per_second": train_results.metrics["train_samples_per_second"],
})

# Perform a final evaluation
eval_results = trainer.evaluate()
wandb.log({"final_eval_loss": eval_results["eval_loss"]})

# Save the trained model
trainer.save_model(output_dir)
print(f"LoRA model saved to {output_dir}")

# Finish wandb run
wandb.finish()
