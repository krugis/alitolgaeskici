import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataset_loader import load_and_split_dataset

# Configuration
dataset_name = 'atekrugis/etsi-AI'
column_name = 'text'
train_size = 60
val_size = 30
eval_size = 10
num_samples = 1000
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
    #num_samples=num_samples #when null whole dataset
)

# Load tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()
model.enable_input_require_grads()  # required for peft.
model = prepare_model_for_kbit_training(model)

# Lora config
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
    return tokenizer(examples[column_name], padding='max_length', truncation=True, max_length=256)  # Reduce to 256.

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
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy='steps',
    eval_steps=1000,
    save_strategy='epoch',
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    push_to_hub=False,
    logging_dir='/home/trainer011/logs',
    logging_steps=1000, #adjust logging steps
    load_best_model_at_end=False,
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_accumulation_steps=8,
    # use_reentrant=False,
    # use_cache=False
    report_to="wandb", #Optional: Use Weights & Biases for monitoring
)

# Trainer
trainer = Trainer(
    model=model,  
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)

# Train the model
train_results = trainer.train()

# Get training history
train_history = trainer.state.log_history

# Extract losses
train_losses = [log['loss'] for log in train_history if 'loss' in log]
eval_losses = [log['eval_loss'] for log in train_history if 'eval_loss' in log]

# Extract steps for train and eval
train_steps = [log['step'] for log in train_history if 'loss' in log]
eval_steps = [log['step'] for log in train_history if 'eval_loss' in log]

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_steps, train_losses, label='Training Loss')
plt.plot(eval_steps, eval_losses, label='Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.show()

# Save the trained model
trainer.save_model(output_dir)

# Print all train losses
print(f"Lora model saved to {output_dir}")