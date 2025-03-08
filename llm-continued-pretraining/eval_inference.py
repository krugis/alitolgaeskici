import torch
from unsloth import FastLanguageModel  # Import Unsloth FIRST
from transformers import AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.table import Table

# Initialize console for pretty printing
console = Console()

# Model configurations
model_name = "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit"
max_seq_length = 2048
dtype = torch.float16
load_in_4bit = True
fine_tuned_model_path = "/home/trainer011/training-output/checkpoint-9615"  # Change to actual path

# Load Base Model
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

base_model = FastLanguageModel.for_inference(base_model)

# Load Fine-Tuned Model
fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
fine_tuned_model = FastLanguageModel.for_inference(fine_tuned_model)

# Function to generate text from a model
def generate_text(model, tokenizer, prompt, max_new_tokens=150):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)

    attention_mask = inputs.get("attention_mask", None)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,  # Reduce randomness
            top_p=0.8,        # Control diversity
            repetition_penalty=1.2,  # Avoid excessive repetition
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# Example prompt (can be modified)
prompt = """The specification of alarms is contained in so called "information elements" in the stage 2 level specifications"""

# Generate text from both models
base_output = generate_text(base_model, tokenizer, prompt)
fine_tuned_output = generate_text(fine_tuned_model, tokenizer, prompt)

# Display results in a formatted table
table = Table(title="Model Output Comparison")
table.add_column("Model", justify="left", style="cyan", no_wrap=True)
table.add_column("Generated Text", justify="left", style="magenta")

table.add_row("Base Model", base_output)
table.add_row("Fine-Tuned Model", fine_tuned_output)

console.print(table)
