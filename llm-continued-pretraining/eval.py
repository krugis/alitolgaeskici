import torch
from unsloth import FastLanguageModel  # Import Unsloth FIRST
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import load as load_metric
from sentence_transformers import SentenceTransformer
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
from peft import PeftModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting evaluation...")

console = Console()
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Control variable to set the evaluation data size
eval_data_size = 10  # You can adjust this value to control the size of eval dataset

# Load evaluation data
from load_data import load_data
eval_dataset = load_data("eval").shuffle(seed=42).select(range(eval_data_size))

# Load base model
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
    max_seq_length=2048,
)

# Prepare the model for inference
base_model = FastLanguageModel.for_inference(base_model)

# Load fine-tuned model by applying the adapter to the base model
fine_tuned_model = PeftModel.from_pretrained(
    base_model,
    "/home/trainer011/training-output/checkpoint-9615"
)

# Prepare the fine-tuned model for inference
fine_tuned_model = FastLanguageModel.for_inference(fine_tuned_model)

# Fine-tuned model uses the same tokenizer as the base model
fine_tuned_tokenizer = base_tokenizer

def evaluate_model(name, model, tokenizer, dataset, batch_size=8):
    losses, generated_texts, references = [], [], []
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeElapsedColumn()
    )
    
    with progress:
        task = progress.add_task(f"Evaluating {name}", total=len(dataset['text']))
        for i in range(0, len(dataset['text']), batch_size):
            batch_texts = dataset['text'][i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=2048).to("cuda")
            
            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss
            
            with torch.no_grad():
                outputs = model(**inputs, labels=labels)
                losses.extend([outputs.loss.item()] * len(batch_texts))
                generated_ids = model.generate(
                    inputs["input_ids"], max_new_tokens=100, do_sample=True, attention_mask=inputs["attention_mask"]
                )
                generated_texts.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
                references.extend(batch_texts)
            progress.update(task, advance=len(batch_texts))

    mean_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(mean_loss)).item()    
    bleu = load_metric("bleu").compute(predictions=generated_texts, references=[[ref] for ref in references])["bleu"]  
    rouge = load_metric("rouge").compute(predictions=generated_texts, references=references)
    gen_embeddings = embedder.encode(generated_texts, convert_to_tensor=True)
    ref_embeddings = embedder.encode(references, convert_to_tensor=True)
    embedding_similarity = torch.nn.functional.cosine_similarity(gen_embeddings, ref_embeddings, dim=-1).mean().item()

    return {
        "Loss": mean_loss,
        "Perplexity": perplexity,
        "BLEU": bleu,
        "ROUGE": rouge["rougeL"],
        "EmbeddingSimilarity": embedding_similarity
    }

# Function to compute percentage of changed/added weights
def compute_weight_change(base_model, fine_tuned_model):
    total_params = 0
    changed_params = 0

    for name, param in fine_tuned_model.named_parameters():
        if any(layer in name for layer in ["lora", "LoRA", "adapter"]):  # Check LoRA-related layers
            total_params += param.numel()
            changed_params += (param.abs() > 1e-6).sum().item()  # Count significantly changed values

    if total_params == 0:
        return 0, 0  # Avoid division by zero if no LoRA weights found

    changed_percentage = (changed_params / total_params) * 100
    return changed_percentage, changed_params

logging.info("Evaluating base model...")
base_metrics = evaluate_model("Base Model", base_model, base_tokenizer, eval_dataset)

logging.info("Evaluating fine-tuned model...")
fine_tuned_metrics = evaluate_model("Fine-Tuned Model", fine_tuned_model, fine_tuned_tokenizer, eval_dataset)

# Compute weight changes
changed_percentage, changed_params = compute_weight_change(base_model, fine_tuned_model)

# Display metrics in a table
table = Table(title="Model Evaluation Metrics")
table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
table.add_column("Base Model", justify="right", style="magenta")
table.add_column("Fine-Tuned Model", justify="right", style="green")

for metric in ["Loss", "Perplexity", "BLEU", "ROUGE", "EmbeddingSimilarity"]:
    table.add_row(
        metric,
        f"{base_metrics[metric]:.4f}",
        f"{fine_tuned_metrics[metric]:.4f}"
    )

table.add_row("Changed Weights (%)", "-", f"{changed_percentage:.2f}%")
table.add_row("Total Changed Weights", "-", f"{changed_params:,}")  # Display with comma separator
console.print(table)

# Display comparison graph
metrics_names = ["Loss", "Perplexity", "BLEU", "ROUGE", "EmbeddingSimilarity"]
base_scores = [base_metrics[m] for m in metrics_names]
fine_tuned_scores = [fine_tuned_metrics[m] for m in metrics_names]

plt.figure(figsize=(10, 6))
x = range(len(metrics_names))
plt.bar(x, base_scores, width=0.4, label='Base Model', align='center', color='purple')
plt.bar([i + 0.4 for i in x], fine_tuned_scores, width=0.4, label='Fine-Tuned Model', align='center', color='green')
plt.xticks([i + 0.2 for i in x], metrics_names, rotation=30)
plt.ylabel("Score")
plt.title("Model Comparison Metrics")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
plt.savefig('model_comparison.png')
