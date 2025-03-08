from datasets import load_dataset
from transformers import AutoTokenizer

def load_data(split="train"):
    # Load the full dataset
    dataset = load_dataset("atekrugis/etsi-AI", split="train")

    # Split into 90% train and 10% eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit")  

    EOS_TOKEN = tokenizer.eos_token

    if split == "train":
        return train_dataset
    elif split == "eval":
        return eval_dataset
    else:
        raise ValueError("Invalid split. Choose either 'train' or 'eval'.")