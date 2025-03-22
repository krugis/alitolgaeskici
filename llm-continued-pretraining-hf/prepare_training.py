from dataset_loader import load_and_split_dataset  # Import the function from the dataset_loader module

def display_first_five_data(train_dataset, val_dataset, eval_dataset):
    """
    Function to display the first 5 entries from train, validation, and evaluation datasets.
    """
    print("First 5 entries of the Training Dataset:")
    for i in range(5):
        text = train_dataset[i]['text']  
        print(text[:50] + "..." if len(text) > 50 else text)
    print(f"Total items in train_dataset: {len(train_dataset)}\n")

    print("\nFirst 5 entries of the Validation Dataset:")
    for i in range(5):
        text = val_dataset[i]['text']  
        print(text[:50] + "..." if len(text) > 50 else text)
    print(f"Total items in val_dataset: {len(val_dataset)}\n")

    print("\nFirst 5 entries of the Evaluation Dataset:")
    for i in range(5):
        text = eval_dataset[i]['text']  
        print(text[:50] + "..." if len(text) > 50 else text)
    print(f"Total items in eval_dataset: {len(eval_dataset)}\n")


# Parameters
dataset_name = 'atekrugis/etsi-AI'  # Replace with the actual dataset name
train_size = 70   # 70% for training
val_size = 20    # 20% for validation
eval_size = 10    # 10% for evaluation
num_samples = 1000  # Limit dataset size to 1000 samples

# Load and split dataset
train_dataset, val_dataset, eval_dataset = load_and_split_dataset(
    dataset_name, 
    train_size=train_size, 
    val_size=val_size, 
    eval_size=eval_size,
    num_samples=num_samples
)

# Display first 5 entries from each dataset
display_first_five_data(train_dataset, val_dataset, eval_dataset)