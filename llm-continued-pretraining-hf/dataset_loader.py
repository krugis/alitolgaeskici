from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

def load_and_split_dataset(dataset_name: str, column_name: str = 'text', 
                           train_size: int = 80, val_size: int = 10, eval_size: int = 10,
                           num_samples: int = None):
    """
    Load a dataset from Hugging Face and split it into train, validation, and evaluation sets.

    Args:
    - dataset_name (str): Name of the dataset to load.
    - column_name (str): The column that contains the text data (default is 'text').
    - train_size (int): Percentage of data to use as the train set.
    - val_size (int): Percentage of data to use as the validation set.
    - eval_size (int): Percentage of data to use as the test/evaluation set.
    - num_samples (int, optional): Maximum number of samples to use from the dataset.

    Returns:
    - train_dataset: The training dataset.
    - val_dataset: The validation dataset.
    - eval_dataset: The evaluation (test) dataset.
    """
    # Ensure total size is 100
    #if train_size + val_size + eval_size != 100:
    #    raise ValueError("The sum of train_size, val_size, and eval_size must be 100.")

    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name)
    
    # Check if the dataset contains the column of interest
    if column_name not in dataset['train'].column_names:
        raise ValueError(f"The dataset does not contain a column named '{column_name}'.")

    # Extract the 'text' column properly
    texts = dataset['train'][column_name]  

    # Limit dataset size if num_samples is provided
    if num_samples is not None:
        texts = texts[:num_samples]


    # Convert percentage values to actual sample counts
    total_samples = len(texts)
    train_count = (train_size * total_samples) // 100
    val_count = (val_size * total_samples) // 100
    eval_count = total_samples - train_count - val_count  # Remaining for eval

    # Split dataset
    train_texts, temp_texts = train_test_split(texts, train_size=train_count, random_state=42)
    val_texts, eval_texts = train_test_split(temp_texts, train_size=val_count, random_state=42)

    # Create new datasets
    train_dataset = Dataset.from_dict({column_name: train_texts})
    val_dataset = Dataset.from_dict({column_name: val_texts})
    eval_dataset = Dataset.from_dict({column_name: eval_texts})

    return train_dataset, val_dataset, eval_dataset
