# Continual Pretraining of a Quantized LLM with PEFT

## Introduction
This repository includes code for the continual pretraining of a quantized LLM using Parameter-Efficient Fine-Tuning (PEFT) with tools such as Hugging Face, Unsloth, PyTorch, NumPy, Matplotlib, and Pandas. The training data consists of ETSI standards documents, and the objective is to create an LLM capable of generating text for standards documentation.

This code is open-source and freely available for use, reuse, and distribution.

## Continual Pretraining
Continual pretraining is an intermediate stage between initial pretraining and fine-tuning, where a pretrained model is further trained on domain-specific data to improve its adaptability. It is useful for:
- **Domain Adaptation**: Specializing in legal, medical, or scientific texts.
- **Knowledge Updating**: Incorporating recent events and evolving terminology.
- **Low-Resource Tasks**: Training on unlabelled but relevant text when labeled data is scarce.
- **Improving Zero- and Few-Shot Learning**: Enhancing the model’s ability to handle specialized tasks without extensive fine-tuning.
- **Avoiding Catastrophic Forgetting**: Retaining general knowledge while adapting to new data.

Fine-tuning is preferred for optimizing the model for specific tasks, whereas continual pretraining improves broad domain knowledge.

## Process
The overall process is similar to pretraining but differs in data preparation. Continual pretraining utilizes self-supervised, unlabelled data.
![image](https://github.com/user-attachments/assets/092b762f-f0a8-496b-bf57-93d5c6424279)

### Data Collection
![image](https://github.com/user-attachments/assets/50d0eab1-b342-48f7-bb1a-91c05503c9a3)

- **Source**: Active ETSI standards documents mentioning AI, downloaded as PDFs from the ETSI document repository.
- **Scripts**:
  - `download_pdf.py`: Downloads PDFs from a CSV file with URLs, skipping previously downloaded files.
  - `prepare_data.py`: Extracts text, splits it into chunks, and saves structured data in JSONL format.

### Data Cleaning
![image](https://github.com/user-attachments/assets/5a423915-f266-4d5b-bec1-5c861609c64c)

- **Script**: `clear_data.py`
- **Tasks**:
  - Removes unwanted characters (e.g., newlines, tabs, extra spaces, and special characters except punctuation).
  - Ensures `<|end_of_text|>` markers remain for Llama format compatibility.

### Data Formatting
![image](https://github.com/user-attachments/assets/ea2909be-4221-4f19-ac67-3ed278a8a9e4)

- **Script**: `dataset_loader.py`
- **Tasks**:
  - Loads the dataset from Hugging Face.
  - Splits it into training (70%), validation (20%), and test (10%) sets.

- **Script**: `prepare_training.py`
- **Tasks**:
  - Loads and validates datasets.
  - Prints the first 5 entries for verification.

### Pretraining
![image](https://github.com/user-attachments/assets/6118f46e-184e-40dc-9a37-dadb9a8b6187)

- **Script**: `pretrain.py`
- **Model Used**: `unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit`
  - Small model size for efficiency in low-resource environments.
  - Pretrained for instruction-based responses.
  - 4-bit quantization for reduced memory usage.
- **PEFT and QLoRA Adoption**:
  - Fine-tunes a subset of parameters to optimize resource usage and prevent overfitting.

#### Training Parameters:
- **PEFT Configuration**:
  - `r=128`: Rank of LoRA adaptation matrices.
  - `target_modules`: Specifies fine-tuned layers (e.g., `q_proj`, `v_proj`, `lm_head`).
  - `lora_alpha=32`: Scaling factor for adaptation.
  - `lora_dropout=0`: No dropout for efficiency.
  - `bias="none"`: Prevents unnecessary bias parameters.
  - `use_rslora=True`: Enables Rank-Stabilized LoRA (RS-LoRA).
- **Training Configuration**:
  - `dataset_num_proc=8`: Uses 8 parallel processes.
  - `per_device_train_batch_size=2`: Small batch size for GPU efficiency.
  - `gradient_accumulation_steps=8`: Effective batch size of 16.
  - `num_train_epochs=1`: One full pass through the dataset.
  - `learning_rate=5e-5`: Standard for transformer fine-tuning.
  - `lr_scheduler_type="cosine"`: Uses cosine decay.
  - `optim="adamw_8bit"`: Efficient optimizer for quantized models.
- **Checkpointing & Monitoring**:
  - `save_steps=500`, `save_total_limit=3`: Saves checkpoints periodically.
  - `logging_steps=1`: Logs training progress at every step.
  - `report_to="wandb"`: Uses Weights & Biases for tracking.

### Evaluation

The trained model is evaluated in comparison to the base model using **Loss, Perplexity, BLEU, ROUGE, and Embedding Similarity**.

#### **Evaluation Process:**
- The **evaluation dataset size** is controlled by `eval_data_size` (default: `10`).
- The **base model** (`unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit`) is evaluated before applying LoRA adapters.
- The **fine-tuned model** is evaluated after applying LoRA adapters.
- A progress bar tracks evaluation steps.
- **Metrics calculated:**
  - **Loss:** Average token loss over the dataset.
  - **Perplexity:** Exponential of the loss, lower is better.
  - **BLEU Score:** Measures text similarity based on n-gram matches.
  - **ROUGE Score:** Measures recall-oriented overlap with reference texts.
  - **Embedding Similarity:** Cosine similarity between generated and reference embeddings.
- **Model weight changes** are also tracked to determine the percentage of altered weights in fine-tuning.

#### **Results Display:**
- A **rich table** presents evaluation metrics for both models.
- A **bar chart** visualizes model performance differences.
- Evaluation results are saved in `model_comparison.png`.
  ![image](https://github.com/user-attachments/assets/399746d2-ef20-4180-ada0-4218f25637a8)
  ![image](https://github.com/user-attachments/assets/26688baa-092d-4601-aa8d-f4845e5f6ac1)



## Training logs
![image](https://github.com/user-attachments/assets/85624d83-b50e-4f40-bcd5-503920308878)
![image](https://github.com/user-attachments/assets/df27ac43-4663-4866-b329-0386db024452)

The training script logs key results, including loss values and validation performance. For more details, refer to the `logs/` directory after training.

---

### License
This project is licensed under an open-source license. Feel free to use and modify it for your own research and applications.

### Contributions
Contributions and improvements are welcome! If you find any issues or have suggestions, feel free to submit a pull request or open an issue.

### Contact
For any inquiries, reach out via GitHub Issues or Discussions.

---

This `README.md` provides an overview of continual pretraining using a quantized LLM with PEFT. Let me know if you need additional details! 🚀

