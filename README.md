# Fine-Tuning a Large Language Model (LLM)

## Description
This project demonstrates the process of fine-tuning a large language model (LLM) using a curated dataset. The fine-tuning enhances the model's ability to generate coherent, on-topic, and logically structured responses. It contrasts the outputs of the base model and the fine-tuned model, highlighting improvements in quality, depth, and focus.

## Features
- Dataset filtering and selection
- Tokenization and preparation of data for training
- Fine-tuning using Hugging Face's Transformers library
- Logging and experiment tracking with Weights & Biases (WandB)
- Evaluation of fine-tuned model performance

## Requirements
The following Python libraries are required:
- `torch`
- `transformers`
- `datasets`
- `wandb`

## Dataset
The project uses a dataset from Hugging Face's `datasets` library. It is filtered for specific formats (e.g., `textbook_academic_tone`) and split into training and testing subsets. I only chose a sample of the subset due to resource limitation.

## Additional Resources

- Model: [H2O-Danube3-500M-Chat](https://huggingface.co/h2oai/h2o-danube3-500m-chat)

- Dataset: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia

## Workflow
1. **Environment Setup**: Import libraries and check GPU availability.
2. **Dataset Preparation**: Load and filter the dataset, then split it into training and testing sets.
3. **Tokenizer and Model Loading**: Load a pre-trained tokenizer and model, and prepare the dataset for tokenization.
4. **Fine-Tuning**:
   - Define training arguments and initialize a trainer using Hugging Face's `Trainer` API.
   - Train the model and save the fine-tuned version.
   - Use WandB for tracking training metrics.
5. **Evaluation**:
   - Generate responses using the fine-tuned model and compare them with the base model.
   - Evaluate on specific prompts for performance analysis.

## Key Code Snippets
### Tokenization
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    combined_text = example["text"]
    tokenized = tokenizer(
        combined_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized
```

### Training
```python
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    save_steps=1000,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    report_to="wandb"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_test_ds,
    tokenizer=tokenizer
)
trainer.train()
trainer.save_model("./fine_tuned_model")
```

### Inference
```python
def generate_output(model, tokenizer, prompt, max_length=200):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    output = model.generate(input_ids, max_length=max_length, do_sample=True)
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
```

## Results
- Please find the model results in this report
- The fine-tuned model stays on topic more consistently than the base model.
- Responses from the fine-tuned model are better organized and more detailed.
- Fine-tuning improves the model's ability to provide example-driven and formal responses.

## Future Work
- Extend the evaluation to additional datasets and prompts.
- Explore different hyperparameter settings for further improvements.
- Integrate additional evaluation metrics.

## Acknowledgments
- Hugging Face Transformers and Datasets libraries
- Weights & Biases for experiment tracking

