from datasets import load_dataset

def get_dataset():
    dataset = load_dataset("thainq107/iwslt2015-en-vi")
    return dataset

def preprocess_dataset(dataset, tokenizer, max_len=75):
    """
    Preprocesses a Hugging Face dataset for sequence-to-sequence tasks using a tokenizer.

    This function tokenizes the source ("en") and target ("vi") language fields of the dataset,
    applies padding and truncation to a specified maximum length, and prepares the labels for
    training by replacing padding token IDs with -100 (to be ignored by the loss function).

    Args:
        dataset (datasets.Dataset): The Hugging Face dataset to preprocess, containing "en" and "vi" fields.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the text.
        max_len (int, optional): The maximum sequence length for padding and truncation. Defaults to 75.

    Returns:
        datasets.Dataset: The preprocessed dataset with "input_ids" and "labels" fields.
    """
    def preprocessing(examples):
        input_ids = tokenizer(
            examples["en"], padding="max_length", truncation=True, max_length=max_len
        )["input_ids"]
        labels = tokenizer(
            examples["vi"], padding="max_length", truncation=True, max_length=max_len
        )["input_ids"]
        labels = [
            [-100 if item == tokenizer.pad_token_id else item for item in label]
            for label in labels
        ]
        return {"input_ids": input_ids, "labels": labels}
    return dataset.map(preprocessing, batched=True)