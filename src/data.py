from datasets import load_dataset

def get_dataset():
    dataset = load_dataset("thainq107/iwslt2015-en-vi")
    return dataset

def preprocess_dataset(dataset, tokenizer, max_len=75):
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