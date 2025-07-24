# train_mt.py
import os
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)
import evaluate
import numpy as np

# 1. Configuration
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG = "en_XX"
TGT_LANG = "vi_VN"
MAX_LEN = 75
BATCH_SIZE = 32
EPOCHS = 3
OUTPUT_DIR = "./mbart50_envi_finetuned"

# 2. Dataset loading
dataset = load_dataset("thainq107/iwslt2015-en-vi")

# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 4. Preprocessing
def preprocessing(examples):
    inputs = [ex for ex in examples["en"]]
    targets = [ex for ex in examples["vi"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=MAX_LEN, truncation=True, padding="max_length")["input_ids"]
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    model_inputs["labels"] = labels
    return model_inputs

tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

# 5. Apply preprocessing for dataset
tokenized_ds = dataset.map(preprocessing, batched=True)

# 6. Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 7. BLEU Evaluation
bleu = evaluate.load("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]
    score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": score["score"]}

# 8. Trainer setup
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir="./logs",
    logging_steps=1000,
    predict_with_generate=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_total_limit=1,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 9. Train
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Training complete. Model saved.")
