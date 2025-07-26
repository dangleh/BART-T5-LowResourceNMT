import torch
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from src.data import get_dataset, preprocess_dataset
from src.model import load_model, load_tokenizer
from src.evaluate import compute_metrics
import logging
logging.basicConfig(level=logging.INFO)

def main():
    import evaluate
    import os

    MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
    OUTPUT_DIR = "./outputs/mbart50_envi_finetuned"
    BATCH_SIZE = 16
    EPOCHS = 3
    MAX_LEN = 75
    SRC_LANG = "en_XX"
    TGT_LANG = "vi_VN"

    os.environ["WANDB_DISABLED"] = "true"

    tokenizer = load_tokenizer(MODEL_NAME)
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG

    dataset = get_dataset()
    preprocessed_ds = preprocess_dataset(dataset, tokenizer, max_len=MAX_LEN)

    model = load_model(MODEL_NAME)
    model.gradient_checkpointing_enable()

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir="./logs",
        logging_steps=1000,
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        disable_tqdm=False,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("sacrebleu")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_ds["train"],
        eval_dataset=preprocessed_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, metric)
    )

    logging.info("Start training...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR + "/model")
    logging.info("Training complete. Model saved.")

if __name__ == "__main__":
    main()