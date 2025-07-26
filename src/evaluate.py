import numpy as np

def compute_metrics(eval_preds, tokenizer, metric):
    """
    Compute BLEU score for evaluation predictions.
    Args:
        eval_preds: Tuple of (predictions, labels)
        tokenizer: Tokenizer object
        metric: Evaluation metric (e.g., sacrebleu)
    Returns:
        dict: BLEU score
    """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}