import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src import config

def validate_input(text):
    if not text.strip():
        return False, "Input is empty."
    if len(text) > config.INPUT_MAX_CHARS:
        return False, f"Input too long (>{config.INPUT_MAX_CHARS} chars)."
    if any(ord(c) < 32 and c not in '\n\t' for c in text):
        return False, "Input contains invalid control characters."
    return True, ""

def translate(text, model_dir=None, max_length=None, num_beams=None):
    """
    Translates input English text to Vietnamese using a pretrained sequence-to-sequence model.
    """
    if model_dir is None:
        model_dir = config.MODEL_NAME
    if max_length is None:
        max_length = config.MAX_LENGTH
    if num_beams is None:
        num_beams = config.NUM_BEAMS
    valid, msg = validate_input(text)
    if not valid:
        logging.warning(f"Input validation failed: {msg}")
        raise ValueError(msg)
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs['input_ids'][0,0] = tokenizer.lang_code_to_id[config.SRC_LANG]
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[config.TGT_LANG],
            num_beams=num_beams,
            max_length=max_length
        )
        vi_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Translation success. Input: {text}")
        return vi_text
    except Exception as e:
        logging.error(f"Translation error: {e}")
        raise

if __name__ == "__main__":
    import os
    text = "Why does train Deep Learning model take time?"
    try:
        print("Vietnamese translation:", translate(text))
    except Exception as e:
        print(f"Error: {e}")