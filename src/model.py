

import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from src import config

def load_tokenizer(model_name=None):
    """
    Loads and returns a tokenizer for the specified model.
    """
    if model_name is None:
        model_name = config.MODEL_NAME
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"Loaded tokenizer: {model_name}")
        return tokenizer
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        raise

def load_model(model_name=None):
    """
    Loads a sequence-to-sequence language model from the Hugging Face Transformers library.
    """
    if model_name is None:
        model_name = config.MODEL_NAME
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        logging.info(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def get_translator():
    """
    Initializes and returns a Hugging Face translation pipeline for English-to-Vietnamese translation using a fine-tuned mBART-50 model.
    """
    try:
        model = load_model()
        tokenizer = load_tokenizer()
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=config.SRC_LANG,
            tgt_lang=config.TGT_LANG
        )
        logging.info("Translation pipeline initialized.")
        return translator
    except Exception as e:
        logging.error(f"Error initializing translation pipeline: {e}")
        raise