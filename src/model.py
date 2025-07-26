from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def load_model(model_name):
    return AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_translator():
    """
    Initializes and returns a Hugging Face translation pipeline for English-to-Vietnamese translation using a fine-tuned mBART-50 model.

    Returns:
        transformers.pipelines.Pipeline: A translation pipeline configured for translating from English ('en_XX') to Vietnamese ('vi_VN').
    """
    MODEL = "dangleh/mbart50_envi_finetuned"
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="en_XX", tgt_lang="vi_VN")
    return translator