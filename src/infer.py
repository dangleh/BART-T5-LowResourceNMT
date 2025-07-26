from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def translate(text, model_dir, max_length=75, num_beams=5):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    inputs['input_ids'][0,0] = tokenizer.lang_code_to_id["en_XX"]
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["vi_VN"],
        num_beams=num_beams,
        max_length=max_length
    )
    vi_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return vi_text

if __name__ == "__main__":
    text = "Why does train Deep Learning model take time?"
    model_dir = "./outputs/mbart50_envi_finetuned/model"
    print("Vietnamese translation:", translate(text, model_dir))