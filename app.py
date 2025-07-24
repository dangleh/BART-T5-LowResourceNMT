import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.title("English → Vietnamese Translation (mBART50)")

@st.cache_resource
def load_models():
    model = AutoModelForSeq2SeqLM.from_pretrained("model")
    tokenizer = AutoTokenizer.from_pretrained("model")
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="en_XX", tgt_lang="vi_VN")
    return translator

translator = load_models()

input_text = st.text_area("Input English text:", "")
if st.button("Translate"):
    if input_text.strip():
        output = translator(input_text, num_beams=5, max_length=75)[0]['translation_text']
        st.success("Vietnamese Translation:")
        st.write(output)
    else:
        st.warning("Please enter text!")
