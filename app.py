import streamlit as st
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.title("English → Vietnamese Translation (mBART50 Finetuned)")
st.markdown(
    """
    **Hướng dẫn sử dụng:**  
    - Nhập câu tiếng Anh vào ô bên dưới.  
    - Nhấn nút **Translate** để nhận bản dịch tiếng Việt.  
    - Kết quả và thời gian dịch sẽ hiển thị phía dưới.
    """
)

MODEL = "dangleh/mbart50_envi_finetuned"

@st.cache_resource
def load_models():
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="en_XX", tgt_lang="vi_VN")
    return translator

def main():
    translator = load_models()
    input_text = st.text_area("Input English text:", "The today weather is hot!")
    if st.button("Translate"):
        if input_text.strip():
            start_time = time.time()
            output = translator(input_text, num_beams=5, max_length=75)[0]['translation_text']
            elapsed = time.time() - start_time
            st.success("Vietnamese Translation:")
            st.write(output)
            st.info(f"⏱️ Translation time: {elapsed:.2f} seconds")
        else:
            st.warning("Please enter text!")

if __name__ == '__main__':
     main()