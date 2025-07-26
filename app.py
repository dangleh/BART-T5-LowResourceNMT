import streamlit as st
import time
import logging
from src.model import get_translator
from src import config

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)

def show_header():
    st.title("English → Vietnamese Translation (mBART50 Finetuned)")
    st.markdown(
        """
        **Hướng dẫn sử dụng:**  
        - Nhập câu tiếng Anh vào ô bên dưới.  
        - Nhấn nút **Translate** để nhận bản dịch tiếng Việt.  
        - Kết quả và thời gian dịch sẽ hiển thị phía dưới.
        """
    )

@st.cache_resource
def load_models():
    try:
        translator = get_translator()
        logging.info("Model loaded successfully.")
        return translator
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error("Failed to load translation model. Please check server logs.")
        return None

def validate_input(text):
    if not text.strip():
        return False, "Please enter text!"
    if len(text) > config.INPUT_MAX_CHARS:
        return False, f"Input too long (>{config.INPUT_MAX_CHARS} chars). Please shorten your text."
    if any(ord(c) < 32 and c not in '\n\t' for c in text):
        return False, "Input contains invalid control characters."
    return True, ""

def main():
    show_header()
    translator = load_models()
    input_text = st.text_area("Input English text:", config.DEFAULT_INPUT_EXAMPLE, max_chars=config.INPUT_MAX_CHARS)
    if st.button("Translate"):
        valid, msg = validate_input(input_text)
        if not valid:
            st.warning(msg)
            logging.warning(f"Input validation failed: {msg}")
            return
        if translator is None:
            st.error("Model not available.")
            return
        try:
            start_time = time.time()
            output = translator(
                input_text,
                num_beams=config.NUM_BEAMS,
                max_length=config.MAX_LENGTH
            )[0]['translation_text']
            elapsed = time.time() - start_time
            st.success("Vietnamese Translation:")
            st.write(output)
            st.info(f"⏱️ Translation time: {elapsed:.2f} seconds")
            logging.info(f"Translation success. Time: {elapsed:.2f}s. Input: {input_text}")
        except Exception as e:
            st.error("Translation failed. Please try again or check logs.")
            logging.error(f"Translation error: {e}")

if __name__ == '__main__':
    main()