import time
import streamlit as st
import numpy as np
from predictor import Predictor
from dataset.add_noise import SynthesizeData

# Khai báo các biến toàn cục để lưu trữ trạng thái của ứng dụng
text_correct = ""
input_text = ""
noise_text = ""


import nltk

def main():
    model, synther = load_model()
    st.title("Chương trình sửa lỗi chính tả tiếng Việt")
    global input_text, noise_text, text_correct
    
    # Load model
    input_text = ""
    noise_text = ""
    text_input = st.text_area("Nhập đầu vào:", value=input_text)
    text_input = text_input.strip()
    
   

    if st.button("Correct"):
        noise_text = text_input
        text_correct = model.spelling_correct(noise_text)
        st.text("Câu nhiễu: ")
        st.success(noise_text)
        st.text("Kết quả:")
        st.success(text_correct)
    

    if st.button("Add noise and Correct"):
        noise_text = synther.add_noise(text_input, percent_err=0.3)
        text_correct = model.spelling_correct(noise_text) 
        st.text("Câu nhiễu: ")
        st.success(noise_text)
        st.text("Kết quả:")
        st.success(text_correct)
    

def mark_errors(text, errors):
    """
    Đánh dấu các từ bị lỗi trong văn bản với HTML để hiển thị màu đỏ.
    """
    for error in errors:
        text = text.replace(error, f"<span style='color:red'>{error}</span>")
    return text
        
@st.cache_resource
def load_model():
    print("Loading model ...")
    nltk.download('punkt')
    model = Predictor(weight_path='weights/seq2seq.pth', have_att=True)
    synther = SynthesizeData()
    
    return model, synther

model, synther = load_model()
# @st.cache(allow_output_mutation=True)  # hash_func
# def load_model():
#     print("Loading model ...")
#     nltk.download('punkt')
#     model = Predictor(weight_path='weights/seq2seq.pth', have_att=True)
#     synther = SynthesizeData()
    
#     return model, synther

if __name__ == "__main__":
    main()
