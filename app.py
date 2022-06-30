"""Interface for the uncertainty-aware vision-language model."""
import streamlit as st
from PIL import Image
from streamlit_chat import message

import caption_image
import answer_question

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title="u-VLM",
    page_icon=":robot:"
)
st.header("Uncertainty-aware Vision language model")

if "image" not in st.session_state:
    st.write("Please upload an image to start the demo.")
    img_file = st.sidebar.file_uploader(label='Upload an image', type=['png', 'jpg', 'jpeg'])
    if img_file:
        img = Image.open(img_file)
        st.session_state.image = img

if "caption" not in st.session_state and "image" in st.session_state:
    with st.spinner('Waking up Eva...'):
        caption = caption_image.caption_api(st.session_state.image)
        st.session_state.caption = caption
    st.image(st.session_state.image)
    message(st.session_state.caption, key="0")

if "question" not in st.session_state and "caption" in st.session_state and "image" in st.session_state:
    question = st.text_input("Ask Eva a question based on what she sees.")
    if question:
        st.session_state.question = question

if "answer" not in st.session_state and "question" in st.session_state and "caption" in st.session_state and "image" in st.session_state:
    st.image(st.session_state.image)
    message(st.session_state.caption, key="1")
    message(st.session_state.question, is_user=True)
    with st.spinner('Asking eva...'):
        answer = answer_question.answer_api(st.session_state.image, st.session_state.question)
    message(answer[0]["answer"])
