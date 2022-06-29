"""Interface for the uncertainty-aware vision-language model."""
from requests import session
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

import caption_image
import answer_question

from streamlit_chat import message

st.set_option('deprecation.showfileUploaderEncoding', False)

st.set_page_config(
    page_title="u-VLM",
    page_icon=":robot:"
)

# Upload an image and set some options for demo purposes
st.header("Uncertainty-aware Vision language model")

# if 'cropped' not in st.session_state:
if "caption" not in st.session_state:
    st.write("Please upload an image to start the demo.")
    img_file = st.sidebar.file_uploader(label='Upload an image', type=['png', 'jpg', 'jpeg'])
    if img_file:
        st.session_state.image = img_file
        img = Image.open(st.session_state.image)
        cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                 aspect_ratio=None)
        _ = cropped_img.thumbnail((224, 224))
        st.image(cropped_img)
        caption_btn = st.button("Wake up Eva")
        if caption_btn:
            st.session_state.cropped = cropped_img
            with st.spinner('Waking up Eva...'):
                caption = caption_image.caption_api(st.session_state.cropped)
            st.session_state.caption = f"Hi there, I can see a {caption[0]}."
            message(st.session_state.caption)
            answer_btn = st.button("Ask Eva a question")
            if answer_btn:
                st.spinner("Loading QA interface.")

else:
    st.image(st.session_state.cropped)
    message(st.session_state.caption)
    question = st.text_input("Ask Eva a question based on what she sees.")
    if question:
        message(question, is_user=True)
        with st.spinner('Asking eva...'):
            answer = answer_question.answer_api(st.session_state.cropped, question)
        message(answer[0]["answer"])
