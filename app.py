"""Interface for the uncertainty-aware vision-language model."""
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

import caption_image
import answer_question

st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload an image and set some options for demo purposes
st.header("Uncertainty-aware Vision language model")
st.write("Please upload an image to start the demo.")
img_file = st.sidebar.file_uploader(label='Upload an image', type=['png', 'jpg', 'jpeg'])

if img_file:
    img = Image.open(img_file)
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                             aspect_ratio=None)

    # Preview cropped image
    st.write("### Preview")
    _ = cropped_img.thumbnail((224, 224))
    st.image(cropped_img)
    if st.button("Start conversation"):
        with st.spinner('Waking up Eva...'):
            caption = caption_image.caption_api(cropped_img)
        st.write(f"Eva: Hi there! Isn't that {caption[0]}?")
        question = st.text_input("Ask Eva a question based on what she sees.")

    st.write("## Ask a question")
    question = st.text_input("Ask the model a question based on the thumbnail")
    if st.button("Ask a question"):
        with st.spinner('Asking Eva...'):
            answer = answer_question.answer_api(cropped_img, question)
        st.write(f"I think the answer is {answer}")
        # question = st.text_input("Ask Eva a question based on what she sees.")

        # if st.button('Ask'):
        #     st.write("Hello there.")
