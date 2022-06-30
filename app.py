"""Interface for the uncertainty-aware vision-language model."""
import streamlit as st
from PIL import Image
from streamlit_chat import message
import matplotlib.pyplot as plt
import seaborn as sns
import math

import caption_image
import answer_question
import robust_answers

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(
    page_title="u-VLM",
    page_icon="🤖"
)
st.header("Uncertainty-aware Vision language model")

if "image" not in st.session_state:
    st.write("Please upload an image to start the demo.")
    img_file = st.sidebar.file_uploader(label='Upload an image', type=['png', 'jpg', 'jpeg'])
    if img_file:
        img = Image.open(img_file)
        st.session_state.image = img

if "caption" not in st.session_state and "image" in st.session_state:
    with st.spinner('Loading model...'):
        caption = caption_image.caption_api(st.session_state.image)
        st.session_state.caption = f"I see {caption[0]}"
    st.image(st.session_state.image)
    message(st.session_state.caption, key="0")

if "question" not in st.session_state and "caption" in st.session_state and "image" in st.session_state:
    question = st.text_input("Ask the model a question based on what it sees.")
    if question:
        st.session_state.question = question

if "answer" not in st.session_state and "question" in st.session_state and "caption" in st.session_state and "image" in st.session_state:
    st.image(st.session_state.image)
    message(st.session_state.caption, key="1")
    message(st.session_state.question, is_user=True)
    with st.spinner('Asking the model...'):
        simple_answer = answer_question.answer_api(st.session_state.image, st.session_state.question)
        uncertainty, all_answers = robust_answers.answer_api(st.session_state.image, st.session_state.question)

    conf_threshold = 0.9
    conf = uncertainty.Probabilities[0]
    answer = uncertainty.index[0]
    if conf > conf_threshold:
        percentage_conf = math.floor(conf * 1e4) / 100
        message(f"I am {percentage_conf}% confident that the answer is {answer}. These are my top 5 predictions:")
    else:
        message("Sorry I am not quite sure. These are my top 5 predictions:")

    fig = plt.figure(figsize=(10, 10))
    sns.violinplot(data=all_answers, x="Predictions", y="Probabilities", scale="width")
    st.pyplot(fig)

    st.sidebar.write("Technical comparison:")
    baseline_conf = simple_answer[0]['probability'] / 100.
    robust_conf = math.floor(conf * 1e4) / 1e4
    delta = robust_conf - baseline_conf

    st.sidebar.metric(label="Baseline Confidence", value=baseline_conf)
    st.sidebar.metric(label="Robust Confidence", value=robust_conf, delta=delta)
