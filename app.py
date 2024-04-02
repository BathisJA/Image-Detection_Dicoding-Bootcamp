import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Function to check the prediction result
def check_result(result):
    if result == 0:
        return 'paper'
    elif result == 1:
        return 'rock'
    elif result == 2:
        return 'scissors'

# Load the saved model
model_path = "model.h5"
model = load_model(model_path)

# Set up Streamlit UI
st.set_page_config(
    page_title="Rock Paper Scissors Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define background color and CSS
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(45deg, #7D3CF8, #7CBAFF);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h1 style='text-align: center;'>Rock Paper Scissors Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This app predicts whether the uploaded image contains rock, paper, or scissors.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Define columns
col1, col2 = st.columns([3, 1])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    col1.image(img, caption='Uploaded Image.', width=500)

    # Predict button
    with col2:
        st.write("   ") 
        if st.button("Predict", key="predict_button"):
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])
            predictions = model.predict(images)
            predicted_class_index = np.argmax(predictions[0])
            prediction_label = check_result(predicted_class_index)
            st.title(f"{prediction_label}")