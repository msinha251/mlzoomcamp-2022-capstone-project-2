# build a streamlit app to predict the image

import streamlit as st
from train import predict_single
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# load the model
from tensorflow import keras
model = keras.models.load_model('./kitchenware-classification/models/model_kaggle.h5')

def load_image():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = predict(uploaded_file)
        st.write(f"This image is a {label}.")
    else:
        st.write("Please upload an image file")

def main():
    st.title("Image Classification")
    st.write("This is a simple image classification web app to predict the image.")
    load_image()


def predict(image):
    return predict_single(image_url=image, model=model, web=False)

if __name__ == '__main__':
    main()
