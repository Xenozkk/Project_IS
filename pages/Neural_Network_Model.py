import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import cv2  # type: ignore

model = load_model('Predict_Num3.keras')

def load_data():
    df = pd.read_csv('mnist_dataset.csv')
    return df

def random_image_from_csv(df):
    random_index = np.random.randint(0, len(df))
    image_data = df.iloc[random_index, 1:].values.astype('float32')
    image_data = image_data.reshape(28, 28)
    label = df.iloc[random_index, 0]
    return image_data, label

def predict_number(image_data):
    image_data = cv2.resize(image_data, (32, 32))  
    image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB) 
    image_data = image_data.reshape(1, 32, 32, 3)
    image_data = image_data / 255.0 
    prediction = model.predict(image_data)
    return np.argmax(prediction)

if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = False  

st.title("MNIST Digit Prediction")

if st.button("สุ่มรูปภาพจาก CSV", disabled=st.session_state.button_disabled):
    st.session_state.button_disabled = True
    
    df = load_data()
    image_data, label = random_image_from_csv(df)

    image_data = (image_data * 255).astype(np.uint8)
    st.image(image_data, width=200)

    prediction = predict_number(image_data)

    st.write(f"โมเดลทำนาย: {prediction}")

    st.session_state.button_disabled = False
