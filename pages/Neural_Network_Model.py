import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image
import cv2  # type: ignore

# โหลดโมเดล Keras
model = load_model('Predict_Num3.keras')

# ฟังก์ชันโหลดข้อมูลจาก CSV
def load_data():
    df = pd.read_csv('mnist_test.csv')
    return df

# ฟังก์ชันสุ่มรูปภาพจาก CSV
def random_image_from_csv(df):
    random_index = np.random.randint(0, len(df))
    image_data = df.iloc[random_index, 1:].values.astype('float32')
    image_data = image_data.reshape(28, 28)
    label = df.iloc[random_index, 0]
    return image_data, label

# ฟังก์ชันทำนายตัวเลข
def predict_number(image_data):
    # ปรับขนาดให้ตรงกับที่โมเดลต้องการ
    image_data = cv2.resize(image_data, (32, 32))  # ปรับเป็น 32x32
    image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)  # แปลงเป็น RGB
    image_data = image_data.reshape(1, 32, 32, 3)  # เปลี่ยนรูปร่างเป็น (1, 32, 32, 3)
    image_data = image_data / 255.0  # ปรับค่าให้อยู่ในช่วง [0, 1]
    prediction = model.predict(image_data)
    return np.argmax(prediction)

# กำหนดค่าเริ่มต้นของปุ่มใน session_state
if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = False  # ปุ่มสามารถกดได้ในตอนแรก

# ส่วนของ Streamlit
st.title("MNIST Digit Prediction")

# แสดงปุ่มสุ่มภาพ และปิดการใช้งานปุ่มถ้ายังไม่ได้ทำนายเสร็จ
if st.button("สุ่มรูปภาพจาก CSV", disabled=st.session_state.button_disabled):
    # ปิดปุ่มชั่วคราวจนกว่าการทำนายจะเสร็จ
    st.session_state.button_disabled = True
    
    # โหลดข้อมูลและสุ่มรูปภาพ
    df = load_data()
    image_data, label = random_image_from_csv(df)

    # แปลงค่าให้เหมาะสมสำหรับการแสดงผล
    image_data = (image_data * 255).astype(np.uint8)  # แปลงกลับเป็นภาพ
    st.image(image_data, width=200)

    # ทำนายผล
    prediction = predict_number(image_data)

    # แสดงผลลัพธ์การทำนาย
    st.write(f"โมเดลทำนาย: {prediction}")

    # เปิดให้กดปุ่มใหม่ได้หลังจากการทำนายเสร็จ
    st.session_state.button_disabled = False
