import streamlit as st

st.title("Project Intelligent System")
st.write("  ")
st.write("## 📌Machine Learning & Neural Networks")

# แบ่งหน้าออกเป็น 2 คอลัมน์
col1, col2 = st.columns(2)

# 📌 คอลัมน์ Machine Learning
with col1:
    st.subheader("Machine Learning")
    
    # ปุ่มเปลี่ยนไปหน้า Machine Learning Explanation
    if st.button("Machine learning explanation"):
        st.switch_page("pages/MachineLearning.py")

    # ปุ่มเปลี่ยนไปหน้า Machine Learning Model
    if st.button("Machine learning Model"):
        st.switch_page("pages/MachineLearning_Model.py")

# 📌 คอลัมน์ Neural Networks
with col2:
    st.subheader("Neural Networks")
    
    # ปุ่มเปลี่ยนไปหน้า Neural Network Explanation
    if st.button("Neural network explaination"):
        st.switch_page("pages/Neural_Network.py")

    # ปุ่มเปลี่ยนไปหน้า Neural Network Model
    if st.button("Neural network Model"):
        st.switch_page("pages/Neural_Network_Model.py")
