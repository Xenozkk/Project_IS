import streamlit as st

st.title("Project Intelligent System")
st.write("  ")
st.write("## 📌Machine Learning & Neural Networks")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Machine Learning")

    if st.button("Machine learning explanation"):
        st.switch_page("pages/MachineLearning.py")

    if st.button("Machine learning Model"):
        st.switch_page("pages/MachineLearning_Model.py")

with col2:
    st.subheader("Neural Networks")
    
    if st.button("Neural network explaination"):
        st.switch_page("pages/Neural_Network.py")

    if st.button("Neural network Model"):
        st.switch_page("pages/Neural_Network_Model.py")
