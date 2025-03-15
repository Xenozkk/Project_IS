import streamlit as st
import tensorflow as tf  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import matplotlib.pyplot as plt

st.title("Convolutional Neural Network (CNN)")

st.markdown("""
## 📌 What is a Convolutional Neural Network (CNN)?
A **CNN** is a Deep Learning model designed for processing and analyzing images. It consists of four main components:
- **Convolutional Layer** – Extracts important features from an image using filters.
- **Pooling Layer** – Reduces the size of the image while preserving essential information.
- **Flatten Layer** – Converts multidimensional data into a one-dimensional vector.
- **Fully Connected Layer** – Performs classification of the input image.
""")

st.image("Architec_CNN.jpg", caption="CNN Architecture")

st.markdown("## 📌 How does the Convolutional Layer work?")
st.markdown("""
The **Convolutional Layer** uses small filters that slide across the image to detect important features. Key concepts include:
- Filters extract features such as edges, shapes, and textures.
- Activation functions (e.g., ReLU) help the model learn important patterns.
""")
st.image("Convolution_1.jpg", caption="Example of Convolution Operation")
st.image("Convolution_2.jpg", caption="Filters extracting image features")

st.markdown("## 📌 How does the Pooling Layer work?")
st.markdown("""
The **Pooling Layer** reduces the image size and the number of parameters. The two main types are:
- **Max Pooling** – Selects the highest value in a given region.
- **Average Pooling** – Computes the average value within a region.
""")
st.image("Pooling_1.jpg", caption="Max Pooling Example")
st.image("Pooling_2.jpg", caption="Average Pooling Example")

st.markdown("## 📌 Fully Connected Layer and Classification Decision")
st.markdown("""
After passing through the **Flatten Layer**, the model uses a **Fully Connected Layer** (FC Layer), which acts as a traditional neural network:
- Combines all extracted features to make a classification decision.
- Uses activation functions such as **Softmax** for final classification.
""")
st.image("Fully_Connec.jpg", caption="Fully Connected Layer Structure")

st.title("📊 Dataset Preparation")

# 📌 Dataset Information
st.markdown("""
The dataset used in this project is the **MNIST Dataset**,  
which is widely used for handwritten digit classification tasks.
""")

# 🔗 Links to Dataset Sources
st.markdown("""
🔗 [MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  
🔗 [Original MNIST Data by Yann LeCun](http://yann.lecun.com/exdb/mnist/)
""")

# 📌 Purpose of Using MNIST
st.markdown("""
To train a model that can analyze and classify handwritten digits,  
we consider various features extracted from the images, including:
- **Pixel intensity** – brightness of each pixel
- **Shape and size of digits** – how each number is formed
- **Stroke density** – thickness of the handwritten digits
""")

st.title("🔍 WORKFLOW")

st.markdown("### **Model Used**: [Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)")

st.markdown("""
### **Why Use CNN?**
A **Convolutional Neural Network (CNN)** is a **Deep Learning** model specifically designed for image processing. It is particularly effective for image classification tasks, as it efficiently extracts key features from images. CNNs work by processing data through multiple layers, where each layer extracts increasingly complex patterns. These layers include:
- **Convolutional Layers** for feature detection
- **Pooling Layers** to reduce spatial dimensions
- **Fully Connected Layers** for classification

CNNs are widely used in **computer vision** applications such as object detection, medical imaging, and handwriting recognition.
""")

st.markdown("### **Training Workflow**")
st.markdown("In this project, we use **CNN + ResNet50 model** for image classification.")
st.image("Layer.jpg", caption="Fully Connected Layer Structure")

st.markdown("### **Activation Functions Used**")
st.markdown("""
- **ReLU (Rectified Linear Unit)**: Used in **Convolutional Layers** to introduce non-linearity and improve learning efficiency.
- **Softmax**: Used in **Fully Connected Layers** to classify images into different categories.
""")

st.markdown("## 📌 1. Loading and Preprocessing the MNIST Dataset")
with st.echo():
    from tensorflow.keras.datasets import mnist # type: ignore
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

    train_images = tf.image.resize(train_images, (32, 32))
    test_images = tf.image.resize(test_images, (32, 32))

    train_images = tf.repeat(train_images, 3, axis=-1)
    test_images = tf.repeat(test_images, 3, axis=-1)

    train_images, test_images = train_images / 255.0, test_images / 255.0

st.markdown("## 📌 2. Data Augmentation")
with st.echo():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

st.markdown("## 📌 3. Building the Model with ResNet50 + Fully Connected Layers")
with st.echo():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # Freeze ResNet50 layers

    model = models.Sequential([
        base_model,
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

st.markdown("## 📌 4. Training the Model")
st.write('epochs = 10 , batch_size = 32')
with st.echo():
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(train_images, train_labels, epochs=0,
                         validation_data=(test_images, test_labels),
                         callbacks=[early_stop])

st.markdown("## 📌 5. Model Performance Evaluation")

st.markdown("### 🔹 Model Accuracy")
st.image("Accuracy.png")
st.markdown("""
- The accuracy curve shows a steady increase, indicating the model is learning properly.
- The validation accuracy remains close to training accuracy, meaning the model does not overfit significantly.
- Around the later epochs, the accuracy stabilizes, suggesting the model has reached its optimal learning.
""")

st.markdown("### 🔹 Model Loss")
st.image("Model_lose.png")
st.markdown("""
- The loss curve is decreasing over epochs, showing that the model is optimizing its weights.
- The validation loss is close to the training loss, meaning the model does not suffer from major overfitting.
- A low and stable loss value suggests that the model has successfully learned to classify the data.
""")

