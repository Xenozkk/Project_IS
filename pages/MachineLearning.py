import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

st.title('Wine Dataset with Model Predictions')
st.write("### Wine Dataset")
st.dataframe(df)

st.header('K-Nearest Neighbors (KNN)')

st.write("""
**K-Nearest Neighbors (KNN)** is a non-parametric supervised learning algorithm. 
It is a type of lazy learning, where the training data is stored and used directly for prediction.

- **Training Phase**: In KNN, the algorithm stores the entire labeled training dataset as a reference. 
- **Prediction Phase**: When making a prediction, the KNN algorithm calculates the distance between the input data point and all the training examples. 
- **Classification**: The input data point is classified based on the majority class of its nearest neighbors.
- **K**: The number of neighbors to consider is a key hyperparameter for the model.

### How KNN works:
- An object is classified by a majority vote of its neighbors, where the object is assigned to the class most common among its k nearest neighbors.
- If **k = 1**, the object is assigned to the class of the single nearest neighbor.

### Distance Metric:
KNN commonly uses **Euclidean distance** for calculating the distance between points.
""")
st.image('Distance_KNN.png' , caption="Distance_KNN")

st.write("### Example")
st.write("If we have weather data like temperature and humidity, we can use KNN to predict whether tomorrow will be a sunny day or a rainy day based on historical data.")

st.image('Weather_KNN.png', caption="Weather_KNN")

st.header('Support Vector Machine (SVM)')

st.write("""
**Support Vector Machine (SVM)** is a supervised learning algorithm that can be used for both classification and regression tasks. 
Its main goal is to find a hyperplane in an N-dimensional space that distinctly classifies the data points.

- **Hyperplane**: The hyperplane is the boundary that best separates the classes in the data.
- **Margin**: SVM aims to maximize the margin between data points of different classes. The distance from the hyperplane to the nearest data points is called the **margin**, and the points closest to the hyperplane are known as **support vectors**.
- **Objective**: The SVM algorithm tries to find the maximum margin separator.

### When the data is not linearly separable:
If the data points are not linearly separable, SVM uses a **kernel trick** to map the data into higher dimensions to make it separable. 
This allows SVM to handle non-linear data by transforming the feature space.

### Soft Margin Classification:
SVM allows some misclassification when the data is noisy or difficult to separate by introducing a regularization parameter **C**, which controls how much misclassification is allowed.
""")

st.image('Soft_margin_SVM.png', caption="SVM Image will be inserted here")


st.header("Random Forest Theory")

st.write("""
**Random Forest** is an **ensemble learning method** used for classification or regression tasks by combining the results from multiple decision trees to improve accuracy and stability.

- **How it works**: 
  - **Random Forest** creates **multiple decision trees** where each tree is trained on a random subset of the data and features.
  - The final prediction is made by **voting** (for classification tasks) or averaging the results (for regression tasks) from all the decision trees.

It is highly effective for handling complex data and can handle both high-dimensional and non-linear data efficiently.
""")

st.header("Data Preparation")

st.write("""
The following steps outline how we prepare the Wine dataset before training the model:
1. **Splitting the data into features (X) and target (y)**
2. **Splitting the data into training and testing sets**
3. **Scaling the features for better model performance**
""")

st.write("#### Step 1: Splitting Data into Features and Target")
st.write("""
In this step, we separate the data into **features (X)**, which are the chemical properties of the wines, and **target (y)**, which is the classification of the wine type (Class 0, Class 1, or Class 2).
""")
with st.echo():
    X = df.drop(columns=['target'])  
    y = df['target']  

st.write(f"Features (X): {X.columns.tolist()}")
st.write(f"Target (y): {y.name}")

st.write("#### Step 2: Splitting Data into Training and Test Sets")
st.write("""
Next, we divide the data into two parts:
- **Training set (80%)**: Used for training the machine learning models.
- **Test set (20%)**: Used for evaluating the performance of the models.

We typically use a ratio of 80% for training and 20% for testing.
""")
with st.echo():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"Training set size: {X_train.shape[0]} samples")
st.write(f"Test set size: {X_test.shape[0]} samples")

st.write("#### Step 3: Feature Scaling")
st.write("""
Feature scaling is an important step for many machine learning algorithms, especially those that rely on distance calculations like KNN and SVM. 
We will use **StandardScaler** to scale the features, transforming them to have zero mean and unit variance. This helps ensure that no feature dominates others due to differences in scale.
""")
with st.echo():
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

st.write("Data scaling complete. The training and test data are now ready for model training.")

st.write("""
### Model Training
In this section, we will train the following machine learning models:
1. **Support Vector Machine (SVM)**
2. **K-Nearest Neighbors (KNN)**
3. **Random Forest**

Each model will be trained using the training set, and we will evaluate the performance using the test set.
""")

st.write("#### Step 1: Support Vector Machine (SVM)")
with st.echo():
    from sklearn.svm import SVC
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train_scaled, y_train)  

st.write("#### Step 2: K-Nearest Neighbors (KNN)")
with st.echo():
    from sklearn.neighbors import KNeighborsClassifier
    k_value = st.slider("Select k for KNN", min_value=1, max_value=20, value=5) 
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X_train_scaled, y_train) 

st.write("#### Step 3: Random Forest")

with st.echo():
    from sklearn.ensemble import RandomForestClassifier
    n_estimators = st.slider("Select n_estimators for Random Forest", min_value=10, max_value=200, value=100)
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

st.write("""
### Data Source:
- The **Wine Dataset** comes from the **UCI Machine Learning Repository** (University of California, Irvine).
- It was originally collected for the purpose of studying classification algorithms and is widely used in machine learning research.

You can access the dataset here: [UCI Wine Dataset](https://archive.ics.uci.edu/ml/datasets/wine)
""")