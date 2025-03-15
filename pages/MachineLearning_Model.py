import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

st.title('Wine Dataset with Model Predictions')
st.write("### Wine Dataset")
st.dataframe(df)

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_value = st.slider("Select k for KNN", min_value=1, max_value=20, value=5)
svm_C = st.slider("Select C for SVM", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
n_estimators = st.slider("Select n_estimators for Random Forest", min_value=10, max_value=200, value=100)

transformation = st.selectbox("Select Data Transformation", ["None", "Log Transformation", "Standardization"])

if transformation == "Log Transformation":
    X_train_scaled = np.log1p(X_train)
    X_test_scaled = np.log1p(X_test)
elif transformation == "Standardization":
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='linear', C=svm_C)
knn_model = KNeighborsClassifier(n_neighbors=k_value)
rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

if st.button('Randomize Input Values'):
    random_data = np.random.choice(df.drop(columns=['target']).values.flatten(), size=len(df.columns)-1)
    random_data = random_data.reshape(1, -1)

    st.write("### Randomized Input Values: ")
    st.write(random_data)

    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)

    knn_model.fit(X_train_scaled, y_train)
    knn_pred = knn_model.predict(X_test_scaled)

    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)

    species = ['Setosa', 'Versicolor', 'Virginica']
    svm_prediction = svm_model.predict(random_data)
    knn_prediction = knn_model.predict(random_data)
    rf_prediction = rf_model.predict(random_data)
    svm_species = species[svm_prediction[0]]
    knn_species = species[knn_prediction[0]]
    rf_species = species[rf_prediction[0]]

    st.write(f'### SVM Predicted Species: {svm_species}')
    st.write(f'### KNN Predicted Species: {knn_species}')
    st.write(f'### Random Forest Predicted Species: {rf_species}')

    species_count = {
        'Setosa': 0,
        'Versicolor': 0,
        'Virginica': 0
    }
    species_count[svm_species] += 1
    species_count[knn_species] += 1
    species_count[rf_species] += 1

    most_common_species = max(species_count, key=species_count.get)

    st.write(f"### Most Common Predicted Species: {most_common_species}")

    if most_common_species == 'Setosa':
        st.write(""" **Setosa**: A light and small wine that tends to be easily separable from others due to its unique characteristics. """)
    elif most_common_species == 'Versicolor':
        st.write(""" **Versicolor**: A medium-sized wine with characteristics that are a mix of Setosa and Virginica, often found in data that's not easily separable. """)
    elif most_common_species == 'Virginica':
        st.write(""" **Virginica**: The largest and heaviest wine in this group, often distinctively separable from other types based on its size and characteristics. """)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig_svm = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    random_data_pca = pca.transform(random_data)
    plt.scatter(random_data_pca[:, 0], random_data_pca[:, 1], color='red', marker='*', s=200, label='Randomized Input (SVM)')
    plt.colorbar(scatter)
    plt.title("PCA Projection of Wine Dataset (SVM)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend()

    fig_knn = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.scatter(random_data_pca[:, 0], random_data_pca[:, 1], color='blue', marker='*', s=200, label='Randomized Input (KNN)')
    plt.colorbar(scatter)
    plt.title("PCA Projection of Wine Dataset (KNN)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend()

    fig_rf = plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.scatter(random_data_pca[:, 0], random_data_pca[:, 1], color='green', marker='*', s=200, label='Randomized Input (Random Forest)')
    plt.colorbar(scatter)
    plt.title("PCA Projection of Wine Dataset (Random Forest)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.legend()

    st.pyplot(fig_svm)
    st.pyplot(fig_knn)
    st.pyplot(fig_rf)
