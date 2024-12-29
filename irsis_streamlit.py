import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris
# Load the iris dataset
iris = load_iris()

# Load the trained model
model = joblib.load('iris_model.pkl')

# App title
st.title("Iris Flower Prediction")

# Input features
sepal_length = st.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.slider("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.slider("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Predict button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    prediction_label = iris.target_names[prediction]
    
    # Display the prediction
    st.success(f"The predicted Iris flower type is: {prediction_label}")
