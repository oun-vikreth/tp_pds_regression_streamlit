import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# Load models and datasets
iris_model = joblib.load('iris_model.pkl')
iris_data = load_iris()

mpg_model = joblib.load('mpg_logistic_regression_model.pkl')
mpg_scaler = joblib.load('mpg_scaler.pkl')

penguin_model = joblib.load('penguins_logistic_regression_model.pkl')

# Main app
st.title("Multi-Purpose Prediction App")
st.sidebar.header("Select Prediction Functionality")

# Sidebar menu
menu = st.sidebar.radio(
    "Choose a Prediction Tool:",
    ("Iris Flower Prediction", "MPG Classification", "Penguin Species Prediction")
)

if menu == "Iris Flower Prediction":
    st.header("Iris Flower Prediction")
    
    # Input sliders for Iris flower features
    sepal_length = st.slider("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_length = st.slider("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

    if st.button("Predict Iris Flower"):
        # Prepare input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = iris_model.predict(input_data)[0]
        prediction_label = iris_data.target_names[prediction]
        st.success(f"The predicted Iris flower type is: **{prediction_label}**")

elif menu == "MPG Classification":
    st.header("MPG Classification")
    st.write("This tool predicts whether a car has High or Low MPG (Miles Per Gallon).")

    # Input fields for MPG prediction
    cylinders = st.number_input("Number of Cylinders", min_value=3, max_value=12, step=1)
    acceleration = st.number_input("Acceleration", min_value=5.0, max_value=25.0, step=0.1)

    if st.button("Predict MPG"):
        # Prepare input data
        input_data = np.array([[cylinders, acceleration]])
        input_data_scaled = mpg_scaler.transform(input_data)
        prediction = mpg_model.predict(input_data_scaled)
        result = "High MPG" if prediction[0] == 1 else "Low MPG"
        st.success(f"The car is predicted to have: **{result}**")

elif menu == "Penguin Species Prediction":
    st.header("Penguin Species Prediction")
    
    # Input fields for Penguin prediction
    bill_length = st.number_input("Bill Length (mm)", min_value=30.0, max_value=70.0, step=0.1)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=13.0, max_value=21.0, step=0.1)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=160.0, max_value=250.0, step=1.0)

    if st.button("Predict Penguin Species"):
        # Prepare input data as a DataFrame
        input_data = pd.DataFrame(
            [[bill_length, bill_depth, flipper_length]],
            columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
        )
        prediction = penguin_model.predict(input_data)
        prediction_proba = penguin_model.predict_proba(input_data)

        st.subheader("Prediction:")
        st.write(f"The predicted species is: **{prediction[0]}**")

        st.subheader("Prediction Probabilities:")
        for i, prob in enumerate(prediction_proba[0]):
            st.write(f"Class {penguin_model.classes_[i]}: {prob:.2%}")
