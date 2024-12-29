import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load('penguins_logistic_regression_model.pkl')

# Streamlit app title
st.title("Penguin Species Predictor")

# Input form for user data
st.header("Input Penguin Measurements:")
bill_length = st.number_input("Bill Length (mm)", min_value=30.0, max_value=70.0, step=0.1)
bill_depth = st.number_input("Bill Depth (mm)", min_value=13.0, max_value=21.0, step=0.1)
flipper_length = st.number_input("Flipper Length (mm)", min_value=160.0, max_value=250.0, step=1.0)

# Predict button
if st.button("Predict"):
    # Prepare the input data as a Pandas DataFrame
    input_data = pd.DataFrame(
        [[bill_length, bill_depth, flipper_length]],
        columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']
    )
    
    # Make a prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"The predicted species is: **{prediction[0]}**")
    
    st.subheader("Prediction Probabilities:")
    for i, prob in enumerate(prediction_proba[0]):
        st.write(f"Class {model.named_steps['classifier'].classes_[i]}: {prob:.2%}")
