import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
log_reg = joblib.load('mpg_logistic_regression_model.pkl')
scaler = joblib.load('mpg_scaler.pkl')

# Streamlit app
st.title("MPG Classification App")

st.write("""
This app predicts whether a car has high or low MPG (Miles Per Gallon) based on the number of cylinders and acceleration.
""")

# User input for features
cylinders = st.number_input("Number of Cylinders", min_value=3, max_value=12, step=1)
acceleration = st.number_input("Acceleration", min_value=5.0, max_value=25.0, step=0.1)

# Prediction
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[cylinders, acceleration]])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the logistic regression model
    prediction = log_reg.predict(input_data_scaled)
    
    # Output the result
    result = "High MPG" if prediction[0] == 1 else "Low MPG"
    st.success(f"The car is predicted to have: **{result}**")
