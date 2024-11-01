import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# model = joblib.load('model.pkl')

# Title of the app
st.title("Customer Segmentation Model")

# Input fields for user data
st.header("Input Customer Information")

# Gender input
gender = st.selectbox("Gender", options=["Male", "Female"])

# Age input
age = st.number_input("Age", min_value=18, max_value=100, value=30)

# Spending Score input
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Family Size input
family_size = st.number_input("Family Size", min_value=1, max_value=10, value=1)

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Spending Score': [spending_score],
    'Family Size': [family_size]
})

# Button to make predictions
if st.button("Predict Segment"):
    # Make predictions
    prediction = model.predict(input_data)
    
    # Display the result
    st.write(f"Predicted Customer Segment: {prediction[0]}")
