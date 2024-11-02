import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define cluster labels (assuming 4 clusters)
cluster_labels = {
    0: 'Budget-Conscious',
    1: 'Standard Spenders',
    2: 'Luxury Seekers',
    3: 'Occasional Shoppers'
}

# Form for user inputs
st.title("Customer Segmentation")

with st.form("customer_form"):
    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Annual Income", min_value=1000, max_value=1000000)
    spending_score = st.slider("Spending Score", 1, 100)
    submit_button = st.form_submit_button(label='Submit')

# Process the form data
if submit_button:
    # Scale input data as per training (you may adjust based on your notebook preprocessing)
    scaler = StandardScaler()
    user_data = scaler.fit_transform([[age, income, spending_score]])

    # Predict cluster and display label
    cluster = model.predict(user_data)[0]
    st.write(f"Cluster: {cluster_labels[cluster]}")
