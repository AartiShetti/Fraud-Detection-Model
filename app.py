import streamlit as st
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('fraud_detection_model.pkl')

# Streamlit app title
st.title("Fraud Detection System")

# Create input fields for the user to input transaction data (adjust based on your dataset features)
amount = st.number_input("Transaction Amount", min_value=0.0)
time = st.number_input("Transaction Time", min_value=0)

# Create a dictionary with user input data
user_input = {
    "Amount": amount,
    "Time": time,
 
}

# Convert the dictionary to a pandas DataFrame (make sure the columns match your model's features)
user_df = pd.DataFrame([user_input])

# When the user clicks the "Predict" button, make a prediction using the model
if st.button("Predict"):
    prediction = model.predict(user_df)
    
    if prediction == 1:
        st.write("This is a Fraudulent Transaction!")
    else:
        st.write("This is a Non-Fraudulent Transaction.")
