import streamlit as st
import joblib
import pandas as pd

# TODO: Load your saved model (hint: joblib.load())
# TODO: Load your saved feature columns

model = joblib.load('loan_approval_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Loan Approval Prediction App")
st.write("Enter details below to prediction the loan approval result")


income = st.number_input("Income", min_value=1, max_value=10000000, value=25000)
credit_score = st.number_input("Credit Score", min_value=1, max_value=900, value=650)
loan_amount = st.number_input("Loan Amount", min_value=1, max_value=200000, value=10000)
years_employed = st.number_input("Years Employed", min_value=1, max_value=100, value=5)
points = st.number_input("Points", min_value=1, max_value=200, value=50)

if st.button("Loan Approval Result"):
    # Create dataframe with EXACT column names from training
    input_data = pd.DataFrame({
    'income': [income],                    # ← lowercase variable
    'credit_score': [credit_score],        # ← lowercase variable
    'loan_amount': [loan_amount],          # ← lowercase variable
    'years_employed': [years_employed],    # ← lowercase variable
    'points': [points]     
    })

    
    # Scale the data (IMPORTANT!)
    input_scaled = scaler.transform(input_data)
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_value = prediction[0]

    # Display result
    if prediction_value == 1:
        st.success("Loan approval chances are high")
        st.write("Please consult a bank")
    else:
        st.error("Loan approval chances low")
        st.write("Please work on improving your credit")