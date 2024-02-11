import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model_loan.pkl", 'rb'))

# Function to predict loan eligibility
def predict_loan_eligibility(details):
    details = pd.DataFrame(details, index=[0])  # Convert details to DataFrame
    prediction = model.predict(details)[0]
    probability = model.predict_proba(details)[0][1]
    return prediction, probability

# Encoding categorical features function
def encode_categorical(data):
    # Mapping dictionaries for encoding
    gender_map = {'Male': 1, 'Female': 0}
    married_map = {'Yes': 1, 'No': 0}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Urban': 0, 'Semiurban': 1 ,'Rural': 2}
   
    # Encoding categorical features
    data['Gender'] = data['Gender'].map(gender_map)
    data['Married'] = data['Married'].map(married_map)
    data['Education'] = data['Education'].map(education_map)
    data['Self_Employed'] = data['Self_Employed'].map(self_employed_map)
    data['Property_Area'] = data['Property_Area']. map(property_area_map)
    return data

# Main function to run the Streamlit web app
def main():
    st.title('Loan Eligibility Prediction')

    # Name input field
    name = st.text_input('Enter Your Name')

    # Sidebar with user input fields
    st.sidebar.header('Enter Customer Details')
    gender = st.sidebar.radio('Gender', ['Male', 'Female'])
    married = st.sidebar.radio('Marital Status', ['Yes', 'No'])
    dependents = st.sidebar.slider('Number of Dependents', min_value=0, max_value=10, step=1)
    education = st.sidebar.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.sidebar.radio('Self Employed', ['Yes', 'No'])
    applicant_income = st.sidebar.number_input('Applicant Income', value=0)
    loan_amount = st.sidebar.number_input('Loan Amount', value=0)
    loan_term = st.sidebar.number_input('Loan Term (in months)', value=0)
    credit_history = st.sidebar.radio('Credit History', [0, 1])
    area = st.sidebar.radio('Property Settlement', ['Urban', 'Semiurban' ,'Rural'])

    # Logarithm of Applicant Income
    applicant_income_log = np.log(applicant_income + 1)  # Adding 1 to handle 0 values
    loan_amount_log = np.log(loan_amount + 1)
    loan_term_log = np.log(loan_term + 1)


    details = {
        #'Name': name,
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'Credit_History': credit_history,
        'Property_Area' : area,
        'ApplicantIncome': applicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'ApplicantIncome_log': applicant_income_log,
        'LoanAmount_log' : loan_amount_log,
        'Loan_Amount_Term-log' : loan_term_log
      
    }
    

    # Encoding categorical features
    details_encoded = encode_categorical(pd.DataFrame([details]))

    if st.sidebar.button('Predict'):
        prediction, probability = predict_loan_eligibility(details_encoded)
        if prediction == 1:
            st.success(f'Hi {name}, you are eligible for a loan with a probability of {probability:.2f}')
        else:
            st.error(f'Hi {name}, you are not eligible for a loan')

if __name__ == '__main__':
    main()
