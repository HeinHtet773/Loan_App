import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
def load_model():
    with open('loan_predict_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to accept user data
def user_input_features():
    employed = st.selectbox('Employed', ['Yes', 'No'])
    bank_balance = st.number_input('Bank Balance', min_value=0, value=0)
    annual_salary = st.number_input('Annual Salary', min_value=0, value=0)


    data = {
        'Employed': employed,
        'Bank Balance': bank_balance,
        'Annual Salary': annual_salary,
    }

    features = pd.DataFrame(data, index=[0]) 
    return features

# Function to transform categorical features to numeric
def transform_features(features):
    print("Before Transformation:")
    print(features)

    features['Employed'] = features['Employed'].map({'Yes': 1, 'No': 0}) #This line can be omitted because 'Employed' is already numerical.

    print("After Transformation:")
    print(features)

    return features

# Main function to run the Streamlit app
def main():
    st.title('Loan Prediction Web App')
    st.write("Please enter the following information to predict loan status")

    model = load_model()
    user_input = user_input_features()
    transformed_input = transform_features(user_input)
    
    if st.button('Predict'):
        prediction = model.predict(transformed_input)
        st.subheader('Prediction Result')
        if prediction[0] == 1:
            st.write("The model predicts that the loan will be approved.")
        else:
            st.write("The model predicts that the loan will not be approved.")

if __name__ == '__main__':
    main()