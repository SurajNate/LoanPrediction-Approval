import streamlit as st
from PIL import Image
import pickle
import numpy as np

st.set_page_config(layout="wide")

# Load the trained machine learning model
model = pickle.load(open('./Model/ML_Model1.pkl', 'rb'))

# Function to run the Streamlit App
def run():
    # Display Bank Logo
    img1 = Image.open('Laxmi Chit Fund.png')
    st.image(img1, use_container_width=True)  # Fixed the deprecation warning

    # App Title
    st.title("Bank Loan Prediction using Machine Learning")

    # User Inputs
    account_no = st.text_input('Account Number')
    full_name = st.text_input('Full Name')

    # Dropdown Selections
    gender = st.selectbox("Gender", ['Female', 'Male'])
    marital_status = st.selectbox("Marital Status", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['No', 'One', 'Two', 'More than Two'])
    education = st.selectbox("Education", ['Not Graduate', 'Graduate'])
    employment_status = st.selectbox("Employment Status", ['Job', 'Business'])
    property_area = st.selectbox("Property Area", ['Rural', 'Semi-Urban', 'Urban'])
    credit_score = st.selectbox("Credit Score", ['Between 300 to 500', 'Above 500'])

    # Numeric Inputs
    monthly_income = st.number_input("Applicant's Monthly Income ($)", min_value=0, value=0, step=500)
    co_monthly_income = st.number_input("Co-Applicant's Monthly Income ($)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=0, step=1000)

    # Loan Duration Mapping
    loan_duration_mapping = {'2 Month': 60, '6 Month': 180, '8 Month': 240, '1 Year': 360, '16 Month': 480}
    loan_duration = st.selectbox("Loan Duration", list(loan_duration_mapping.keys()))

    # Submit Button
    if st.button("Submit"):
        # Convert categorical values to numerical for the model
        feature_dict = {
            'gender': {'Female': 0, 'Male': 1},
            'marital_status': {'No': 0, 'Yes': 1},
            'dependents': {'No': 0, 'One': 1, 'Two': 2, 'More than Two': 3},
            'education': {'Not Graduate': 0, 'Graduate': 1},
            'employment_status': {'Job': 0, 'Business': 1},
            'property_area': {'Rural': 0, 'Semi-Urban': 1, 'Urban': 2},
            'credit_score': {'Between 300 to 500': 0, 'Above 500': 1},
        }

        # Total Income Calculation
        total_income = monthly_income + co_monthly_income

        # Loan Approval Conditions Before Model Prediction
        if total_income < 2000:
            st.error(f"Hello {full_name} || Account Number: {account_no} || Loan Denied: Income too low.")
            return

        if loan_amount > (total_income * 10):
            st.error(f"Hello {full_name} || Account Number: {account_no} || Loan Denied: Income is Incompatable.")
            return

        if feature_dict['credit_score'][credit_score] == 0 and loan_amount > 5000:
            st.error(f"Hello {full_name} || Account Number: {account_no} || Loan Denied: Poor credit score with high loan amount.")
            return

        # Prepare input features for model prediction
        features = np.array([[
            feature_dict['gender'][gender],
            feature_dict['marital_status'][marital_status],
            feature_dict['dependents'][dependents],
            feature_dict['education'][education],
            feature_dict['employment_status'][employment_status],
            monthly_income,
            co_monthly_income,
            loan_amount,
            loan_duration_mapping[loan_duration],
            feature_dict['credit_score'][credit_score],
            feature_dict['property_area'][property_area]
        ]])

        # Model Prediction
        prediction = model.predict(features)[0]  # Get the first prediction (0 or 1)

        # Display Results
        if prediction == 0:
            st.error(f"Hello {full_name} || Account Number: {account_no} || Loan Denied.")
        else:
            st.success(f"Hello {full_name} || Account Number: {account_no} || Congratulations! Loan Approved.")

# Run the Streamlit app
run()


# Footer
st.write("---")
st.markdown('<center><a href="https://www.instagram.com/suraj_nate/" target="_blank" style="color:white;text-decoration:none">&copy; 2025 @suraj_nate All rights reserved.</a></center>', unsafe_allow_html=True)
