# app.py

import streamlit as st
from backend import predict_diabetes

# --- Page Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="💉",
    layout="centered"
)

# --- Title and Description ---
st.title("Diabetes Prediction App")
st.write("Enter the patient details below to predict the likelihood of diabetes.")

# --- Sidebar (Optional) ---
st.sidebar.header("About")
st.sidebar.info("DiaXAi is an advanced, AI-powered application designed to predict the likelihood of diabetes in patients using the Pima Indians Diabetes dataset. Built using state-of-the-art deep learning techniques with TensorFlow and Keras, DiaXAi not only provides accurate predictions but also emphasizes transparency and interpretability through explainable AI (XAI) methods.")

# --- Patient Input Section ---
st.header("Patient Information")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0, format="%.2f")
age = st.number_input("Age", min_value=0, max_value=120, value=0)

# --- Prediction ---
if st.button("Predict Diabetes"):
    # Collect the input values into a list.
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    
    # If your model requires scaling/normalization, apply that transformation here.
    probability, result = predict_diabetes(input_data)
    
    # Display the result.
    st.subheader("Prediction Result")
    st.write(f"**Probability:** {probability:.2f}")
    st.success(f"The model predicts that the person is: **{result}**")
