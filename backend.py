# backend.py

import numpy as np
import tensorflow as tf
import streamlit as st

# Cache the model so it's loaded only once.
@st.cache_resource
def load_model():
    """
    Loads and returns the Keras model from 'diabeties.h5'.
    Make sure the file path is correct.
    """
    model = tf.keras.models.load_model("d_model.h5")
    return model

def predict_diabetes(input_data):
    """
    Predicts diabetes based on input data.
    
    Parameters:
        input_data (list): A list of 8 numerical features in the order:
            [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    
    Returns:
        probability (float): The output probability from the model.
        result (str): 'Diabetic' if probability > 0.5 else 'Not Diabetic'.
    
    Note:
        If your model was trained on scaled data, include the appropriate scaling logic here.
    """
    model = load_model()
    # Convert the input list into a numpy array with shape (1, 8)
    input_array = np.array([input_data])
    
    # Make a prediction using the model
    prediction = model.predict(input_array)
    
    # Assuming the model outputs a probability using a sigmoid activation.
    probability = prediction[0][0]
    result = "Diabetic" if probability > 0.5 else "Not Diabetic"
    return probability, result
