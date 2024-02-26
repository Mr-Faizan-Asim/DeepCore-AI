import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('your_model_path.h5')  # Replace 'your_model_path.h5' with the path to your trained model file

# Function for preprocessing the captured frame
def preprocess_image(frame):
    # Add your image preprocessing steps (e.g., resize, convert to grayscale)
    # ...

    return processed_frame

# Open a connection to the camera (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Display the captured frame
    cv2.imshow('Captured Frame', frame)

    # Preprocess the frame (grayscale, resize, etc.)
    processed_frame = preprocess_image(frame)

    # Reshape the processed frame to match the input shape expected by the model
    processed_frame = np.expand_dims(processed_frame, axis=0)

    # Apply the trained model to make a prediction
    prediction = model.predict(processed_frame)

    # Display the result
    result_text = "Malaria Diagnosis: Infected" if prediction > 0.5 else "Malaria Diagnosis: Uninfected"
    print(result_text)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
