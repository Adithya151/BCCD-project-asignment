Project Overview

This project is a Streamlit-based web application for detecting blood cells in images. Users can upload an image, and the model will predict the type of blood cell (Platelets, RBC, or WBC) and display a bounding box around the detected object.

Project Process:

1️⃣ Data Collection & Preprocessing
The dataset consists of images containing three types of blood cells: Platelets, RBCs, and WBCs.
Each image is labeled with its respective class and bounding box coordinates.
The images are resized to 128x128 pixels and normalized for better model performance.


2️⃣ Model Development
A Convolutional Neural Network (CNN) model was developed using TensorFlow/Keras.
The model has two outputs:
Classification Output – Predicts whether the blood cell is Platelets, RBC, or WBC.
Bounding Box Output – Predicts the position (x1, y1, x2, y2) of the detected cell in the image.
A custom loss function was used, combining categorical crossentropy (for classification) and Huber loss (for bounding box prediction).


3️⃣ Model Training & Evaluation
The model was trained using a dataset of blood cell images with Adam optimizer.
The dataset was split into training and validation sets.
After training, the model was evaluated on a test dataset to ensure accurate detection and localization of blood cells.


4️⃣ Building the Web Application
The Streamlit framework was used to create a simple and interactive web interface.
The app allows users to:
Upload an image.
View the predicted class of the blood cell.
See the detected object highlighted with a bounding box.
The application uses OpenCV to draw the bounding boxes on the uploaded image.


5️⃣ Model Deployment & Execution
The trained model is loaded using TensorFlow's SavedModel format instead of the traditional .h5 format.
The Streamlit app runs locally and can be deployed to Streamlit Cloud or a server.
Users can interact with the model through a web interface without needing to install complex software.
Project Features
✅ Accepts image uploads for analysis.
✅ Predicts the class of the blood cell (Platelets, RBC, or WBC).
✅ Draws bounding boxes to highlight detected objects.
✅ Provides a user-friendly Streamlit-based web interface.
