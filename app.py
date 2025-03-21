import tensorflow 
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ✅ Classes (Ensure this matches your dataset)
CLASSES = ['Platelets', 'RBC', 'WBC']

# ✅ Load the model without requiring custom objects
try:
    model = tf.keras.models.load_model("simple_object_detector.h5", compile=False)  # Don't compile the model
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ✅ Preprocessing Function
def preprocess_image(image, target_size=(128, 128)):
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# ✅ Streamlit UI for Prediction
st.title("🩸 Blood Cell Detection App")
st.write("Upload an image for predictions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Load and display the uploaded image
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Preprocess the image for the model
    processed_image = preprocess_image(image)

    # ✅ Make predictions
    try:
        predictions = model.predict(processed_image)
        st.success("✅ Prediction successful!")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.stop()

    # ✅ Extract the predicted class
    cls_pred = predictions[0][:len(CLASSES)]  # Extract class prediction probabilities
    predicted_class = np.argmax(cls_pred)
    confidence_score = cls_pred[predicted_class]

    # ✅ Display the result
    st.write(f"**Predicted Class:** {CLASSES[predicted_class]}")
    st.write(f"**Confidence Score:** {confidence_score:.2f}")
