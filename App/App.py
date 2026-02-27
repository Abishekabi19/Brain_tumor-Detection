import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam
from PIL import Image

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("brain_tumor_model.h5")

# Compile model (removes warning)
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Class labels
class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to detect brain tumor")

st.write("---")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an MRI image...",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    # Convert image to RGB (fix grayscale issue)
    image = Image.open(uploaded_file).convert("RGB")

    # Show image
    st.image(image, caption="Uploaded MRI Image", width=300)

    # Resize image
    img = image.resize((150, 150))

    # Convert image to array
    img_array = img_to_array(img)

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize image
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]
    confidence = np.max(prediction) * 100

    st.write("---")

    # Show prediction
    st.success(f"Prediction: {predicted_label}")

    # Show confidence
    st.info(f"Confidence: {confidence:.2f}%")

    # Show all probabilities
    st.write("### Prediction Probabilities:")
    
    for i in range(len(class_labels)):
        st.write(f"{class_labels[i]}: {prediction[0][i]*100:.2f}%")

st.write("---")
st.write("Developed using CNN and Streamlit")