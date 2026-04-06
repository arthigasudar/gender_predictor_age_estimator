import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Caching model load to prevent reloading
# ==========================================
@st.cache_resource
def load_keras_model():
    return tf.keras.models.load_model('models/gender_age_model.h5', compile=False)

# Attempt to load model
try:
    model = load_keras_model()
    model_loaded = True
except Exception as e:
    st.error(f"Internal Error loading model: {e}")
    model_loaded = False

# Load OpenCV face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("🤖 Gender Prediction & Age Estimation")
st.markdown("Upload a photo to detect faces and predict gender and age using deep learning **(CNN)**.")

if not model_loaded:
    st.warning("⚠️ **Model not found!** Please train the model and ensure it is saved at `models/gender_age_model.h5` first.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the uploaded image (Balanced strictly)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10, minSize=(100, 100))
    
    st.image(image, caption='Original Uploaded Image', use_column_width=True)
    st.info(f"Detected **{len(faces)} face(s)** in the image.")
    
    if len(faces) > 0:
        # Create a copy of the image to draw bounding boxes
        output_img = img_array.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract the actual face bounding box
            face_img = img_array[y:y+h, x:x+w]
            
            # Preprocess the face image for the model
            face_resized = cv2.resize(face_img, (64, 64))
            face_resized = face_resized.astype(np.float32) / 255.0
            face_input = np.expand_dims(face_resized, axis=0) 
            
            # Prediction
            preds = model.predict(face_input)
            gender_pred = preds[0][0][0] # Index 0 in output list is Gender
            age_pred_probs = preds[1][0] # Index 1 is array of 5 age probabilities
            
            gender = "Female" if gender_pred > 0.5 else "Male"
            
            age_groups = ["Child (0-12)", "Teen (13-19)", "Young Adult (20-35)", "Adult (36-55)", "Senior (56+)"]
            age_class = np.argmax(age_pred_probs)
            age_label = age_groups[age_class]
            
            # Format text for display
            label = f"{gender}, {age_label}"
            
            # Draw rectangle bounds and prediction text
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.rectangle(output_img, (x, y-35), (x+250, y), (0, 255, 0), -1)
            cv2.putText(output_img, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Output textual info to streamlit
            st.success(f"**Face {i+1}:** {gender}, Age Category: **{age_label}**")
            
        st.markdown("### Processed Image Highlight")
        st.image(output_img, caption="Faces Analyzed", use_column_width=True)
    else:
        st.warning("No faces were detected in the supplied image. Try another photo.")
