import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model = load_model('model/pretrained_tumor_model.keras')

# Class mapping
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Set Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Inject internal CSS
st.markdown("""
    <style>
    .stMainBlockContainer {
        padding: 4rem 2rem 1rem;
        max-width: 800px;
    }
    h3 {
        width:800px;
    }
    </style>
""", unsafe_allow_html=True)

# Session state default page
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

st.title("🧠 Brain Tumor Classification App")

class Pages:
    def home(self):
        with col1:
            st.image('images/Home.jpg')

        with col2:
            st.markdown("""
                This application uses a pre-trained deep learning model to classify MRI brain images into four categories:
                **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary Tumor**.

                Upload your MRI image to get instant predictions along with confidence scores.
                """, unsafe_allow_html=True)

            st.session_state['upload'] = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])
            
            if st.button("🔍 Submit for Prediction"): 
                st.session_state['page'] = 'result'
                st.rerun()

    def result(self):
        if st.session_state['upload'] is not None:
            with col1:
                img = Image.open(st.session_state['upload']).convert("RGB")
                st.image(img, caption="Uploaded Image", use_container_width=True)

            # Preprocess image
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predict
            predictions = model.predict(img_array)[0]
            predicted_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_index]
            confidence = predictions[predicted_index] * 100

            with col2:
                st.markdown(f"### 🧾 Prediction: `{predicted_label.upper()}`")
                st.markdown(f"**Confidence:** {confidence:.2f}%")

                st.subheader("📊 Prediction Probabilities:")
                for i, prob in enumerate(predictions):
                    st.write(f"{class_labels[i].capitalize()}: {prob*100:.2f}%")

                if st.button('Home'): 
                    st.session_state['page'] = 'home'
                    st.rerun()
        else:
            st.session_state['page'] = 'home'


# Render layout
page = Pages()

with st.container(border=True):
    col1, col2 = st.columns(2, gap='medium', vertical_alignment='center')
    
    if st.session_state['page'] == 'home':
        page.home()
    elif st.session_state['page'] == 'result':
        page.result()