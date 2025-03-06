import streamlit as st
import tensorflow as tf
import os
import gdown
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="ClassiFy-153",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model loading and preprocessing functions
import gdown
import zipfile

# Google Drive file ID (replace with your actual ID)
GDRIVE_MODEL_ID = "12outZsKRCu6G0IBBbPlHluoAi6lmoxLK"
MODEL_ZIP_PATH = "ensemble_model_tf.zip"
MODEL_DIR = "ensemble_model_tf"

@st.cache_resource()
def load_model():
    # Download model only if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        st.info("⏳ Downloaded model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}", MODEL_ZIP_PATH, quiet=False)
        
        # Extract ZIP
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(".")
        
        # Remove ZIP after extraction
        os.remove(MODEL_ZIP_PATH)

    # Load model
    return tf.keras.models.load_model(MODEL_DIR)

def get_class_names():
    return np.sort(os.listdir("dataset")).tolist()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Enhanced Styling with additional interactions and a video game aesthetic
st.markdown("""
    <style>
    /* Import futuristic fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Space+Grotesk:wght@400;700&display=swap');
    
    :root {
        --void-black: #020204;
        --stardust-white: #E5E8F0;
        --nebula-purple: #6A5ACD;
        --quantum-blue: #008CBA;
        --supernova-pink: #FF1493;
        --cosmic-cyan: #00CED1;
        --galaxy-gold: #FFD700;
    }
    
    .stApp {
        background: radial-gradient(ellipse at center, var(--void-black), #000022);
        color: var(--stardust-white);
        font-family: 'Space Grotesk', sans-serif;
        min-height: 100vh;
    }
    
    /* Main title styling with glow effect */
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem;
        text-align: center;
        background: linear-gradient(135deg, var(--nebula-purple), var(--quantum-blue), var(--supernova-pink));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: title-glow 3s ease-in-out infinite;
        letter-spacing: 3px;
        margin: 2rem 0;
    }
    
    @keyframes title-glow {
        0% { text-shadow: 0 0 30px var(--nebula-purple); }
        50% { text-shadow: 0 0 50px var(--quantum-blue), 0 0 30px var(--supernova-pink); }
        100% { text-shadow: 0 0 30px var(--nebula-purple); }
    }
    
    /* Upload container with subtle glassmorphism */
    .upload-container {
        border: 2px solid var(--nebula-purple);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        background: rgba(15, 20, 40, 0.4);
        backdrop-filter: blur(10px);
        box-shadow: 0 0 40px rgba(138, 124, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .upload-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 60px rgba(138, 124, 255, 0.4);
    }
    
    /* Prediction box with a futuristic border glow */
    .prediction-box {
        background: linear-gradient(145deg, rgba(15, 20, 40, 0.9), rgba(45, 50, 80, 0.9));
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid var(--quantum-blue);
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-top: 2rem;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, var(--cosmic-cyan), transparent);
        animation: border-glow 4s linear infinite;
        opacity: 0.3;
    }
    
    @keyframes border-glow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .predicted-class {
        font-size: 2.5rem;
        color: var(--galaxy-gold);
        font-weight: bold;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.4);
        margin: 1rem 0;
    }
    
    .confidence-label {
        font-size: 1.3rem;
        color: var(--cosmic-cyan);
        margin-top: 1.5rem;
        letter-spacing: 1px;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--supernova-pink), var(--cosmic-cyan));
        height: 15px;
        border-radius: 8px;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
    }
    
    /* Sidebar button styling */
    .sidebar .stButton button {
        background: linear-gradient(135deg, var(--nebula-purple), var(--quantum-blue));
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .sidebar .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(138, 124, 255, 0.4);
    }
    
    /* Animal list styling in sidebar */
    .animal-list {
        max-height: 60vh;
        overflow-y: auto;
        padding-right: 1rem;
        margin-top: 1rem;
    }
    
    .animal-list::-webkit-scrollbar {
        width: 6px;
    }
    
    .animal-list::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 3px;
    }
    
    .animal-list::-webkit-scrollbar-thumb {
        background: var(--quantum-blue);
        border-radius: 3px;
    }
    
    .animal-item {
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .animal-item:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(10px);
    }
    
    /* Image frame styling with fixed height to avoid stretching */
    .image-frame {\n        border: 2px solid var(--quantum-blue);\n        border-radius: 20px;\n        padding: 10px;\n        box-shadow: 0 0 40px rgba(0, 212, 255, 0.2);\n        position: relative;\n        overflow: hidden;\n        background: rgba(0, 0, 0, 0.3);\n    }\n    \n    .image-frame img {\n        height: 300px;\n        max-width: 100%;\n        object-fit: contain;\n        display: block;\n        margin: 0 auto;\n    }\n    \n    .image-frame::after {\n        content: '';\n        position: absolute;\n        top: -50%;\n        left: -50%;\n        width: 200%;\n        height: 200%;\n        background: linear-gradient(45deg, transparent, rgba(0, 206, 209, 0.1), transparent);\n        animation: scan 6s linear infinite;\n    }\n    \n    @keyframes scan {\n        0% { transform: rotate(45deg) translate(-30%, -30%); }\n        100% { transform: rotate(45deg) translate(70%, 70%); }\n    }\n    \n    /* Reset button styling */\n    .reset-button {\n        margin-top: 1rem;\n        background: linear-gradient(135deg, var(--cosmic-cyan), var(--supernova-pink));\n        border: none;\n        border-radius: 8px;\n        padding: 0.6rem 1rem;\n        font-size: 1rem;\n        cursor: pointer;\n        transition: transform 0.3s ease;\n    }\n    \n    .reset-button:hover {\n        transform: scale(1.05);\n    }\n    \n    /* Footer styling */\n    .footer {\n        text-align: center;\n        margin-top: 2rem;\n        padding: 1rem;\n        font-size: 0.9rem;\n        color: #aaa;\n    }\n    </style>\n""", unsafe_allow_html=True)

def main():
    model = load_model()
    class_names = get_class_names()
    
    st.markdown('<h1 class="main-title">🌠 ClassiFy-153 - Galactic Animal Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar - Navigation for Celestial Species and Model Info
    with st.sidebar:
        st.markdown("### 🪐 Navigation")
        if st.button("🪐 View Celestial Species"):
            st.markdown("#### 🌠 Supported Animal Classes")
            with st.container():
                st.markdown('<div class="animal-list">', unsafe_allow_html=True)
                for animal in class_names:
                    st.markdown(f'<div class="animal-item">{animal}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        if st.button("🤖 About the Model"):
            st.markdown("##### Galactic Ensemble Model Info")
            st.info("This ensemble model leverages state-of-the-art neural networks with cosmic accuracy. Trained on a diverse dataset of animal images, it uses transfer learning to achieve stellar performance.")
    
    # Upload Section with enhanced styling
    with st.container():
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📡 Upload an image for cosmic classification", type=["jpg", "png", "jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        image_col, result_col = st.columns([1.2, 1])
        image = Image.open(uploaded_file)

        with image_col:
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(image, caption="🛸 Captured Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with result_col:
            with st.spinner('🚀 Scanning with quantum neural networks...'):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                # Display results in a styled prediction box
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="predicted-class">✨ {predicted_class.upper()} ✨</div>', unsafe_allow_html=True)
                st.progress(confidence / 100)
                st.markdown(f'<div class="confidence-label">Model Certainty: {confidence:.2f}%</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with credits
    st.markdown('<div class="footer">Powered by Galactic Neural Networks © 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
