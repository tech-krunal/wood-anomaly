import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

# Set page config
st.set_page_config(
    page_title="Wood Anomaly Detector",
    page_icon="🪵",
    layout="centered"
)
st.markdown("""
    <div style="background-color:#0D9488;padding:1rem;border-radius:10px;margin-bottom:1rem;">
        <h1 style="color:white;text-align:center;">🪵 Wood Anomaly Detector</h1>
        <p style="color:white;text-align:center;">AI-Powered Quality Control for Wood Products</p>
    </div>
""", unsafe_allow_html=True)


# Load model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Styling with Markdown and CSS
st.markdown(
    """
    <style>
        .main {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            border-radius: 8px;
            padding: 0.5rem 1rem;
            background-color: #007bff;
            color: white;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar info
with st.sidebar:
    st.title("🧾 Project Info")
    st.subheader("Wood Anomaly Detector")
    st.write(
        "Detect defects in wood products using a trained machine learning model built with Teachable Machine."
    )
    st.markdown("---")
    st.subheader("📂 Classes Detected")
    st.markdown("- ✅ Good\n- ❌ Anomaly (hole, scratch, liquid, etc.)")
    st.markdown("---")
    st.write("Created by: *Kushal Parekh*")

# Header
st.title("🪵 Wood Anomaly Detection System")
st.caption("Upload or capture an image to check if it's GOOD or has ANOMALY.")

# Image input method
input_method = st.radio("Choose Input Method", ["📁 Upload Image", "📷 Use Camera"])

image = None

if input_method == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a Wood Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=400)
            st.success("Image uploaded successfully!")
        except Exception as e:
            st.error(f"Error loading image: {e}")

elif input_method == "📷 Use Camera":
    camera_image = st.camera_input("Take a Picture")
    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert("RGB")
            st.image(image, caption="Captured Image", width=400)
            st.success("Image captured successfully!")
        except Exception as e:
            st.error(f"Error loading camera image: {e}")

# Prediction function
def predict_image(img):
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img).astype(np.float32)
    normalized_image_array = (img_array / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = prediction[0][index]
    return class_name, confidence

def show_prediction_result(class_name, confidence):
    if "Good" in class_name:
        st.success(f"✅ The product is **Good** with {confidence*100:.2f}% confidence.")
        
        st.progress(min(float(confidence), 1.0))  # Progress bar up to 100%
        
        st.markdown("""
        **Prediction Breakdown**  
        🟩 Good: {:.2f}%  
        🟥 Anomaly: {:.2f}%
        """.format(confidence * 100, 100 - confidence * 100))
    else:
        st.error(f"⚠️ Anomaly detected: **Bad** with {confidence*100:.2f}% confidence.")

        st.progress(min(float(confidence), 1.0))  # Progress bar

        st.markdown("""
        **Prediction Breakdown**  
        🟩 Good: {:.2f}%  
        🟥 Anomaly: {:.2f}%
        """.format(100 - confidence * 100, confidence * 100))

# Submit button
if st.button("🔍 Run Detection") and image:
    with st.spinner("Analyzing image..."):
        result, confidence = predict_image(image)
        st.subheader("📊 Prediction Result")
        show_prediction_result(result.strip(), confidence)


else:
    st.info("Upload or capture an image and click 'Run Detection' to analyze.")

