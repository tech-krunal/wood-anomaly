# Wood Anomaly Detection App

This project uses a TensorFlow/Keras model trained on the MVTec wood dataset using [Teachable Machine](https://teachablemachine.withgoogle.com/) and deployed with [Streamlit](https://streamlit.io/).

## 📦 Features

- Detects anomalies in wood product images (e.g., hole, color, scratch, etc.)
- Users can upload or use webcam to check wood quality
- Real-time classification as "Good" or "Bad"

## 🧠 Model

The model was trained using Teachable Machine with 2 classes:
- Good (images from `train/good`)
- Bad (images from `test/color`, `hole`, etc.)

Exported as a `.h5` Keras model and integrated into this Streamlit app.

## 🚀 Run Locally

```bash
git clone https://github.com/mrkparekh/wood-anomaly-detector.git
cd wood-anomaly-detector
python -m venv venv
venv\Scripts\activate   # or source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
