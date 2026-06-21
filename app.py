import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

#Configuration
MODEL_PATH = 'models/best_model.keras'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

#Page config
st.set_page_config(
    page_title="FaceEmo",
    page_icon="😊",
    layout="centered"
)

#load model
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return model, face_cascade

model, face_cascade = load_resources()

#helper preprocess face
def preprocess_face(face_img):
    if face_img is None or face_img.size == 0:
        return None
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img/255.0
    face_img = face_img.reshape(1, 48, 48, 1)
    return face_img

#Helper: predict emotion
def predict_emotion(gray_frame, x, y, w, h):
    face_roi = gray_frame[y:y+h, x:x+w]
    if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
        return None, None
    face_input = preprocess_face(face_roi)
    if face_input is None:
        return None, None
    predictions = model.predict(face_input, verbose=0)
    emotion_idx = np.argmax(predictions[0])
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = predictions[0][emotion_idx] * 100
    return emotion, confidence

#process image and draw results
def process_image(image):
    #convert PIL image to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return image, [], "No faces detected in this image."
    
    results = []

    for i, (x, y, w, h) in enumerate(faces):
        emotion, confidence = predict_emotion(gray, x, y, w, h)
        if emotion is None:
            continue

        results.append({
            'face': i+1,
            'emotion': emotion,
            'confidence': confidence
        })

        #Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLOR, 2)
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x, y-10), FONT, 0.9, TEXT_COLOR, 2)

    #Convert back to PIL for Streamlit
    output_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return output_image, results, None

#UI
st.title("😊 FaceEmo")
st.subheader("Real-Time Facial Emotion Detector")
st.markdown("Built with a custom CNN trained on FER2013 — 66% test accuracy across 7 emotions.")
st.divider()

tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Webcam"])

#tab1: image upload
with tab1:
    st.markdown("### Upload a photo to detect emotions")
    uploaded_file = st.file_uploader(
        "Choose an image", type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            st.image(image, use_column_width=True)

        with st.spinner("Detecting emotions..."):
            output_image, results, error = process_image(image)

        with col2:
            st.markdown("**Result**")
            st.image(output_image, use_container_width=True)

        if error:
            st.warning(error)

        else:
            st.markdown("### Results")
            for r in results:
                st.success(f"Face {r['face']}: **{r['emotion']}** ({r['confidence']:.1f}% confidence)")

#webcam tab2
with tab2:
    st.markdown("### Take a photo using your webcam")
    st.info("Click 'Take Photo' to capture and detect emotions")

    webcam_image = st.camera_input("Take Photo")

    if webcam_image is not None:
        image = Image.open(webcam_image).convert('RGB')

        with st.spinner("Detecting emotions..."):
            output_image, results, error = process_image(image)

        st.markdown("**Results**")
        st.image(output_image, use_container_width=True)

        if error:
            st.warning(error)
        else:
            st.markdown("### Results")
            for r in results:
                st.success(f"Face {r['face']}: **{r['emotion']}** ({r['confidence']:.1f}% confidence)")
