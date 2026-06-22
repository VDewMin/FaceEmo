import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

#Configuration
MODEL_PATH = '../models/best_model.keras'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX

#Load model and face detector
print("Loading model...")
model= load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
print("Model loaded!")

#Preprocess face for model
def preprocess_face(face_img):
    if face_img is None or face_img.size == 0:
        return None
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = face_img.reshape(1, 48, 48, 1)
    return face_img

#Predict emotion for one face
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

#Process a single image file
def process_image(image_path):
    #Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return
    
    #Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image - {image_path}")
        return
    
    print(f"Processing: {image_path}")

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        print("No faces found in image.")
        return
    
    for i, (x, y, w, h) in enumerate(faces):
        emotion, confidence = predict_emotion(gray, x, y, w, h)

        if emotion is None:
            continue

        print(f"Face {i+1}: {emotion} ({confidence:.1f}%)")

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLOR, 2)

        
        label = f"{emotion} ({confidence:.1f}%)"
        cv2.putText(frame, label, (x, y-10),
                    FONT, 0.9, TEXT_COLOR, 2)
        
        
        filename = os.path.basename(image_path)
        output_path = f"../models/output_{filename}"
        cv2.imwrite(output_path, frame)
        print(f"Output saved to: {output_path}")

       
        cv2.imshow('FaceEmo - Image Emotion Detection', frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#Main
if __name__ == '__main__':
    #check if image path was provided
    if len(sys.argv) < 2:
        print("Usage: python image_emotion.py <path_to_image>")
        print("Example: python image_emotion.py test.jpg")
    else:
        image_path = sys.argv[1]
        process_image(image_path)