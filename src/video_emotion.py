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
FONT = cv2.FONT_HERSHEY_SIMPLEX

#Load model and face detector
print("Loading model...")
model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
print("Model loaded!")

#Preprocess face 
def preprocess_face(face_img):
    if face_img is None or face_img.size == 0:
        return None
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = face_img.reshape(1, 48, 48, 1)
    return face_img

# Predict emotion 
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

#  Process video file 
def run_video(video_path):
    
    if not os.path.exists(video_path):
        print(f"Error: File not found — {video_path}")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('FaceEmo - Video Emotion Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('FaceEmo - Video Emotion Detector', 960, 540)

    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Video loaded! FPS: {fps:.1f}, Duration: {duration:.1f}s")
    print("Press 'q' to quit, 'p' to pause/resume")

    paused = False

    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()

            # Loop video when it ends
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            
            if frame_count % 3 == 0:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
            frame_count += 1

            
            for (x, y, w, h) in faces:
                emotion, confidence = predict_emotion(gray, x, y, w, h)
                if emotion is None:
                    continue

              
                cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLOR, 2)
                label = f"{emotion} ({confidence:.1f}%)"
                label_y = y - 10 if y - 10 > 20 else y + h + 25
                cv2.putText(frame, label, (x, label_y),
                        FONT, 0.9, TEXT_COLOR, 2)

           
            cv2.imshow('FaceEmo - Video Emotion Detector', frame)

        # Key controls
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")

# Main 
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python video_emotion.py <path_to_video>")
        print("Example: python video_emotion.py clip.mp4")
    else:
        video_path = sys.argv[1]
        run_video(video_path)