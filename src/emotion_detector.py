import cv2
import numpy as np
from tensorflow.keras.models import load_model

#Configuration
MODEL_PATH = '../models/best_model.keras'
#OpenCV's built-in face detector
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#Colors for bounding box (BGR)
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX

#Load model and face detector
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
print("Face detector loaded!")

#helper function: preprocess a face for the model
def preprocess_face(face_img):

    #skip if empty
    if face_img is None or face_img.size == 0:
        return None

    #convert to grayscale if it isn't
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    face_img = cv2.resize(face_img, (48, 48))

    #normalize pixel value 0-256 to 0-1
    face_img = face_img/255.0

    #Reshape to (1, 48, 48, 1) - model expects batch dimension
    face_img = face_img.reshape(1, 48, 48, 1)

    return face_img
    
#detect faces in a frame
def detect_faces(frame):

    #convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # scaleFactor: how much image size is reduced at each scale
    # minNeighbors: how many neighbors each rectangle should have
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return gray, faces

#predict emotion for one face
def predict_emotion(gray_frame, x, y, w, h):
    #crop the face
    face_roi = gray_frame[y:y+h, x:x+w]

    if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
        return None, None

    #preprocess it
    face_input = preprocess_face(face_roi)

    #Skip if preprocessing fails
    if face_input is None:
        return None, None

    #get model predictions - probabilities for each emotion
    predictions = model.predict(face_input, verbose=0)

    #get the emotion with highest probability
    emotion_idx = np.argmax(predictions[0])
    emotion = EMOTION_LABELS[emotion_idx]
    confidence = predictions[0][emotion_idx] * 100

    return emotion, confidence

#Main webcam loop
def run_webcam():
    #open webcam (0 = default webcam)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Webcam started! Press 'q' to quit")

    while True:
        #Read one frame from webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        #Detect faces in the frame
        gray, faces = detect_faces(frame)

        #process each detect face
        for(x, y, w, h) in faces:
            #predict emotion
            emotion, confidence = predict_emotion(gray, x, y, w, h)

            if emotion is None:
                continue

            #draw green rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), BOX_COLOR, 2)

            #display emotion label and confidence above the box
            label = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(frame, label, (x, y-10),
                        FONT, 0.9, TEXT_COLOR, 2)
            
        #Show the frame in a window
        cv2.imshow('FaceEmo - Real Time Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam stopped")

if __name__ == '__main__':
    run_webcam()