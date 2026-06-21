# 😊 FaceEmo — Real-Time Facial Emotion Detector

A deep learning project that detects human facial emotions in real-time using a custom CNN trained on the FER2013 dataset.

## 🎯 Features
- Real-time webcam emotion detection
- Image upload emotion detection
- Streamlit web interface
- Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

## 🧠 Model Performance
| Metric | Value |
|---|---|
| Test Accuracy | 66.19% |
| Dataset | FER2013 (35,887 images) |
| Architecture | Custom CNN (6 Conv2D layers) |
| Training Epochs | 50 |

## 📊 Key Results
- Resolved severe overfitting (validation accuracy improved from 19% → 66%)
- Applied data augmentation and class weights to handle class imbalance
- Disgust F1-score improved from 0.38 → 0.60 after class weighting

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy
- Matplotlib / Seaborn
- Scikit-learn

## 📁 Project Structure
FaceEmo/

├── data/               ← FER2013 dataset (not included)

├── models/             ← Saved trained model

├── notebooks/

│   ├── 01_data_exploration.ipynb

│   ├── 02_preprocessing.ipynb

│   ├── 03_model.ipynb

│   └── 04_evaluation.ipynb

├── src/

│   ├── emotion_detector.py   ← Real-time webcam script

│   └── image_emotion.py      ← Image upload script

├── app.py              ← Streamlit web app

└── requirements.txt

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/VDewMin/FaceEmo.git
cd FaceEmo
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download FER2013 from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it in the `data/` folder.

**4. Run the notebooks in order**
01_data_exploration.ipynb

02_preprocessing.ipynb

03_model.ipynb

04_evaluation.ipynb

**5. Launch the Streamlit app**
```bash
streamlit run app.py
```

**6. Or run the webcam script directly**
```bash
cd src
python emotion_detector.py
```

## 📈 Training Process
The model went through three iterations:

1. **First attempt** — Severe overfitting (train: 89%, val: 19%)
2. **Fixed overfitting** — Added data shuffling and augmentation (val: 64%)
3. **Added class weights** — Improved minority class detection (val: 66.19%)

## 🔍 Confusion Matrix Insights
- **Happy** and **Surprise** perform best (clear visual features)
- **Fear** and **Disgust** are hardest (visually similar to other emotions + limited training data)
- **Fear vs Sad** is the most common confusion

## 🔮 Future Improvements
- Transfer learning with MobileNet or ResNet50
- Larger dataset or data collection for underrepresented emotions
- Real-time video file processing

## 👩‍💻 Author
**W.A. Vindiya** — 3rd Year BSc IT (Data Science), SLIIT  
[GitHub](https://github.com/VDewMin)