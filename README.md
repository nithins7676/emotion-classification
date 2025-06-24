# 💭 Emotion Detector Web App 😃😢😡

Welcome to the **Emotion Detector Web App**! This project is a user-friendly web application that detects emotions from text using a machine learning model trained on the GoEmotions dataset. Whether you're building a chatbot, analyzing social media, or just curious about NLP, this project is for you! 🚀

## 📦 Features
- Detects a wide range of emotions from text
- Beautiful and interactive web interface (Streamlit)
- Preprocessing, training, and evaluation scripts included
- Uses scikit-learn for easy customization
- Ready-to-use trained model

## 📁 Project Structure
```
GALLERY/
├── app.py                # Streamlit web app for emotion detection
├── train.py              # Training and evaluation script
├── model.py              # Model pipeline definition
├── preprocess.py         # Text cleaning utilities
├── goemotions/           # Dataset and label files
│   ├── train.tsv
│   ├── test.tsv
│   ├── dev.tsv
│   ├── emotions.txt
│   └── ekman_mapping.json
├── emotion_model.pkl     # Saved trained model
├── emotion.txt           # (Optional) Additional emotion info
└── README.md             # Project documentation
```

## 🛠️ How to Run the Web App

1. **Clone the repository**
   ```bash
   git clone https://github.com/nithins7676/emotion-classification.git
   cd emotion-classification
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit scikit-learn pandas numpy joblib
   ```

3. **Train the model (if not already trained)**
   ```bash
   python train.py
   ```
   This will preprocess the data, train the model, and save it as `emotion_model.pkl`.

4. **Run the web app**
   ```bash
   streamlit run app.py
   ```
   Then open your browser and go to the local URL shown in the terminal (usually http://localhost:8501).

## 📝 Example: Predicting Emotions in Python
```python
from joblib import load
model = load('emotion_model.pkl')
text = ["I am so happy to see you!"]
pred = model.predict(text)
print(f"Predicted emotion: {pred[0]}")
```

## 🤖 Dataset
- Based on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions)
- Contains 27 emotion labels + neutral

## ✨ Contributing
Pull requests are welcome! For major changes, please open an issue first.

## 📧 Contact
For questions, open an issue or contact the maintainer.

---

Made with ❤️, Streamlit, and Python 🐍 