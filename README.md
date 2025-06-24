# 🧠 Emotion Classification from Text 😃😢😡

Welcome to the **Emotion Classification** project! This repository contains a machine learning pipeline to detect emotions from text using the GoEmotions dataset. Whether you're building a chatbot, analyzing social media, or just curious about NLP, this project is for you! 🚀

## 📦 Features
- Detects a wide range of emotions from text
- Preprocessing, training, and evaluation scripts included
- Uses scikit-learn for easy customization
- Ready-to-use trained model

## 📁 Project Structure
```
GALLERY/
├── app.py                # (Optional) App interface for predictions
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

## 🛠️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/nithins7676/emotion-classification.git
   cd emotion-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install: pandas, numpy, scikit-learn, joblib)*

3. **Train the model**
   ```bash
   python train.py
   ```
   This will preprocess the data, train the model, and save it as `emotion_model.pkl`.

4. **(Optional) Predict emotions for new text**
   - Use `app.py` (if implemented) or load `emotion_model.pkl` in your own script.

## 📝 Example Usage
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

Made with ❤️ and Python 🐍 