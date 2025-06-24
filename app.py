import streamlit as st
from model import load_model
from preprocess import clean_text
import numpy as np

# Set page config
st.set_page_config(page_title="Emotion Detector", page_icon="💭", layout="centered")

# Cache the model so it loads only once
@st.cache_resource
def get_model():
    model = load_model()
    if model is None:
        st.error("Model not found. Please run train.py first.")
        st.stop()
    return model

# Load the model
model = get_model()

# Emojis for each emotion
emotion_emoji_map = {
    "joy": "😄",
    "sadness": "😢",
    "fear": "😨",
    "anger": "😡",
    "disgust": "🤢",
    "surprise": "😮",
    "happiness": "😊",
    "sad": "😞",
    "neutral": "😐",
    "disappointment": "😔",
    "admiration": "😊",
    "gratitude": "🙏",
    "optimism": "😊",
    "annoyance": "😒",
    "approval": "👍",
    "caring": "❤️",
    "confusion": "😕",
    "curiosity": "🤔",
    "desire": "😍",
    "embarrassment": "😳",
    "excitement": "🎉",
    "grief": "😢",
    "love": "❤️",
    "nervousness": "😰",
    "pride": "🦁",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔"
}

# Custom styling for the page
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #4B8BBE;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        font-size: 1.1em;
    }
    .result-box {
        background-color: #1e2021;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5em;
        margin-top: 20px;
        color: white;
    }
    .emotion-display {
        font-size: 2em;
        margin: 20px 0;
    }
    .confidence {
        font-size: 1.2em;
        color: #4B8BBE;
        margin-top: 10px;
    }
    .code-section {
        background-color: #1e2021;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: white;
    }
    .code-title {
        color: #4B8BBE;
        font-size: 1.2em;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introductory message
st.markdown('<div class="title">💭 Emotion Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a sentence and let the model predict the emotion.</div>', unsafe_allow_html=True)

# Text input box
tweet = st.text_area("✍️ Your Tweet", height=150, placeholder="e.g. I am so happy today")

# Prediction logic
if st.button("🔍 Analyze"):
    if not tweet.strip():
        st.warning("Please enter a sentence.")
    else:
        try:
            cleaned = clean_text(tweet)
            prediction = model.predict([cleaned])[0]
            probabilities = model.predict_proba([cleaned])[0]
            
            # Get emotion names in the same order as the model's classes
            emotion_names = model.classes_
            
            # Create a dictionary of emotion -> probability
            emotion_probs = dict(zip(emotion_names, probabilities))
            
            # Get the top emotion
            top_emotion = max(emotion_probs.items(), key=lambda x: x[1])
            emotion, prob = top_emotion
            emoji = emotion_emoji_map.get(emotion.lower(), "😶")
            percentage = prob * 100
            
            # Display results
            st.markdown(f"""
            <div class="result-box">
                <div class="emotion-display">
                    {emoji} {emotion.title()}
                </div>
                <div class="confidence">
                    Confidence: {percentage:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Code sections
with st.expander("📝 Code Implementation"):
    st.markdown("""
    ### Preprocessing Code
    ```python
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Words that indicate negation
    negation_words = {
        'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'nowhere',
        'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'shouldnt', 'couldnt', 'cant',
        'havent', 'hasnt', 'hadnt', 'arent', 'isnt', 'wasnt', 'werent'
    }

    # Words that indicate strong emotion
    emotion_indicators = {
        'hate', 'love', 'angry', 'sad', 'happy', 'fear', 'scared', 'excited',
        'disgust', 'surprise', 'shock', 'disappoint', 'annoy', 'frustrate',
        'terrible', 'awful', 'horrible', 'amazing', 'wonderful', 'great'
    }

    def clean_text(text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and mentions
        text = re.sub(r"http\S+|www\S+", '', text)
        text = re.sub(r"@\w+|#\w+", '', text)
        
        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        
        # Remove special characters but keep basic punctuation and negation words
        text = re.sub(r"[^a-zA-Z\s.,!?]", '', text)
        
        # Tokenize and lemmatize
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Keep negation words and emotion indicators
        filtered_tokens = []
        for token in tokens:
            if (token not in stop_words or 
                token in negation_words or 
                any(indicator in token for indicator in emotion_indicators)):
                filtered_tokens.append(token)
        
        # Add negation context
        if any(word in filtered_tokens for word in negation_words):
            filtered_tokens.append('negation')
        
        return ' '.join(filtered_tokens)
    ```

    ### Model Code
    ```python
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report

    def build_model():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=15000,
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=100,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        return pipeline

    def load_model():
        try:
            return joblib.load("emotion_model.pkl")
        except FileNotFoundError:
            print("Model file not found. Please run train.py first.")
            return None

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print feature importance for top features
        feature_names = model.named_steps['tfidf'].get_feature_names_out()
        importances = model.named_steps['clf'].feature_importances_
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
        print("\nTop 20 important features:")
        for feature, importance in top_features:
            print(f"{feature}: {importance:.4f}")
    ```
    """)

# Model information
with st.expander("📊 Model Information"):
    st.markdown("""
    - Model: Random Forest Classifier
    - Features: TF-IDF with n-grams (1-3)
    - Training: Balanced class weights
    - Preprocessing: Advanced text cleaning with negation and emotion detection
    """)
    st.code("Accuracy: 0.76")
