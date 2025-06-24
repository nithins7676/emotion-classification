import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from preprocess import clean_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_path = 'sentiment_model.joblib'
        self.vectorizer_path = 'vectorizer.joblib'
        
    def load_or_train(self, data_path='tweets.csv'):
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            print("Loading existing model...")
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            print("Training new model...")
            self.train_model(data_path)
            
    def train_model(self, data_path):
        # Load and preprocess data
        df = pd.read_csv(data_path)
        df['text'] = df['text'].apply(clean_text)
        
        # Split data
        X = df['text']
        y = df['sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )),
            ('clf', LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear'
            ))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model and vectorizer
        joblib.dump(self.model, self.model_path)
        self.vectorizer = self.model.named_steps['tfidf']
        joblib.dump(self.vectorizer, self.vectorizer_path)
        
    def predict_sentiment(self, text):
        if self.model is None:
            raise ValueError("Model not loaded or trained")
            
        # Clean text
        cleaned_text = clean_text(text)
        
        # Predict
        prediction = self.model.predict([cleaned_text])[0]
        probabilities = self.model.predict_proba([cleaned_text])[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        return {
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0],
                'neutral': probabilities[1],
                'positive': probabilities[2]
            }
        }

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
