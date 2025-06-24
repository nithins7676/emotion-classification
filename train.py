import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from preprocess import clean_text
from model import build_model

# Load data
df = pd.read_csv("goemotions/train.tsv", sep="\t", names=["text", "labels", "ids"])

# Load label names (emotions)
with open("goemotions/emotions.txt") as f:
    emotion_list = [line.strip() for line in f]

# Keep only single-label examples for simplicity
df = df[df['labels'].str.count(',') == 0]
df['labels'] = df['labels'].astype(int)
df['emotion'] = df['labels'].map(lambda x: emotion_list[x])

# Clean text
print("Cleaning text data...")
df['text'] = df['text'].apply(clean_text)

# Remove very short texts
df = df[df['text'].str.split().str.len() >= 3]

# Balance the dataset by undersampling majority classes
print("Balancing dataset...")
min_samples = df['emotion'].value_counts().min()
balanced_df = pd.DataFrame()
for emotion in df['emotion'].unique():
    emotion_df = df[df['emotion'] == emotion]
    if len(emotion_df) > min_samples:
        emotion_df = emotion_df.sample(min_samples, random_state=42)
    balanced_df = pd.concat([balanced_df, emotion_df])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    balanced_df['text'], 
    balanced_df['emotion'], 
    test_size=0.2, 
    random_state=42,
    stratify=balanced_df['emotion']
)

# Build and train model
print("Training model...")
model = build_model()

# Perform cross-validation
print("Performing cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Fit the model
model.fit(X_train, y_train)

# Save model
print("Saving model...")
joblib.dump(model, "emotion_model.pkl")

# Evaluate on test set
print("\nEvaluating on test set...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.2f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print class distribution
print("\nClass distribution in test set:")
print(y_test.value_counts())
