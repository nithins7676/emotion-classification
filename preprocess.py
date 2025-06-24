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
