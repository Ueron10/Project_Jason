import numpy as np
import pickle
import re
import os
import nltk
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

# Load artifacts
print("Loading BERT Large model...")
model = TFBertForSequenceClassification.from_pretrained(os.path.join(models_dir, "bert_model"))
with open(os.path.join(outputs_dir, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)

# Preprocessing setup - Optimized for BERT
minimal_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'up', 'down', 'out', 'over', 'under',
                    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                    'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
                    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
                    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
                    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                    'won', "won't", 'wouldn', "wouldn't"}
stop_words = minimal_stopwords

def expand_contractions(text):
    """Expand common contractions for better BERT understanding"""
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "can't": "cannot",
        "couldn't": "could not", "shouldn't": "should not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "i'm": "i am", "you're": "you are", "he's": "he is",
        "she's": "she is", "it's": "it is", "we're": "we are",
        "they're": "they are", "i've": "i have", "you've": "you have",
        "we've": "we have", "they've": "they have", "i'll": "i will",
        "you'll": "you will", "he'll": "he will", "she'll": "she will",
        "we'll": "we will", "they'll": "they will", "i'd": "i would",
        "you'd": "you would", "he'd": "he would", "she'd": "she would",
        "we'd": "we would", "they'd": "they would"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess(text):
    """Optimized preprocessing for BERT - keeps more context"""
    if not text:
        return ""
    
    # Expand contractions first
    text = expand_contractions(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Keep letters, numbers, and basic punctuation (BERT handles punctuation well)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def predict(text):
    cleaned = preprocess(text)
    if not cleaned:
        return None
    # BERT Large tokenization
    encoded = tokenizer(
        cleaned,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='tf'
    )
    # BERT Large prediction
    logits = model.predict(
        {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']},
        verbose=0
    )
    probs = tf.nn.softmax(logits.logits, axis=-1).numpy()[0]
    labels = [str(label).lower() for label in label_classes]
    pred_idx = int(np.argmax(probs))
    return {
        'label': labels[pred_idx],
        'confidence': float(probs[pred_idx]),
        'all_probs': {labels[i]: float(probs[i]) for i in range(len(labels))},
        'cleaned': cleaned
    }

# Interactive mode
print("Sentiment Analysis Prediction")
print("="*50)
print("Enter text to analyze (or 'quit' to exit)\n")

while True:
    text = input("> ").strip()
    if text.lower() in ['quit', 'exit', 'q']:
        break
    if not text:
        continue
    
    result = predict(text)
    if result is None:
        print("\n  Result: unable to predict (cleaned text is empty)\n")
        continue
    print(f"\n  Result: {result['label'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print()

print("Goodbye!")
