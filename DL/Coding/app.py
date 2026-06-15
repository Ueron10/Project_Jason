from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import re
import os
import nltk
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

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

model = None
tokenizer = None
label_classes = None

def load_artifacts():
    global model, tokenizer, label_classes
    if model is None:
        model = tf.keras.models.load_model(os.path.join(models_dir, "lstm_model.h5"))
        with open(os.path.join(outputs_dir, "tokenizer.pkl"), "rb") as f:
            tokenizer = pickle.load(f)
        label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)
    return model, tokenizer, label_classes

def expand_contractions(text):
    contractions = {
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "won't": "will not", "wouldn't": "would not", "can't": "cannot",
        "couldn't": "could not", "shouldn't": "should not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn": "had not",
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

def preprocess_text(text):
    if not text:
        return ""
    text = expand_contractions(text)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
    text = text.strip()
    return text

def predict_sentiment(text):
    global model, tokenizer, label_classes
    model, tokenizer, label_classes = load_artifacts()
    cleaned = preprocess_text(text)
    if not cleaned:
        return None
    sequences = tokenizer.texts_to_sequences([cleaned])
    padded_sequences = pad_sequences(sequences, maxlen=128, padding='post', truncating='post')
    probs = model.predict(padded_sequences, verbose=0)[0]
    labels = [str(label).lower() for label in label_classes]
    probabilities = {labels[i]: float(probs[i]) for i in range(len(labels))}
    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx])
    return {
        'label': pred_label,
        'confidence': confidence,
        'probabilities': probabilities,
        'cleaned': cleaned
    }

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Please enter some text'}), 400
    result = predict_sentiment(text)
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
