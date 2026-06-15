import pandas as pd
import re
import nltk
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")
outputs_dir = os.path.join(project_root, "Outputs")
os.makedirs(outputs_dir, exist_ok=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

print("="*70)
print("PREPROCESSING - ALL STEPS")
print("="*70)

print("\n[1/4] Loading and cleaning data...")
df = pd.read_csv(os.path.join(data_dir, "threads_reviews.csv"))

std_stopwords = set(stopwords.words('english'))
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
    contractions = {
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "won't": "will not",
        "wouldn't": "would not",
        "can't": "cannot",
        "couldn't": "could not",
        "shouldn't": "should not",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "mightn't": "might not",
        "mustn't": "must not",
        "needn't": "need not",
        "shan't": "shall not",
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "i'll": "i will",
        "you'll": "you will",
        "he'll": "he will",
        "she'll": "she will",
        "we'll": "we will",
        "they'll": "they will",
        "i'd": "i would",
        "you'd": "you would",
        "he'd": "he would",
        "she'd": "she would",
        "we'd": "we would",
        "they'd": "they would"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text):
    if pd.isna(text):
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

df['clean_text'] = df['review_description'].apply(preprocess_text)
print(f"  Total samples: {len(df)}")

print("\n[2/4] Preparing labels...")
def map_rating_to_sentiment(rating):
    if pd.isna(rating):
        return 'neutral'
    rating = int(rating)
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

df['sentiment'] = df['rating'].apply(map_rating_to_sentiment)

distribusi = df['sentiment'].value_counts()
print("Label distribution:")
for label in ['positive', 'negative', 'neutral']:
    if label in distribusi:
        pct = (distribusi[label] / len(df)) * 100
        print(f"  {label:10}: {distribusi[label]:5d} samples ({pct:5.1f}%)")

print("\n[3/4] Tokenizing...")
max_words = 10000
max_length = 128

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df['clean_text'].fillna("").astype(str).tolist())

sequences = tokenizer.texts_to_sequences(df['clean_text'].fillna("").astype(str).tolist())
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

print(f"  Max words: {max_words}")
print(f"  Max sequence length: {max_length}")
print(f"  Vocabulary size: {len(tokenizer.word_index)}")

print("\n[4/4] Splitting data...")
y_text = df['sentiment'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

desired_order = ['negative', 'neutral', 'positive']
if len(label_encoder.classes_) == 3:
    label_mapping = {label: idx for idx, label in enumerate(desired_order)}
    y = np.array([label_mapping[label] for label in y_text])
    np.save(os.path.join(outputs_dir, "label_encoding.npy"), np.array(desired_order))
    print(f"Label order: {desired_order}")
else:
    np.save(os.path.join(outputs_dir, "label_encoding.npy"), label_encoder.classes_)

train_ratio = 0.8
np.random.seed(42)
indices = np.random.permutation(len(padded_sequences))

train_size = int(len(padded_sequences) * train_ratio)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train_sequences, X_test_sequences = padded_sequences[train_idx], padded_sequences[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

np.save(os.path.join(outputs_dir, "X_train_sequences.npy"), X_train_sequences)
np.save(os.path.join(outputs_dir, "X_test_sequences.npy"), X_test_sequences)
np.save(os.path.join(outputs_dir, "y_train.npy"), y_train)
np.save(os.path.join(outputs_dir, "y_test.npy"), y_test)
with open(os.path.join(outputs_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print(f"  Training: {len(X_train_sequences):,} samples")
print(f"  Testing:  {len(X_test_sequences):,} samples")

df.to_csv(os.path.join(data_dir, "threads_reviews_labeled.csv"), index=False, encoding='utf-8-sig')

print("\n" + "="*70)
print("PREPROCESSING COMPLETE")
print("="*70)
print("Output files:")
print("  - X_train_sequences.npy")
print("  - X_test_sequences.npy")
print("  - y_train.npy")
print("  - y_test.npy")
print("  - label_encoding.npy")
print("  - tokenizer.pkl")
print("  - threads_reviews_labeled.csv")
print("="*70)
