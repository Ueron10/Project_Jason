import pandas as pd
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")

# Download resource NLTK (hanya jika belum ada)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Load data
df = pd.read_csv(os.path.join(data_dir, "threads_reviews.csv"))

# Inisialisasi - Optimized for BERT (keep more context)
# BERT performs better with less aggressive preprocessing
std_stopwords = set(stopwords.words('english'))
# Keep sentiment-carrying words and remove only truly non-informative stopwords
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
    """Optimized preprocessing for BERT - keeps more context"""
    if pd.isna(text):
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

# Terapkan preprocessing
print("Preprocessing text data...")
df['clean_text'] = df['review_description'].apply(preprocess_text)

# Simpan hasil
df.to_csv(os.path.join(data_dir, "threads_reviews_clean.csv"), index=False, encoding='utf-8-sig')

print("Preprocessing selesai!")
print(f"  Total samples: {len(df)}")
print(f"  Output: threads_reviews_clean.csv")
print("\nSample:")
print(df[['review_description', 'clean_text', 'rating']].head())
