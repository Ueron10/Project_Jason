import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
import os

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

print("="*70)
print("DATA AUGMENTATION FOR BETTER ACCURACY")
print("="*70)

# Load cleaned data
df = pd.read_csv(os.path.join(data_dir, "imdb_reviews_clean.csv"))
print(f"\nOriginal data size: {len(df)}")

def get_synonyms(word):
    """Get synonyms for a word using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name().replace('_', ' '))
    return list(synonyms)

def synonym_replacement(text, n=2):
    """Replace n random words with their synonyms"""
    words = text.split()
    new_words = words.copy()
    
    # Get words that can be replaced (have synonyms)
    random_word_list = list(set([word for word in words if get_synonyms(word)]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        if num_replaced >= n:
            break
        synonyms = get_synonyms(random_word)
        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
    
    return ' '.join(new_words)

def random_insertion(text, n=1):
    """Randomly insert n words into the text"""
    words = text.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word = random.choice(words)
        synonyms = get_synonyms(add_word)
        if len(synonyms) > 0:
            synonym = random.choice(synonyms)
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, synonym)
    
    return ' '.join(new_words)

def random_swap(text, n=1):
    """Randomly swap two words in the text n times"""
    words = text.split()
    new_words = words.copy()
    
    for _ in range(n):
        if len(new_words) >= 2:
            random_idx_1 = random.randint(0, len(new_words) - 1)
            random_idx_2 = random.randint(0, len(new_words) - 1)
            new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    
    return ' '.join(new_words)

def random_deletion(text, p=0.1):
    """Randomly delete words with probability p"""
    words = text.split()
    if len(words) == 1:
        return text
    
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    # If all words are deleted, return original
    if len(new_words) == 0:
        return text
    
    return ' '.join(new_words)

def augment_text(text, num_augmentations=2):
    """Apply multiple augmentation techniques"""
    augmented_texts = [text]  # Include original
    
    for _ in range(num_augmentations):
        # Randomly choose augmentation technique
        technique = random.choice(['synonym', 'insertion', 'swap', 'deletion'])
        
        if technique == 'synonym':
            augmented = synonym_replacement(text, n=2)
        elif technique == 'insertion':
            augmented = random_insertion(text, n=1)
        elif technique == 'swap':
            augmented = random_swap(text, n=1)
        else:  # deletion
            augmented = random_deletion(text, p=0.1)
        
        if augmented != text and len(augmented.split()) > 3:  # Ensure meaningful text
            augmented_texts.append(augmented)
    
    return augmented_texts

print("\n[1/3] Applying data augmentation...")
augmented_data = []

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Apply augmentation to a subset of data (to avoid too much noise)
augmentation_ratio = 0.3  # Augment 30% of the data
sample_indices = np.random.choice(len(df), int(len(df) * augmentation_ratio), replace=False)

for idx in range(len(df)):
    original_text = df.iloc[idx]['clean_text']
    sentiment = df.iloc[idx]['sentiment']
    
    # Add original
    augmented_data.append({'clean_text': original_text, 'sentiment': sentiment})
    
    # Add augmented versions for sampled indices
    if idx in sample_indices:
        augmented_texts = augment_text(original_text, num_augmentations=1)
        for aug_text in augmented_texts[1:]:  # Skip original (already added)
            augmented_data.append({'clean_text': aug_text, 'sentiment': sentiment})

print(f"  Augmented {len(sample_indices)} samples")
print(f"  Total samples after augmentation: {len(augmented_data)}")

# Create dataframe
augmented_df = pd.DataFrame(augmented_data)

# Shuffle the data
augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n[2/3] Saving augmented data...")
augmented_df.to_csv(os.path.join(data_dir, "imdb_reviews_augmented.csv"), index=False, encoding='utf-8-sig')

print("\n[3/3] Updating main dataset...")
# Also update the main file to use augmented data
augmented_df.to_csv(os.path.join(data_dir, "imdb_reviews_labeled.csv"), index=False, encoding='utf-8-sig')

print("\n" + "="*70)
print("DATA AUGMENTATION SUMMARY")
print("="*70)
print(f"Original samples: {len(df)}")
print(f"Augmented samples: {len(augmented_data)}")
print(f"Increase: {((len(augmented_data) - len(df)) / len(df) * 100):.1f}%")
print(f"\nAugmented data saved to:")
print(f"  - imdb_reviews_augmented.csv")
print(f"  - imdb_reviews_labeled.csv (updated)")
print("="*70)
