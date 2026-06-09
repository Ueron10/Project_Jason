import pandas as pd
import numpy as np
from transformers import BertTokenizer
import os
import pickle

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")
outputs_dir = os.path.join(project_root, "Outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(data_dir, "threads_reviews_labeled.csv"))
texts = df['clean_text'].fillna("").astype(str)

print(f"Tokenizing {len(texts)} texts with BERT...")

# BERT Tokenization
tokenizer_name = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
max_length = 256

# Tokenize with BERT
encodings = tokenizer(
    texts.tolist(),
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='np'
)

# Extract input_ids and attention_mask
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

print(f"\nTokenization selesai!")
print(f"  Model: {tokenizer_name}")
print(f"  Max sequence length: {max_length}")
print(f"  Data shape: {input_ids.shape}")

# Simpan
np.save(os.path.join(outputs_dir, "X_input_ids.npy"), input_ids)
np.save(os.path.join(outputs_dir, "X_attention_mask.npy"), attention_mask)
with open(os.path.join(outputs_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("\nOutput files:")
print("  - X_input_ids.npy")
print("  - X_attention_mask.npy")
print("  - tokenizer.pkl")
