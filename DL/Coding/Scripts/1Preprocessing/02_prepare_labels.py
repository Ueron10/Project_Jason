import pandas as pd
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")

# Load data yang sudah di-preprocess
df = pd.read_csv(os.path.join(data_dir, "threads_reviews_clean.csv"))

# Mapping rating ke sentiment: 1-2=negatif, 3=netral, 4-5=positif
def map_rating_to_sentiment(rating):
    if pd.isna(rating):
        return 'neutral'
    rating = int(rating)
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:  # rating >= 4
        return 'positive'

df['sentiment'] = df['rating'].apply(map_rating_to_sentiment)

# Cek distribusi label
print("Distribusi Label:")
distribusi = df['sentiment'].value_counts()
print(distribusi)
print(f"\nTotal data: {len(df)}")

# Hitung persentase
for label in ['positive', 'negative', 'neutral']:
    if label in distribusi:
        pct = (distribusi[label] / len(df)) * 100
        print(f"  {label:10}: {distribusi[label]:5d} samples ({pct:5.1f}%)")

# Simpan hasil
df.to_csv(os.path.join(data_dir, "threads_reviews_labeled.csv"), index=False, encoding='utf-8-sig')

print("\nLabel preparation selesai!")
print("  Output: threads_reviews_labeled.csv")
