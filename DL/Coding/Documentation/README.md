# Sentiment Analysis - Thread App Reviews

Project Deep Learning untuk analisis sentimen menggunakan BERT.

**Dataset**: Thread App Reviews (32,910 reviews from Google Play and App Store)

---

## Panduan Setup Manual

### 1. Download Dataset
1. Download Thread app reviews dataset
2. Copy `threads_reviews.csv` ke folder `Data/`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running Manual

### Quick Start (Full Pipeline)
```bash
cd Scripts
python main.py                   # Run complete pipeline: preprocessing + training
```

### Step 1: Preprocessing
```bash
cd Scripts/1Preprocessing
python 01_clean_text.py         # Cleaning text
python 02_prepare_labels.py     # Convert rating to sentiment
python 03_tokenize.py           # BERT tokenization
python 04_split_data.py         # Split train/test
python 05_data_augmentation.py  # Optional: Data augmentation
```

### Step 2: Training
```bash
cd ../2Training
python 01_build_and_train.py    # Build BERT model and train
```

### Flask Web App
```bash
cd ../..
python app.py                   # Web UI untuk prediksi (http://localhost:5000)
```

---

## Struktur Folder

```
DL/
├── Data/               # Dataset (download manual)
├── Scripts/
│   ├── 1Preprocessing/ # Cleaning, labeling, tokenization
│   ├── 2Training/      # Training pipeline
│   └── main.py         # Full pipeline runner
├── Models/             # Saved models
├── Outputs/            # Tokenizer, metrics
├── templates/         # Flask HTML templates
└── app.py             # Flask web app
```

---

## Rating to Sentiment Conversion

**Threshold** (di `02_prepare_labels.py`):
- `positive`: rating >= 4
- `negative`: rating <= 2
- `neutral`: rating = 3

**Total Classes**: 3 (negative, neutral, positive)

---

## Git Notes

File besar (CSV, model) sudah di `.gitignore`. Jangan commit:
- `Data/*.csv`
- `Models/*`
- `Outputs/*.npy` / `Outputs/*.pkl`

## Model Architecture

- **Framework**: PyTorch
- **Model**: BERT Large (bert-large-uncased)
- **Classes**: 3 (negative, neutral, positive)
- **Max Sequence Length**: 256
- **Optimizer**: AdamW with warmup scheduler
