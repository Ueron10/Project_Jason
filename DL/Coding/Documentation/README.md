# Sentiment Analysis - IMDB Movie Reviews

Project Deep Learning untuk analisis sentimen menggunakan CNN + VADER.

**Dataset**: [IMDB 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## Panduan Setup Manual

### 1. Download Dataset
1. Download dari Kaggle: [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Extract file zip
3. Copy `IMDB Dataset.csv` ke folder `Data/`

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running Manual

### Step 1: Preprocessing
```bash
cd Scripts/1Preprocessing
python 2preprocesing.py    # Cleaning text
python 3labeling.py         # VADER labeling
python 4token_padding.py    # Tokenization
python 5splitting_data.py   # Split train/test
```

### Step 2: Training
```bash
cd ../2Traning
python 1_model_architecture.py   # Build model
python 2_model_training.py       # Training (target >80%)
python 4_model_save.py           # Save model
```

### Step 3: Prediction
```bash
cd ../3Prediction
python 5_model_prediction.py     # Input text untuk prediksi
```

### Alternatif: Streamlit App
```bash
streamlit run app.py              # Web UI untuk prediksi
```

---

## Struktur Folder

```
DL/
├── Data/               # Dataset (download manual)
├── Scripts/
│   ├── 1Preprocessing/ # Cleaning, labeling, tokenization
│   ├── 2Traning/       # Training pipeline
│   └── 3Prediction/    # Inference
├── Models/             # Saved models
└── Outputs/            # Tokenizer, metrics
```

---

## VADER Labeling

**Threshold** (di `3labeling.py`):
- `positive`: compound >= 0.50
- `negative`: compound <= -0.30
- `neutral`: lainnya

---

## Git Notes

File besar (CSV, model) sudah di `.gitignore`. Jangan commit:
- `Data/*.csv`
- `Models/*.keras`
- `Outputs/*.npy`
