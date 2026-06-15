# K-Means Property Segmentation - Jabodetabek

## Latar Belakang

### Penerapan Algoritma K-Means untuk Segmentasi Properti Berdasarkan Harga dan Luas Bangunan di Jabodetabek

Pertumbuhan sektor properti di wilayah Jabodetabek (Jakarta, Bogor, Depok, Tangerang, dan Bekasi) mengalami peningkatan yang signifikan seiring dengan perkembangan ekonomi dan urbanisasi. Wilayah ini menjadi pusat aktivitas ekonomi dan hunian yang menarik bagi berbagai kalangan masyarakat, mulai dari pekerja dengan pendapatan menengah hingga kelompok berpendapatan tinggi. Keragaman kebutuhan dan kemampuan finansial masyarakat menciptakan pasar properti yang kompleks dengan variasi harga dan spesifikasi bangunan yang sangat beragam.

Dalam konteks ini, pemahaman mengenai segmentasi pasar properti menjadi sangat penting bagi berbagai pihak, termasuk pengembang properti, agen real estate, investor, maupun pembeli potensial. Segmentasi yang akurat memungkinkan pengambilan keputusan yang lebih tepat sasaran dalam hal strategi pemasaran, penetapan harga, pengembangan proyek baru, dan investasi. Namun, dengan volume data properti yang besar dan variabel yang kompleks, pendekatan manual untuk segmentasi menjadi tidak efisien dan cenderung subjektif.

Algoritma K-Means clustering menawarkan solusi berbasis data yang efektif untuk mengatasi tantangan ini. K-Means adalah salah satu algoritma unsupervised learning yang paling populer dan sederhana untuk pengelompokan data. Algoritma ini bekerja dengan membagi data ke dalam K klaster berdasarkan kedekatan jarak antara titik data dengan pusat klaster (centroid). Keunggulan utama K-Means meliputi kemudahan implementasi, efisiensi komputasi, dan kemampuan untuk menangani dataset berukuran besar.

Dalam penelitian ini, algoritma K-Means diterapkan untuk segmentasi properti di Jabodetabek dengan fokus pada dua variabel utama: harga properti (dalam Rupiah) dan luas bangunan (dalam meter persegi). Pemilihan kedua variabel ini didasarkan pada pertimbangan bahwa harga dan ukuran bangunan merupakan faktor paling fundamental yang mempengaruhi nilai dan kelas properti. Harga mencerminkan nilai pasar dan daya beli target konsumen, sedangkan luas bangunan berkorelasi dengan kapasitas hunian dan kenyamanan.

Proses segmentasi dimulai dengan preprocessing data yang meliputi pembersihan data, penanganan missing values, dan penghapusan outlier menggunakan metode IQR (Interquartile Range). Selanjutnya, data dinormalisasi menggunakan StandardScaler untuk memastikan bahwa kedua variabel memiliki skala yang seimbang dalam perhitungan jarak. Penentuan jumlah klaster optimal dilakukan menggunakan metode elbow, yang mengevaluasi inersia (within-cluster sum of squares) untuk berbagai nilai K.

Hasil segmentasi diharapkan dapat mengidentifikasi pola-pola alami dalam data properti yang menggambarkan segmen-segmen pasar yang berbeda. Setiap klaster akan dikarakterisasi berdasarkan statistik deskriptif seperti rata-rata harga, rata-rata luas bangunan, dan jumlah properti dalam klaster tersebut. Berdasarkan karakteristik ini, properti dapat diklasifikasikan ke dalam kategori-kategori yang lebih mudah dipahami, seperti Budget, Standard, Premium, dan Luxury dengan sub-kategori berdasarkan ukuran.

Penerapan pendekatan ini memberikan beberapa manfaat signifikan. Pertama, memberikan pemahaman yang lebih objektif dan berbasis data mengenai struktur pasar properti di Jabodetabek. Kedua, membantu mengidentifikasi segmen pasar yang dominan dan segmen yang kurang terlayani, yang dapat menjadi peluang bagi pengembang properti. Ketiga, memfasilitasi strategi pemasaran yang lebih terarah dengan memahami profil dan preferensi setiap segmen. Keempat, memberikan panduan bagi pembeli potensial dalam memilih properti yang sesuai dengan budget dan kebutuhan ruang.

Secara keseluruhan, penerapan algoritma K-Means untuk segmentasi properti di Jabodetabek merupakan contoh konkret bagaimana machine learning dapat diaplikasikan untuk memecahkan masalah bisnis nyata. Pendekatan ini mengubah data mentah menjadi wawasan yang dapat ditindaklanjuti (actionable insights), mendukung pengambilan keputusan yang lebih informasi dan strategis dalam industri properti.

## Project Overview
Implementasi algoritma K-Means untuk melakukan segmentasi properti di wilayah Jabodetabek berdasarkan harga rumah dan luas bangunan.

## Dataset
- **Source**: Kaggle - Daftar Harga Rumah Jabodetabek
- **URL**: https://www.kaggle.com/datasets/nafisbarizki/daftar-harga-rumah-jabodetabek
- **Total Records**: 3,553 properties
- **Features**: 27 columns including price, building size, location, etc.
- **Focus Features**: `price_in_rp` dan `building_size_m2`
- **Data Type**: Secondary data (publik)

## Directory Structure
```
├── data/
│   └── dataset.csv                    # Original dataset
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── main.py                       # Main pipeline orchestrator
│   ├── preprocessing.py              # Data cleaning & normalization
│   ├── elbow_method.py               # Optimal cluster determination
│   ├── kmeans.py                     # K-Means clustering implementation
│   ├── visualization.py              # Result visualization
│   └── analysis.py                   # Cluster analysis & insights
├── output/                           # Generated files
│   ├── features_normalized.csv       # Normalized data for clustering
│   ├── features_clean.csv            # Cleaned data (original scale)
│   ├── scaler.pkl                    # StandardScaler object
│   ├── elbow_method_results.csv      # Elbow method metrics
│   ├── optimal_k.txt                 # Optimal number of clusters
│   ├── cluster_labels.csv            # Cluster assignments
│   ├── kmeans_model.pkl              # Trained K-Means model
│   ├── clustering_metrics.csv        # Performance metrics
│   ├── clustering_visualizations.png # Main visualizations
│   ├── cluster_summary.png           # Cluster summary plots
│   ├── detailed_cluster_analysis.csv # Detailed statistics
│   ├── analysis_insights.txt         # Key insights
│   └── analysis_report.txt           # Complete analysis report
├── docs/                             # Documentation
│   └── README.md                     # Detailed documentation
└── README.md                         # This file
```

## Pipeline Components

### 1. Preprocessing & Normalization
- Data loading and exploration
- Missing value handling
- Outlier removal using IQR method
- Feature selection (price, building_size)
- Data normalization using StandardScaler

### 2. Elbow Method
- Calculate inertia for K=2-10
- Determine optimal number of clusters
- Visualize elbow method analysis

### 3. K-Means Clustering
- Load optimal number of clusters
- Perform K-Means clustering
- Calculate performance metrics
- Save clustering model and results

### 4. Visualization
- Create comprehensive clustering visualizations
- Generate cluster summary plots
- Save all visualization outputs

### 5. Analysis
- Detailed cluster characteristics analysis
- Property category classification
- Generate insights and recommendations
- Create comprehensive analysis report

## Usage

### Run Complete Pipeline
```bash
cd src
python main.py
```

### Run Individual Components
```bash
cd src

# Preprocessing only
python preprocessing.py

# Elbow method only
python elbow_method.py

# K-Means clustering only
python kmeans.py

# Visualization only
python visualization.py

# Analysis only
python analysis.py
```

## Preprocessing Results
- **Initial Data**: 3,553 records
- **After Cleaning**: 3,551 records
- **After Outlier Removal**: 2,992 records
- **Final Features**: 2 normalized features (price, building_size)

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Property Categories
The analysis classifies properties into categories such as:
- Budget Small/Medium
- Standard Small/Medium/Large
- Premium Small/Medium/Large
- Luxury Compact/Standard/Estate

## Key Insights Generated
- Market segmentation analysis
- Price range distribution
- Size preferences
- Investment opportunities
- Market gaps and development potential

## Output Files Description

### Data Files
- `features_normalized.csv`: Normalized features for clustering
- `features_clean.csv`: Cleaned data with original scale
- `cluster_labels.csv`: Cluster assignment for each property

### Model Files
- `scaler.pkl`: StandardScaler for data normalization
- `kmeans_model.pkl`: Trained K-Means model

### Analysis Files
- `elbow_method_results.csv`: Metrics for different K values
- `optimal_k.txt`: Optimal number of clusters
- `clustering_metrics.csv`: Performance metrics
- `detailed_cluster_analysis.csv`: Comprehensive cluster statistics
- `analysis_insights.txt`: Key insights from clustering
- `analysis_report.txt`: Complete analysis report

### Visualization Files
- `clustering_visualizations.png`: Main clustering visualizations
- `elbow_method_analysis.png`: Elbow method analysis
