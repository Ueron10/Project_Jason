# K-Means Property Segmentation - Jabodetabek

## Project Overview
Implementasi algoritma K-Means untuk melakukan segmentasi properti di wilayah Jabodetabek berdasarkan harga rumah dan luas bangunan.

## Dataset
- **Source**: Rumah123 property listings
- **Total Records**: 3,553 properties
- **Features**: 27 columns including price, building size, location, etc.
- **Focus Features**: `price_in_rp` dan `building_size_m2`

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
- Calculate inertia and silhouette scores for K=2-10
- Determine optimal number of clusters
- Visualize elbow method and silhouette analysis

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
