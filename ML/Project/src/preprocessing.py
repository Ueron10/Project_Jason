import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

def load_data(file_path='../data/dataset.csv'):
    """Load dataset from CSV file"""
    return pd.read_csv(file_path)

def explore_data(df):
    """Explore basic statistics of the dataset"""
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics for Price and Building Size:")
    print(df[['price_in_rp', 'building_size_m2']].describe())
    
    print(f"\nPrice Range: {df.price_in_rp.min():,.0f} - {df.price_in_rp.max():,.0f}")
    print(f"Building Size Range: {df.building_size_m2.min():,.0f} - {df.building_size_m2.max():,.0f}")
    
    return df

def clean_data(df):
    """Clean dataset by handling missing values and outliers"""
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Check missing values for key features
    print("Missing values before cleaning:")
    print(df_clean[['price_in_rp', 'building_size_m2']].isnull().sum())
    
    # Remove rows with missing values in key features
    df_clean = df_clean.dropna(subset=['price_in_rp', 'building_size_m2'])
    
    # Remove zero or negative values
    df_clean = df_clean[(df_clean['price_in_rp'] > 0) & (df_clean['building_size_m2'] > 0)]
    
    print(f"\nData points after removing missing/invalid values: {len(df_clean)}")
    
    return df_clean

def handle_outliers(df):
    """Handle outliers using IQR method"""
    df_outlier = df.copy()
    
    # Function to remove outliers using IQR
    def remove_outliers_iqr(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Remove outliers for price and building size
    df_outlier = remove_outliers_iqr(df_outlier, 'price_in_rp')
    df_outlier = remove_outliers_iqr(df_outlier, 'building_size_m2')
    
    print(f"Data points after removing outliers: {len(df_outlier)}")
    
    return df_outlier

def select_features(df):
    """Select relevant features for clustering"""
    # Select only price and building size for clustering
    features = df[['price_in_rp', 'building_size_m2']].copy()
    
    print("Selected features for clustering:")
    print(features.head())
    
    return features

def normalize_data(df_features):
    """Normalize data using StandardScaler"""
    scaler = StandardScaler()
    
    # Fit and transform the features
    features_normalized = scaler.fit_transform(df_features)
    
    # Convert back to DataFrame for easier handling
    features_normalized_df = pd.DataFrame(
        features_normalized, 
        columns=['price_normalized', 'building_size_normalized'],
        index=df_features.index
    )
    
    print("Normalized features:")
    print(features_normalized_df.head())
    
    print(f"\nNormalization statistics:")
    print(f"Price - Mean: {features_normalized[:, 0].mean():.4f}, Std: {features_normalized[:, 0].std():.4f}")
    print(f"Building Size - Mean: {features_normalized[:, 1].mean():.4f}, Std: {features_normalized[:, 1].std():.4f}")
    
    return features_normalized_df, scaler

def save_processed_data(df_normalized, df_clean, scaler):
    """Save processed data and scaler for later use"""
    # Save normalized features
    df_normalized.to_csv('../output/features_normalized.csv', index=False)
    
    # Save cleaned original features (with original scale)
    df_clean[['price_in_rp', 'building_size_m2']].to_csv('../output/features_clean.csv', index=False)
    
    # Save scaler object
    with open('../output/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Processed data and scaler saved successfully!")

def main():
    """Main preprocessing pipeline"""
    print("=== PREPROCESSING & NORMALIZATION ===\n")
    
    # Load data
    df = load_data()
    
    # Explore data
    df_explored = explore_data(df)
    
    # Clean data
    df_clean = clean_data(df_explored)
    
    # Handle outliers
    df_no_outliers = handle_outliers(df_clean)
    
    # Select features
    features = select_features(df_no_outliers)
    
    # Normalize data
    features_normalized, scaler = normalize_data(features)
    
    # Save processed data
    save_processed_data(features_normalized, df_no_outliers, scaler)
    
    print(f"\n=== PREPROCESSING COMPLETE ===")
    print(f"Final dataset shape: {features_normalized.shape}")
    print(f"Ready for K-Means clustering!")
    
    return features_normalized, scaler

if __name__ == "__main__":
    features_normalized, scaler = main()
