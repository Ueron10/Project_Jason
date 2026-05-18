import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_normalized_data():
    """Load normalized features for clustering"""
    features_normalized = pd.read_csv('../output/features_normalized.csv')
    print(f"Loaded {len(features_normalized)} normalized data points")
    return features_normalized

def load_optimal_k():
    """Load optimal number of clusters from elbow method"""
    try:
        with open('../output/optimal_k.txt', 'r') as f:
            optimal_k = int(f.read().strip())
        print(f"Loaded optimal K: {optimal_k}")
        return optimal_k
    except FileNotFoundError:
        print("Optimal K file not found. Using default K=3")
        return 3

def perform_kmeans_clustering(features, n_clusters):
    """Perform K-Means clustering"""
    print(f"\n=== PERFORMING K-MEANS CLUSTERING (K={n_clusters}) ===\n")
    
    # Initialize and fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette_avg = silhouette_score(features, cluster_labels)
    
    print(f"Clustering completed!")
    print(f"Inertia: {inertia:.2f}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    return kmeans, cluster_labels, cluster_centers, inertia, silhouette_avg

def save_clustering_results(kmeans, cluster_labels, inertia, silhouette_avg):
    """Save clustering results and model"""
    # Save cluster labels
    cluster_results = pd.DataFrame({
        'cluster': cluster_labels
    })
    cluster_results.to_csv('../output/cluster_labels.csv', index=False)
    
    # Save K-Means model
    with open('../output/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    # Save clustering metrics
    metrics = pd.DataFrame({
        'metric': ['inertia', 'silhouette_score'],
        'value': [inertia, silhouette_avg]
    })
    metrics.to_csv('../output/clustering_metrics.csv', index=False)
    
    print("Clustering results saved!")
    print("- cluster_labels.csv: Cluster labels for each data point")
    print("- kmeans_model.pkl: Trained K-Means model")
    print("- clustering_metrics.csv: Clustering performance metrics")

def main():
    """Main K-Means clustering pipeline"""
    print("=== K-MEANS CLUSTERING PIPELINE ===\n")
    
    # Load normalized data
    features_normalized = load_normalized_data()
    
    # Load optimal number of clusters
    optimal_k = load_optimal_k()
    
    # Perform K-Means clustering
    kmeans, cluster_labels, cluster_centers, inertia, silhouette_avg = perform_kmeans_clustering(
        features_normalized, optimal_k
    )
    
    # Save results
    save_clustering_results(kmeans, cluster_labels, inertia, silhouette_avg)
    
    print(f"\n=== K-MEANS CLUSTERING COMPLETE ===")
    print(f"Number of clusters: {optimal_k}")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    return kmeans, cluster_labels, cluster_centers

if __name__ == "__main__":
    kmeans, cluster_labels, cluster_centers = main()
