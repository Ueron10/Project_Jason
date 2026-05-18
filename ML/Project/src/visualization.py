import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_clustering_data():
    """Load all necessary data for visualization"""
    # Load normalized features
    features_normalized = pd.read_csv('../output/features_normalized.csv')
    
    # Load clean features (original scale)
    features_clean = pd.read_csv('../output/features_clean.csv')
    
    # Load cluster labels
    cluster_labels_df = pd.read_csv('../output/cluster_labels.csv')
    cluster_labels = cluster_labels_df['cluster'].values
    
    # Load K-Means model
    with open('../output/kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    
    # Load scaler
    with open('../output/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Loaded data for visualization: {len(features_normalized)} data points")
    return features_normalized, features_clean, cluster_labels, kmeans, scaler

def create_clustering_visualizations(features_normalized, features_clean, cluster_labels, kmeans, scaler):
    """Create comprehensive visualizations of clustering results"""
    print("\n=== CREATING CLUSTERING VISUALIZATIONS ===\n")
    
    # Transform cluster centers back to original scale
    cluster_centers_normalized = kmeans.cluster_centers_
    cluster_centers_original = scaler.inverse_transform(cluster_centers_normalized)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Clustering results with normalized data
    scatter1 = axes[0, 0].scatter(
        features_normalized.iloc[:, 1], 
        features_normalized.iloc[:, 0], 
        c=cluster_labels, 
        cmap='viridis', 
        alpha=0.6, 
        s=50
    )
    axes[0, 0].set_xlabel('Building Size (Normalized)')
    axes[0, 0].set_ylabel('Price (Normalized)')
    axes[0, 0].set_title('K-Means Clustering Results (Normalized)')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # 2. Clustering results with original data
    scatter2 = axes[0, 1].scatter(
        features_clean['building_size_m2'], 
        features_clean['price_in_rp'], 
        c=cluster_labels, 
        cmap='viridis', 
        alpha=0.6, 
        s=50
    )
    axes[0, 1].set_xlabel('Building Size (m²)')
    axes[0, 1].set_ylabel('Price (Rp)')
    axes[0, 1].set_title('K-Means Clustering Results (Original Scale)')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 3. Clusters with centers
    cluster_labels_unique = np.unique(cluster_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_labels_unique)))
    
    for i, (label, color) in enumerate(zip(cluster_labels_unique, colors)):
        cluster_data = features_clean[cluster_labels == label]
        axes[0, 2].scatter(
            cluster_data['building_size_m2'], 
            cluster_data['price_in_rp'], 
            c=[color], 
            alpha=0.6, 
            s=30, 
            label=f'Cluster {label}'
        )
    
    # Plot cluster centers
    axes[0, 2].scatter(
        cluster_centers_original[:, 1], 
        cluster_centers_original[:, 0], 
        c='red', 
        marker='x', 
        s=200, 
        linewidths=3, 
        label='Cluster Centers'
    )
    axes[0, 2].set_xlabel('Building Size (m²)')
    axes[0, 2].set_ylabel('Price (Rp)')
    axes[0, 2].set_title('Clusters with Centers')
    axes[0, 2].legend()
    
    # 4. Cluster distribution
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    axes[1, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Number of Properties')
    axes[1, 0].set_title('Cluster Distribution')
    axes[1, 0].set_xticks(cluster_counts.index)
    
    for i, count in enumerate(cluster_counts.values):
        axes[1, 0].text(i, count + 10, str(count), ha='center', va='bottom')
    
    # 5. Price distribution by cluster
    cluster_data_with_labels = features_clean.copy()
    cluster_data_with_labels['cluster'] = cluster_labels
    
    for i in cluster_labels_unique:
        cluster_price_data = cluster_data_with_labels[cluster_data_with_labels['cluster'] == i]['price_in_rp']
        axes[1, 1].hist(cluster_price_data, bins=20, alpha=0.7, label=f'Cluster {i}')
    
    axes[1, 1].set_xlabel('Price (Rp)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Price Distribution by Cluster')
    axes[1, 1].legend()
    axes[1, 1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # 6. Building size distribution by cluster
    for i in cluster_labels_unique:
        cluster_size_data = cluster_data_with_labels[cluster_data_with_labels['cluster'] == i]['building_size_m2']
        axes[1, 2].hist(cluster_size_data, bins=20, alpha=0.7, label=f'Cluster {i}')
    
    axes[1, 2].set_xlabel('Building Size (m²)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Building Size Distribution by Cluster')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('../output/clustering_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_cluster_summary_plot(features_clean, cluster_labels, cluster_centers_original):
    """Create a summary plot showing cluster characteristics"""
    print("\n=== CREATING CLUSTER SUMMARY PLOT ===\n")
    
    # Create DataFrame with cluster labels
    cluster_data = features_clean.copy()
    cluster_data['cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = cluster_data.groupby('cluster').agg({
        'price_in_rp': ['mean', 'median', 'std'],
        'building_size_m2': ['mean', 'median', 'std'],
        'cluster': 'count'
    }).round(2)
    
    cluster_stats.columns = ['price_mean', 'price_median', 'price_std', 
                           'size_mean', 'size_median', 'size_std', 'count']
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Average price by cluster
    axes[0, 0].bar(cluster_stats.index, cluster_stats['price_mean'], color='lightcoral')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Average Price (Rp)')
    axes[0, 0].set_title('Average Price by Cluster')
    axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    for i, price in enumerate(cluster_stats['price_mean']):
        axes[0, 0].text(i, price, f'{price/1e9:.1f}B', ha='center', va='bottom')
    
    # 2. Average building size by cluster
    axes[0, 1].bar(cluster_stats.index, cluster_stats['size_mean'], color='lightblue')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Average Building Size (m²)')
    axes[0, 1].set_title('Average Building Size by Cluster')
    
    for i, size in enumerate(cluster_stats['size_mean']):
        axes[0, 1].text(i, size, f'{size:.0f}', ha='center', va='bottom')
    
    # 3. Cluster centers scatter plot
    axes[1, 0].scatter(cluster_centers_original[:, 1], cluster_centers_original[:, 0], 
                       c='red', s=200, marker='o')
    axes[1, 0].set_xlabel('Building Size (m²)')
    axes[1, 0].set_ylabel('Price (Rp)')
    axes[1, 0].set_title('Cluster Centers')
    axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    for i, (size, price) in enumerate(cluster_centers_original):
        axes[1, 0].annotate(f'C{i}', (size, price), xytext=(5, 5), 
                           textcoords='offset points', fontsize=12, fontweight='bold')
    
    # 4. Cluster size distribution
    axes[1, 1].pie(cluster_stats['count'], labels=[f'Cluster {i}' for i in cluster_stats.index],
                   autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    plt.savefig('../output/cluster_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig, cluster_stats

def save_visualization_summary(cluster_stats):
    """Save visualization summary and statistics"""
    # Save cluster statistics
    cluster_stats.to_csv('../output/cluster_summary_statistics.csv')
    
    # Create visualization summary
    summary = {
        'total_properties': len(cluster_stats),
        'total_clusters': len(cluster_stats),
        'visualization_files': [
            'clustering_visualizations.png',
            'cluster_summary.png',
            'elbow_method_analysis.png'
        ]
    }
    
    with open('../output/visualization_summary.txt', 'w') as f:
        f.write("CLUSTERING VISUALIZATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total Properties: {summary['total_properties']}\n")
        f.write(f"Total Clusters: {summary['total_clusters']}\n\n")
        f.write("Generated Visualizations:\n")
        for viz_file in summary['visualization_files']:
            f.write(f"- {viz_file}\n")
    
    print("Visualization summary saved!")
    print("- cluster_summary_statistics.csv: Detailed cluster statistics")
    print("- visualization_summary.txt: Summary of all visualizations")

def main():
    """Main visualization pipeline"""
    print("=== VISUALIZATION PIPELINE ===\n")
    
    # Load clustering data
    features_normalized, features_clean, cluster_labels, kmeans, scaler = load_clustering_data()
    
    # Create main clustering visualizations
    create_clustering_visualizations(features_normalized, features_clean, cluster_labels, kmeans, scaler)
    
    # Create cluster summary plot
    cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    fig_summary, cluster_stats = create_cluster_summary_plot(
        features_clean, cluster_labels, cluster_centers_original
    )
    
    # Save visualization summary
    save_visualization_summary(cluster_stats)
    
    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"Generated visualizations for {len(np.unique(cluster_labels))} clusters")
    
    return cluster_stats

if __name__ == "__main__":
    cluster_stats = main()
