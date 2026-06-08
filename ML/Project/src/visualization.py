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
    
    # Set style for better appearance
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('K-Means Clustering Analysis - Property Segmentation', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Clustering results with original data (main plot)
    scatter1 = axes[0, 0].scatter(
        features_clean['building_size_m2'], 
        features_clean['price_in_rp'], 
        c=cluster_labels, 
        cmap='viridis', 
        alpha=0.5, 
        s=40,
        edgecolors='none'
    )
    axes[0, 0].set_xlabel('Building Size (m2)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Price (Rp)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Property Clustering Results', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Cluster', rotation=270, labelpad=20)
    axes[0, 0].ticklabel_format(style='plain', axis='y')
    
    # 2. Clusters with centers (highlighted)
    cluster_labels_unique = np.unique(cluster_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_labels_unique)))
    
    for i, (label, color) in enumerate(zip(cluster_labels_unique, colors)):
        cluster_data = features_clean[cluster_labels == label]
        axes[0, 1].scatter(
            cluster_data['building_size_m2'], 
            cluster_data['price_in_rp'], 
            c=[color], 
            alpha=0.5, 
            s=35, 
            edgecolors='none',
            label=f'Cluster {label}'
        )
    
    # Plot cluster centers
    axes[0, 1].scatter(
        cluster_centers_original[:, 1], 
        cluster_centers_original[:, 0], 
        c='red', 
        marker='X', 
        s=300, 
        linewidths=3, 
        edgecolors='darkred',
        label='Cluster Centers',
        zorder=10
    )
    axes[0, 1].set_xlabel('Building Size (m2)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Price (Rp)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Clusters with Centers', fontsize=12, fontweight='bold')
    axes[0, 1].legend(loc='upper right', fontsize=9, framealpha=0.9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='plain', axis='y')
    
    # 3. Cluster distribution (bar chart with better styling)
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    colors_bar = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
    bars = axes[0, 2].bar(cluster_counts.index, cluster_counts.values, color=colors_bar, edgecolor='black', linewidth=1.5)
    axes[0, 2].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[0, 2].set_ylabel('Number of Properties', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('Cluster Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks(cluster_counts.index)
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    for i, (bar, count) in enumerate(zip(bars, cluster_counts.values)):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + max(cluster_counts.values)*0.02, 
                       f'{count}\n({count/len(cluster_labels)*100:.1f}%)', 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Price distribution by cluster (boxplot for better visualization)
    cluster_data_with_labels = features_clean.copy()
    cluster_data_with_labels['cluster'] = cluster_labels
    
    price_data = [cluster_data_with_labels[cluster_data_with_labels['cluster'] == i]['price_in_rp'].values 
                  for i in cluster_labels_unique]
    bp = axes[1, 0].boxplot(price_data, labels=[f'C{i}' for i in cluster_labels_unique], 
                            patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 0].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Price (Rp)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Price Distribution by Cluster', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    
    # 5. Building size distribution by cluster (boxplot)
    size_data = [cluster_data_with_labels[cluster_data_with_labels['cluster'] == i]['building_size_m2'].values 
                 for i in cluster_labels_unique]
    bp2 = axes[1, 1].boxplot(size_data, labels=[f'C{i}' for i in cluster_labels_unique], 
                             patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 1].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Building Size (m2)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Building Size Distribution by Cluster', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Cluster pie chart (percentage distribution)
    cluster_counts_sorted = cluster_counts.sort_values(ascending=False)
    colors_pie = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts_sorted)))
    wedges, texts, autotexts = axes[1, 2].pie(
        cluster_counts_sorted.values, 
        labels=[f'Cluster {i}' for i in cluster_counts_sorted.index],
        autopct='%1.1f%%', 
        startangle=90,
        colors=colors_pie,
        explode=[0.05] * len(cluster_counts_sorted),
        shadow=True,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    axes[1, 2].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('../output/clustering_visualizations.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
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
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cluster Summary Statistics', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Average price by cluster with better styling
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_stats)))
    bars1 = axes[0, 0].bar(cluster_stats.index, cluster_stats['price_mean'], color=colors, 
                          edgecolor='black', linewidth=1.5)
    axes[0, 0].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Average Price (Rp)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Average Price by Cluster', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].ticklabel_format(style='plain', axis='y')
    
    for i, (bar, price) in enumerate(zip(bars1, cluster_stats['price_mean'])):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(cluster_stats['price_mean'])*0.02, 
                       f'{price/1e9:.2f}B', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Average building size by cluster
    bars2 = axes[0, 1].bar(cluster_stats.index, cluster_stats['size_mean'], color=colors, 
                          edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlabel('Cluster', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Average Building Size (m2)', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Average Building Size by Cluster', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for i, (bar, size) in enumerate(zip(bars2, cluster_stats['size_mean'])):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + max(cluster_stats['size_mean'])*0.02, 
                       f'{size:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Cluster centers scatter plot with better visualization
    axes[1, 0].scatter(cluster_centers_original[:, 1], cluster_centers_original[:, 0], 
                       c=colors, s=400, marker='o', edgecolors='black', linewidths=2, alpha=0.8)
    axes[1, 0].set_xlabel('Building Size (m2)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Price (Rp)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Cluster Centers', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].ticklabel_format(style='plain', axis='y')
    
    for i, (size, price) in enumerate(cluster_centers_original):
        axes[1, 0].annotate(f'C{i}\n({price/1e9:.2f}B)', (size, price), 
                          xytext=(0, 15), textcoords='offset points', 
                          fontsize=10, fontweight='bold', ha='center',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 4. Cluster size distribution with better pie chart
    wedges, texts, autotexts = axes[1, 1].pie(
        cluster_stats['count'], 
        labels=[f'Cluster {i}' for i in cluster_stats.index],
        autopct='%1.1f%%', 
        startangle=90,
        colors=colors,
        explode=[0.05] * len(cluster_stats),
        shadow=True,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    axes[1, 1].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('../output/cluster_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
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
