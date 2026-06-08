import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_normalized_data():
    """Load normalized features for clustering"""
    features_normalized = pd.read_csv('../output/features_normalized.csv')
    print(f"Loaded {len(features_normalized)} normalized data points")
    return features_normalized

def calculate_inertia(features, max_clusters=10):
    """Calculate inertia for different number of clusters"""
    print("=== CALCULATING CLUSTER METRICS ===\n")
    
    inertias = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}")
    
    return k_range, inertias

def find_optimal_clusters(k_range, inertias):
    """Find optimal number of clusters using elbow method"""
    
    # Find elbow point using the "knee" method
    def find_elbow_point(inertias):
        """Find elbow point using the knee detection method"""
        n_points = len(inertias)
        all_coord = np.vstack((range(n_points), inertias)).T
        first_point = all_coord[0]
        line_vec = all_coord[-1] - first_point
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = all_coord - first_point
        proj_length = np.dot(vec_from_first, line_vec_norm)
        proj_length = proj_length.reshape(-1, 1)
        proj = proj_length * line_vec_norm
        perp = vec_from_first - proj
        dist = np.sqrt(np.sum(perp**2, axis=1))
        knee_idx = np.argmax(dist)
        return knee_idx + 2  # +2 because we start from K=2
    
    optimal_k_elbow = find_elbow_point(inertias)
    
    print(f"\nOptimal K (Elbow Method): {optimal_k_elbow}")
    
    return optimal_k_elbow

def visualize_elbow_analysis(k_range, inertias, optimal_k):
    """Create visualization for elbow method"""
    print("\n=== CREATING ELBOW METHOD VISUALIZATION ===\n")
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Elbow method plot
    ax1.plot(k_range, inertias, 'bo-', markersize=8, linewidth=2)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    plt.tight_layout()
    plt.savefig('../output/elbow_method_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def save_elbow_results(k_range, inertias, optimal_k):
    """Save elbow method results"""
    results_df = pd.DataFrame({
        'k': k_range,
        'inertia': inertias
    })
    
    results_df.to_csv('../output/elbow_method_results.csv', index=False)
    
    # Save optimal K
    with open('../output/optimal_k.txt', 'w') as f:
        f.write(str(optimal_k))
    
    print("Elbow method results saved!")
    print("- elbow_method_results.csv: Metrics for each K")
    print(f"- optimal_k.txt: Optimal K = {optimal_k}")

def main():
    """Main elbow method pipeline"""
    print("=== ELBOW METHOD PIPELINE ===\n")
    
    # Load normalized data
    features_normalized = load_normalized_data()
    
    # Calculate metrics for different K values
    k_range, inertias = calculate_inertia(features_normalized)
    
    # Find optimal number of clusters
    optimal_k = find_optimal_clusters(k_range, inertias)
    
    # Visualize results
    visualize_elbow_analysis(k_range, inertias, optimal_k)
    
    # Save results
    save_elbow_results(k_range, inertias, optimal_k)
    
    print(f"\n=== ELBOW METHOD COMPLETE ===")
    print(f"Optimal number of clusters: {optimal_k}")
    
    return optimal_k

if __name__ == "__main__":
    optimal_k = main()
