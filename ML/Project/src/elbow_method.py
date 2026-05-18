import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_normalized_data():
    """Load normalized features for clustering"""
    features_normalized = pd.read_csv('../output/features_normalized.csv')
    print(f"Loaded {len(features_normalized)} normalized data points")
    return features_normalized

def calculate_inertia_and_silhouette(features, max_clusters=10):
    """Calculate inertia and silhouette scores for different number of clusters"""
    print("=== CALCULATING CLUSTER METRICS ===\n")
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(features, kmeans.labels_))
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_score(features, kmeans.labels_):.4f}")
    
    return k_range, inertias, silhouette_scores

def find_optimal_clusters(k_range, inertias, silhouette_scores):
    """Find optimal number of clusters using elbow method and silhouette score"""
    
    # Find optimal K based on silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    
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
    
    print(f"\nOptimal K (Silhouette Score): {optimal_k_silhouette}")
    print(f"Optimal K (Elbow Method): {optimal_k_elbow}")
    
    # Choose the final optimal K (prioritize silhouette score if close)
    if abs(optimal_k_silhouette - optimal_k_elbow) <= 1:
        optimal_k = optimal_k_silhouette
        print(f"Final Optimal K: {optimal_k} (Silhouette score chosen)")
    else:
        optimal_k = optimal_k_silhouette
        print(f"Final Optimal K: {optimal_k} (Silhouette score prioritized)")
    
    return optimal_k, optimal_k_silhouette, optimal_k_elbow

def visualize_elbow_analysis(k_range, inertias, silhouette_scores, optimal_k):
    """Create visualization for elbow method and silhouette analysis"""
    print("\n=== CREATING ELBOW METHOD VISUALIZATION ===\n")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow method plot
    ax1.plot(k_range, inertias, 'bo-', markersize=8, linewidth=2)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Silhouette score plot
    ax2.plot(k_range, silhouette_scores, 'ro-', markersize=8, linewidth=2, color='red')
    ax2.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal K={optimal_k}')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('../output/elbow_method_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def save_elbow_results(k_range, inertias, silhouette_scores, optimal_k):
    """Save elbow method results"""
    results_df = pd.DataFrame({
        'k': k_range,
        'inertia': inertias,
        'silhouette_score': silhouette_scores
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
    k_range, inertias, silhouette_scores = calculate_inertia_and_silhouette(features_normalized)
    
    # Find optimal number of clusters
    optimal_k, optimal_k_silhouette, optimal_k_elbow = find_optimal_clusters(
        k_range, inertias, silhouette_scores
    )
    
    # Visualize results
    visualize_elbow_analysis(k_range, inertias, silhouette_scores, optimal_k)
    
    # Save results
    save_elbow_results(k_range, inertias, silhouette_scores, optimal_k)
    
    print(f"\n=== ELBOW METHOD COMPLETE ===")
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Silhouette Score: {max(silhouette_scores):.4f}")
    
    return optimal_k

if __name__ == "__main__":
    optimal_k = main()
