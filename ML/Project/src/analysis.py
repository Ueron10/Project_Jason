import pandas as pd
import numpy as np
import pickle

def load_analysis_data():
    """Load all data needed for cluster analysis"""
    # Load clean features
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
    
    # Load clustering metrics
    clustering_metrics = pd.read_csv('../output/clustering_metrics.csv')
    
    print(f"Loaded data for analysis: {len(features_clean)} data points")
    return features_clean, cluster_labels, kmeans, scaler, clustering_metrics

def analyze_cluster_characteristics(features_clean, cluster_labels):
    """Analyze detailed characteristics of each cluster"""
    print("\n=== ANALYZING CLUSTER CHARACTERISTICS ===\n")
    
    # Combine features with cluster labels
    cluster_data = features_clean.copy()
    cluster_data['cluster'] = cluster_labels
    
    # Calculate comprehensive statistics for each cluster
    cluster_analysis = []
    
    for cluster_id in sorted(cluster_data['cluster'].unique()):
        cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
        
        analysis = {
            'cluster_id': cluster_id,
            'count': len(cluster_subset),
            'percentage': (len(cluster_subset) / len(cluster_data)) * 100,
            
            # Price statistics
            'price_mean': cluster_subset['price_in_rp'].mean(),
            'price_median': cluster_subset['price_in_rp'].median(),
            'price_std': cluster_subset['price_in_rp'].std(),
            'price_min': cluster_subset['price_in_rp'].min(),
            'price_max': cluster_subset['price_in_rp'].max(),
            'price_range': cluster_subset['price_in_rp'].max() - cluster_subset['price_in_rp'].min(),
            
            # Building size statistics
            'size_mean': cluster_subset['building_size_m2'].mean(),
            'size_median': cluster_subset['building_size_m2'].median(),
            'size_std': cluster_subset['building_size_m2'].std(),
            'size_min': cluster_subset['building_size_m2'].min(),
            'size_max': cluster_subset['building_size_m2'].max(),
            'size_range': cluster_subset['building_size_m2'].max() - cluster_subset['building_size_m2'].min(),
            
            # Price per square meter
            'price_per_m2_mean': (cluster_subset['price_in_rp'] / cluster_subset['building_size_m2']).mean(),
            'price_per_m2_median': (cluster_subset['price_in_rp'] / cluster_subset['building_size_m2']).median(),
        }
        
        cluster_analysis.append(analysis)
    
    # Convert to DataFrame
    cluster_analysis_df = pd.DataFrame(cluster_analysis)
    
    return cluster_analysis_df, cluster_data

def classify_property_categories(cluster_analysis_df):
    """Classify clusters into property categories based on characteristics"""
    print("\n=== CLASSIFYING PROPERTY CATEGORIES ===\n")
    
    categories = []
    
    for _, cluster in cluster_analysis_df.iterrows():
        price_mean = cluster['price_mean']
        size_mean = cluster['size_mean']
        price_per_m2 = cluster['price_per_m2_mean']
        
        # Classification logic based on price and size
        if price_mean < 1e9:  # Less than 1 billion
            if size_mean < 100:
                category = "Budget Small"
            else:
                category = "Budget Medium"
        elif price_mean < 3e9:  # 1-3 billion
            if size_mean < 150:
                category = "Standard Small"
            elif size_mean < 250:
                category = "Standard Medium"
            else:
                category = "Standard Large"
        elif price_mean < 7e9:  # 3-7 billion
            if size_mean < 200:
                category = "Premium Small"
            elif size_mean < 350:
                category = "Premium Medium"
            else:
                category = "Premium Large"
        else:  # More than 7 billion
            if size_mean < 300:
                category = "Luxury Compact"
            elif size_mean < 500:
                category = "Luxury Standard"
            else:
                category = "Luxury Estate"
        
        categories.append(category)
    
    cluster_analysis_df['category'] = categories
    
    return cluster_analysis_df

def generate_insights(cluster_analysis_df, clustering_metrics):
    """Generate insights and recommendations from clustering results"""
    print("\n=== GENERATING INSIGHTS ===\n")
    
    insights = []
    
    # Overall clustering performance
    silhouette_score = clustering_metrics[clustering_metrics['metric'] == 'silhouette_score']['value'].iloc[0]
    inertia = clustering_metrics[clustering_metrics['metric'] == 'inertia']['value'].iloc[0]
    
    insights.append(f"Clustering Performance:")
    insights.append(f"- Silhouette Score: {silhouette_score:.4f}")
    insights.append(f"- Inertia: {inertia:.2f}")
    insights.append(f"- Number of Clusters: {len(cluster_analysis_df)}")
    
    # Market segmentation insights
    insights.append(f"\nMarket Segmentation Insights:")
    
    # Find dominant segments
    dominant_cluster = cluster_analysis_df.loc[cluster_analysis_df['count'].idxmax()]
    insights.append(f"- Largest Segment: {dominant_cluster['category']} ({dominant_cluster['count']} properties, {dominant_cluster['percentage']:.1f}%)")
    
    # Price range analysis
    price_ranges = cluster_analysis_df[['price_mean', 'category']].sort_values('price_mean')
    insights.append(f"\nPrice Range Analysis:")
    for _, row in price_ranges.iterrows():
        insights.append(f"- {row['category']}: Avg Price Rp {row['price_mean']/1e9:.2f}B")
    
    # Size preferences
    size_analysis = cluster_analysis_df[['size_mean', 'category']].sort_values('size_mean')
    insights.append(f"\nSize Preferences:")
    for _, row in size_analysis.iterrows():
        insights.append(f"- {row['category']}: Avg Size {row['size_mean']:.0f} m²")
    
    # Market gaps and opportunities
    insights.append(f"\nMarket Insights:")
    
    # Check for underrepresented segments
    small_segments = cluster_analysis_df[cluster_analysis_df['percentage'] < 10]
    if len(small_segments) > 0:
        insights.append(f"- Underrepresented segments (potential opportunities):")
        for _, row in small_segments.iterrows():
            insights.append(f"  * {row['category']}: {row['percentage']:.1f}% of market")
    
    # Price per square meter analysis
    avg_price_per_m2 = cluster_analysis_df['price_per_m2_mean'].mean()
    expensive_segments = cluster_analysis_df[cluster_analysis_df['price_per_m2_mean'] > avg_price_per_m2 * 1.5]
    if len(expensive_segments) > 0:
        insights.append(f"- High-value segments (price/m² > 1.5x average):")
        for _, row in expensive_segments.iterrows():
            insights.append(f"  * {row['category']}: Rp {row['price_per_m2_mean']/1e6:.1f}M/m²")
    
    return insights

def create_detailed_report(cluster_analysis_df, insights, clustering_metrics):
    """Create a detailed analysis report"""
    print("\n=== CREATING DETAILED REPORT ===\n")
    
    report = []
    report.append("PROPERTY SEGMENTATION ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"Total Properties Analyzed: {cluster_analysis_df['count'].sum()}")
    report.append(f"Number of Segments Identified: {len(cluster_analysis_df)}")
    silhouette_score = clustering_metrics[clustering_metrics['metric'] == 'silhouette_score']['value'].iloc[0]
    report.append(f"Clustering Quality (Silhouette Score): {silhouette_score:.4f}")
    report.append("")
    
    # Cluster Details
    report.append("DETAILED CLUSTER ANALYSIS")
    report.append("-" * 30)
    report.append("")
    
    for _, cluster in cluster_analysis_df.iterrows():
        report.append(f"Cluster {cluster['cluster_id']}: {cluster['category']}")
        report.append(f"  Properties: {cluster['count']} ({cluster['percentage']:.1f}%)")
        report.append(f"  Price Range: Rp {cluster['price_min']/1e9:.2f}B - Rp {cluster['price_max']/1e9:.2f}B")
        report.append(f"  Average Price: Rp {cluster['price_mean']/1e9:.2f}B")
        report.append(f"  Size Range: {cluster['size_min']:.0f} - {cluster['size_max']:.0f} m²")
        report.append(f"  Average Size: {cluster['size_mean']:.0f} m²")
        report.append(f"  Price per m²: Rp {cluster['price_per_m2_mean']/1e6:.1f}M")
        report.append("")
    
    # Insights
    report.append("KEY INSIGHTS")
    report.append("-" * 15)
    for insight in insights:
        report.append(insight)
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 15)
    
    # Generate recommendations based on analysis
    recommendations = []
    
    # Investment recommendations
    premium_segments = cluster_analysis_df[cluster_analysis_df['price_mean'] > cluster_analysis_df['price_mean'].median()]
    if len(premium_segments) > 0:
        recommendations.append("Investment Opportunities:")
        for _, row in premium_segments.iterrows():
            recommendations.append(f"  - Focus on {row['category']} segment for high-value properties")
    
    # Market development recommendations
    small_segments = cluster_analysis_df[cluster_analysis_df['percentage'] < 10]
    if len(small_segments) > 0:
        recommendations.append("Market Development:")
        recommendations.append("  - Consider developing properties in underrepresented segments")
    
    # Pricing strategy recommendations
    recommendations.append("Pricing Strategy:")
    avg_price_per_m2 = cluster_analysis_df['price_per_m2_mean'].mean()
    recommendations.append(f"  - Market average price per m²: Rp {avg_price_per_m2/1e6:.1f}M")
    recommendations.append("  - Use cluster-specific pricing for targeted marketing")
    
    for rec in recommendations:
        report.append(rec)
    
    return report

def save_analysis_results(cluster_analysis_df, insights, report):
    """Save all analysis results"""
    print("\n=== SAVING ANALYSIS RESULTS ===\n")
    
    # Save detailed cluster analysis
    cluster_analysis_df.to_csv('../output/detailed_cluster_analysis.csv', index=False)
    
    # Save insights
    with open('../output/analysis_insights.txt', 'w') as f:
        f.write("CLUSTERING ANALYSIS INSIGHTS\n")
        f.write("=" * 40 + "\n\n")
        for insight in insights:
            f.write(insight + "\n")
    
    # Save detailed report
    with open('../output/analysis_report.txt', 'w') as f:
        for line in report:
            f.write(line + "\n")
    
    print("Analysis results saved!")
    print("- detailed_cluster_analysis.csv: Comprehensive cluster statistics")
    print("- analysis_insights.txt: Key insights from clustering")
    print("- analysis_report.txt: Detailed analysis report")

def main():
    """Main analysis pipeline"""
    print("=== CLUSTERING ANALYSIS PIPELINE ===\n")
    
    # Load data
    features_clean, cluster_labels, kmeans, scaler, clustering_metrics = load_analysis_data()
    
    # Analyze cluster characteristics
    cluster_analysis_df, cluster_data = analyze_cluster_characteristics(features_clean, cluster_labels)
    
    # Classify property categories
    cluster_analysis_df = classify_property_categories(cluster_analysis_df)
    
    # Generate insights
    insights = generate_insights(cluster_analysis_df, clustering_metrics)
    
    # Create detailed report
    report = create_detailed_report(cluster_analysis_df, insights, clustering_metrics)
    
    # Save results
    save_analysis_results(cluster_analysis_df, insights, report)
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Analyzed {len(cluster_analysis_df)} property segments")
    print(f"Generated {len(insights)} key insights")
    
    return cluster_analysis_df, insights, report

if __name__ == "__main__":
    cluster_analysis_df, insights, report = main()
