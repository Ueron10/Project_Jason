from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import pickle
import os
import sys

app = Flask(__name__)

# Load model and data
def load_model():
    with open('output/kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('output/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return kmeans, scaler

def load_base_data():
    """Load base data needed for dynamic analysis generation"""
    cluster_labels = pd.read_csv('output/cluster_labels.csv')
    features_clean = pd.read_csv('output/features_clean.csv')
    clustering_metrics = pd.read_csv('output/clustering_metrics.csv')
    return cluster_labels, features_clean, clustering_metrics

def analyze_cluster_characteristics(features_clean, cluster_labels):
    """Analyze detailed characteristics of each cluster"""
    cluster_data = features_clean.copy()
    cluster_data['cluster'] = cluster_labels['cluster'].values
    
    cluster_analysis = []
    
    for cluster_id in sorted(cluster_data['cluster'].unique()):
        cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
        
        analysis = {
            'cluster_id': cluster_id,
            'count': len(cluster_subset),
            'percentage': (len(cluster_subset) / len(cluster_data)) * 100,
            'price_mean': cluster_subset['price_in_rp'].mean(),
            'size_mean': cluster_subset['building_size_m2'].mean(),
            'price_per_m2_mean': (cluster_subset['price_in_rp'] / cluster_subset['building_size_m2']).mean(),
        }
        
        cluster_analysis.append(analysis)
    
    cluster_analysis_df = pd.DataFrame(cluster_analysis)
    return cluster_analysis_df, cluster_data

def classify_property_categories(cluster_analysis_df):
    """Classify clusters into property categories based on characteristics"""
    categories = []
    
    for _, cluster in cluster_analysis_df.iterrows():
        price_mean = cluster['price_mean']
        size_mean = cluster['size_mean']
        
        if price_mean < 1e9:
            if size_mean < 100:
                category = "Budget Small"
            else:
                category = "Budget Medium"
        elif price_mean < 3e9:
            if size_mean < 150:
                category = "Standard Small"
            elif size_mean < 250:
                category = "Standard Medium"
            else:
                category = "Standard Large"
        elif price_mean < 7e9:
            if size_mean < 200:
                category = "Premium Small"
            elif size_mean < 350:
                category = "Premium Medium"
            else:
                category = "Premium Large"
        else:
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
    insights = []
    
    inertia = clustering_metrics[clustering_metrics['metric'] == 'inertia']['value'].iloc[0]
    
    insights.append(f"**Clustering Performance:**")
    insights.append(f"- Inertia: {inertia:.2f}")
    insights.append(f"- Number of Clusters: {len(cluster_analysis_df)}")
    
    insights.append(f"\n**Market Segmentation Insights:**")
    
    dominant_cluster = cluster_analysis_df.loc[cluster_analysis_df['count'].idxmax()]
    insights.append(f"- Largest Segment: {dominant_cluster['category']} ({dominant_cluster['count']} properties, {dominant_cluster['percentage']:.1f}%)")
    
    price_ranges = cluster_analysis_df[['price_mean', 'category']].sort_values('price_mean')
    insights.append(f"\n**Price Range Analysis:**")
    for _, row in price_ranges.iterrows():
        insights.append(f"- {row['category']}: Avg Price Rp {row['price_mean']/1e9:.2f}B")
    
    size_analysis = cluster_analysis_df[['size_mean', 'category']].sort_values('size_mean')
    insights.append(f"\n**Size Preferences:**")
    for _, row in size_analysis.iterrows():
        insights.append(f"- {row['category']}: Avg Size {row['size_mean']:.0f} m2")
    
    return '\n'.join(insights)

def create_detailed_report(cluster_analysis_df, insights, clustering_metrics):
    """Create a detailed analysis report"""
    report = []
    report.append("PROPERTY SEGMENTATION ANALYSIS REPORT")
    report.append("=" * 50)
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"Total Properties Analyzed: {cluster_analysis_df['count'].sum()}")
    report.append(f"Number of Segments Identified: {len(cluster_analysis_df)}")
    report.append("")
    
    report.append("DETAILED CLUSTER ANALYSIS")
    report.append("-" * 30)
    report.append("")
    
    for _, cluster in cluster_analysis_df.iterrows():
        report.append(f"Cluster {cluster['cluster_id']}: {cluster['category']}")
        report.append(f"  Properties: {cluster['count']} ({cluster['percentage']:.1f}%)")
        report.append(f"  Average Price: Rp {cluster['price_mean']/1e9:.2f}B")
        report.append(f"  Average Size: {cluster['size_mean']:.0f} m2")
        report.append(f"  Price per m2: Rp {cluster['price_per_m2_mean']/1e6:.1f}M")
        report.append("")
    
    return '\n'.join(report)

def generate_analysis():
    """Generate analysis dynamically"""
    cluster_labels, features_clean, clustering_metrics = load_base_data()
    cluster_analysis_df, cluster_data = analyze_cluster_characteristics(features_clean, cluster_labels)
    cluster_analysis_df = classify_property_categories(cluster_analysis_df)
    insights = generate_insights(cluster_analysis_df, clustering_metrics)
    report = create_detailed_report(cluster_analysis_df, insights, clustering_metrics)
    return cluster_analysis_df, cluster_labels, features_clean, insights, report, clustering_metrics

# Load model at startup
kmeans, scaler = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from the output directory"""
    return send_from_directory('output', filename)

@app.route('/api/cluster_data')
def get_cluster_data():
    """Return clustering data for the review tab - generated dynamically"""
    cluster_analysis_df, cluster_labels, features_clean, insights, report, clustering_metrics = generate_analysis()
    
    display_df = cluster_analysis_df.copy()
    display_df['price_mean_billion'] = (display_df['price_mean'] / 1e9).round(2)
    display_df['size_mean_m2'] = display_df['size_mean'].round(0)
    display_df['price_per_m2_mean_million'] = (display_df['price_per_m2_mean'] / 1e6).round(1)
    
    display_cols = ['cluster_id', 'category', 'count', 'percentage', 
                   'price_mean_billion', 'size_mean_m2', 'price_per_m2_mean_million']
    
    return jsonify({
        'cluster_data': display_df[display_cols].to_dict('records'),
        'total_properties': int(len(features_clean)),
        'num_clusters': int(len(cluster_analysis_df)),
        'insights': insights,
        'report': report
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict cluster for user input - uses dynamic analysis"""
    data = request.json
    price = float(data['price'])
    size = float(data['size'])
    
    # Generate analysis dynamically
    cluster_analysis_df, cluster_labels, features_clean, insights, report, clustering_metrics = generate_analysis()
    
    # Prepare input data
    input_data = np.array([[price, size]])
    
    # Normalize the input
    input_normalized = scaler.transform(input_data)
    
    # Predict cluster
    cluster_pred = int(kmeans.predict(input_normalized)[0])
    
    # Get cluster information
    cluster_info = cluster_analysis_df[cluster_analysis_df['cluster_id'] == cluster_pred].iloc[0]
    
    # Calculate distance to cluster center
    cluster_center = kmeans.cluster_centers_[cluster_pred]
    distance = float(np.linalg.norm(input_normalized[0] - cluster_center))
    
    # Calculate differences
    diff_price = ((price - cluster_info['price_mean']) / cluster_info['price_mean']) * 100
    diff_size = ((size - cluster_info['size_mean']) / cluster_info['size_mean']) * 100
    
    # Compare with all clusters
    comparison_data = []
    for idx, row in cluster_analysis_df.iterrows():
        center = kmeans.cluster_centers_[int(row['cluster_id'])]
        dist = float(np.linalg.norm(input_normalized[0] - center))
        comparison_data.append({
            'cluster_id': int(row['cluster_id']),
            'category': row['category'],
            'distance': dist,
            'avg_price_billion': float(row['price_mean']/1e9),
            'avg_size': float(row['size_mean'])
        })
    
    comparison_data.sort(key=lambda x: x['distance'])
    
    # Determine confidence
    if distance < 1.0:
        confidence = "High"
    elif distance < 2.0:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return jsonify({
        'cluster_id': cluster_pred,
        'category': cluster_info['category'],
        'count': int(cluster_info['count']),
        'percentage': float(cluster_info['percentage']),
        'avg_price_billion': float(cluster_info['price_mean']/1e9),
        'avg_size': float(cluster_info['size_mean']),
        'price_per_m2_million': float(cluster_info['price_per_m2_mean']/1e6),
        'distance': distance,
        'confidence': confidence,
        'diff_price': diff_price,
        'diff_size': diff_size,
        'comparison': comparison_data
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
