#!/usr/bin/env python3
"""
K-Means Property Segmentation Pipeline
Main orchestrator for the complete clustering analysis
"""

import os
import sys
from datetime import datetime

def print_header():
    """Print project header"""
    print("=" * 60)
    print("K-MEANS PROPERTY SEGMENTATION - JABODETABEK")
    print("Property Market Analysis Using Machine Learning")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def print_section_header(title):
    """Print section header"""
    print("\n" + "=" * 40)
    print(f"{title}")
    print("=" * 40)

def check_dependencies():
    """Check if required files exist"""
    print("Checking dependencies...")
    
    required_files = ['../data/dataset.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        sys.exit(1)
    
    print("All dependencies found")

def run_preprocessing():
    """Run preprocessing and normalization"""
    print_section_header("STEP 1: PREPROCESSING & NORMALIZATION")
    
    try:
        from preprocessing import main as preprocess_main
        features_normalized, scaler = preprocess_main()
        print("Preprocessing completed successfully")
        return True
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return False

def run_elbow_method():
    """Run elbow method to find optimal clusters"""
    print_section_header("STEP 2: ELBOW METHOD")
    
    try:
        from elbow_method import main as elbow_main
        optimal_k = elbow_main()
        print(f"Optimal number of clusters found: {optimal_k}")
        return True
    except Exception as e:
        print(f"Elbow method failed: {e}")
        return False

def run_kmeans_clustering():
    """Run K-Means clustering"""
    print_section_header("STEP 3: K-MEANS CLUSTERING")
    
    try:
        from kmeans import main as kmeans_main
        kmeans, cluster_labels, cluster_centers = kmeans_main()
        print("K-Means clustering completed successfully")
        return True
    except Exception as e:
        print(f"K-Means clustering failed: {e}")
        return False

def run_visualization():
    """Run visualization pipeline"""
    print_section_header("STEP 4: VISUALIZATION")
    
    try:
        from visualization import main as viz_main
        cluster_stats = viz_main()
        print("Visualization completed successfully")
        return True
    except Exception as e:
        print(f"Visualization failed: {e}")
        return False

def run_analysis():
    """Run cluster analysis"""
    print_section_header("STEP 5: ANALYSIS & INSIGHTS")
    
    try:
        from analysis import main as analysis_main
        cluster_analysis_df, insights, report = analysis_main()
        print("Analysis completed successfully")
        return True
    except Exception as e:
        print(f"Analysis failed: {e}")
        return False

def generate_final_summary():
    """Generate final project summary"""
    print_section_header("FINAL SUMMARY")
    
    # Check all output files
    output_files = [
        '../output/features_normalized.csv',
        '../output/features_clean.csv',
        '../output/scaler.pkl',
        '../output/elbow_method_results.csv',
        '../output/optimal_k.txt',
        '../output/cluster_labels.csv',
        '../output/kmeans_model.pkl',
        '../output/clustering_metrics.csv',
        '../output/clustering_visualizations.png',
        '../output/cluster_summary.png',
        '../output/detailed_cluster_analysis.csv',
        '../output/analysis_insights.txt',
        '../output/analysis_report.txt'
    ]
    
    print("Generated Files:")
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  {file} ({size:,} bytes)")
        else:
            print(f"  {file} (missing)")
    
    print(f"\nProject completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main pipeline orchestrator"""
    print_header()
    
    # Check dependencies
    check_dependencies()
    
    # Define pipeline steps
    pipeline_steps = [
        ("Preprocessing & Normalization", run_preprocessing),
        ("Elbow Method", run_elbow_method),
        ("K-Means Clustering", run_kmeans_clustering),
        ("Visualization", run_visualization),
        ("Analysis & Insights", run_analysis)
    ]
    
    # Run pipeline
    completed_steps = []
    failed_steps = []
    
    for step_name, step_func in pipeline_steps:
        print(f"\n{'='*60}")
        print(f"EXECUTING: {step_name}")
        print('='*60)
        
        if step_func():
            completed_steps.append(step_name)
            print(f"✓ {step_name} completed")
        else:
            failed_steps.append(step_name)
            print(f"✗ {step_name} failed")
            print("Stopping pipeline due to failure")
            break
    
    # Generate final summary
    generate_final_summary()
    
    # Print completion status
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETION STATUS")
    print('='*60)
    
    if completed_steps:
        print("Completed Steps:")
        for step in completed_steps:
            print(f"  {step}")
    
    if failed_steps:
        print("Failed Steps:")
        for step in failed_steps:
            print(f"  {step}")
    
    success_rate = len(completed_steps) / len(pipeline_steps) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("All steps completed successfully!")
    elif success_rate >= 80:
        print("Most steps completed, but some failed")
    else:
        print("Pipeline failed significantly")
    
    return success_rate == 100

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
