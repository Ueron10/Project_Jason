import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"Running: {script_path}")
    print('='*60)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"[FAILED] {script_path}")
        return False
    print(f"[DONE] {script_path}")
    return True

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    pipeline = [
        ("1Preprocessing", "01_clean_text.py"),
        ("1Preprocessing", "05_data_augmentation.py"),  # Data augmentation for better accuracy
        ("1Preprocessing", "02_prepare_labels.py"),
        ("1Preprocessing", "03_tokenize.py"),
        ("1Preprocessing", "04_split_data.py"),
        ("2Training", "01_build_and_train.py"),
        ("2Training", "02_evaluate.py"),
    ]
    
    print("SENTIMENT ANALYSIS PIPELINE")
    print("="*60)
    
    for folder, script in pipeline:
        script_path = os.path.join(base_dir, folder, script)
        if not run_script(script_path):
            print(f"\nPipeline stopped at {script}")
            return 1
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run prediction: python Scripts/3Prediction/01_predict.py")
    print("  2. Start Streamlit: streamlit run app.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
