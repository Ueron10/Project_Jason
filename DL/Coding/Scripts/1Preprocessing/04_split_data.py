import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")
outputs_dir = os.path.join(project_root, "Outputs")

# Load data
df = pd.read_csv(os.path.join(data_dir, "threads_reviews_labeled.csv"))
X_input_ids = np.load(os.path.join(outputs_dir, "X_input_ids.npy"))
X_attention_mask = np.load(os.path.join(outputs_dir, "X_attention_mask.npy"))

# Encode labels with fixed order: negative=0, neutral=1, positive=2
y_text = df['sentiment'].values

# Use LabelEncoder but ensure consistent mapping
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

# Verify and print the encoding order
print("Label encoding mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i} = {label}")

# Ensure order is [negative, neutral, positive]
desired_order = ['negative', 'neutral', 'positive']
if len(label_encoder.classes_) == 3:
    # Map to desired order
    label_mapping = {label: idx for idx, label in enumerate(desired_order)}
    y = np.array([label_mapping[label] for label in y_text])
    np.save(os.path.join(outputs_dir, "label_encoding.npy"), np.array(desired_order))
    print(f"Label order set to: {desired_order}")
else:
    # Save whatever order we have
    np.save(os.path.join(outputs_dir, "label_encoding.npy"), label_encoder.classes_)

# Split parameters
train_ratio = 0.8
np.random.seed(42)
indices = np.random.permutation(len(X_input_ids))

train_size = int(len(X_input_ids) * train_ratio)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train_input_ids, X_test_input_ids = X_input_ids[train_idx], X_input_ids[test_idx]
X_train_attention_mask, X_test_attention_mask = X_attention_mask[train_idx], X_attention_mask[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Simpan
np.save(os.path.join(outputs_dir, "X_train_input_ids.npy"), X_train_input_ids)
np.save(os.path.join(outputs_dir, "X_test_input_ids.npy"), X_test_input_ids)
np.save(os.path.join(outputs_dir, "X_train_attention_mask.npy"), X_train_attention_mask)
np.save(os.path.join(outputs_dir, "X_test_attention_mask.npy"), X_test_attention_mask)
np.save(os.path.join(outputs_dir, "y_train.npy"), y_train)
np.save(os.path.join(outputs_dir, "y_test.npy"), y_test)
# Note: label_encoding.npy is already saved above with correct order

print("Data splitting selesai!")
print(f"\nDataset split (80/20):")
print(f"  Training: {len(X_train_input_ids):,} samples")
print(f"  Testing:  {len(X_test_input_ids):,} samples")
print(f"\nLabel mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i} = {label}")
