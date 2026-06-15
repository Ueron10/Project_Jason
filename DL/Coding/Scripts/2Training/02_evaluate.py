import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import tensorflow as tf
import seaborn as sns
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

print("Loading LSTM model...")
model = tf.keras.models.load_model(os.path.join(models_dir, "lstm_model.h5"))

X_test_sequences = np.load(os.path.join(outputs_dir, "X_test_sequences.npy"))
y_test = np.load(os.path.join(outputs_dir, "y_test.npy"))
label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)

print(f"Test data: {X_test_sequences.shape}")

test_loss, test_acc = model.evaluate(X_test_sequences, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

y_pred_probs = model.predict(X_test_sequences, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n" + "="*60)
print("Comprehensive Evaluation Metrics")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_classes))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_classes, yticklabels=label_classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - LSTM')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'confusion_matrix.png'), dpi=300)
print("\nConfusion matrix saved!")
