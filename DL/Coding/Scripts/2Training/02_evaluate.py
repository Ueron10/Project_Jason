import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from transformers import TFBertForSequenceClassification
import tensorflow as tf
import seaborn as sns
import os

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

# Load model
print("Loading BERT Large model...")
model = TFBertForSequenceClassification.from_pretrained(os.path.join(models_dir, "bert_model"))

# Load test data
X_test_input_ids = np.load(os.path.join(outputs_dir, "X_test_input_ids.npy"))
X_test_attention_mask = np.load(os.path.join(outputs_dir, "X_test_attention_mask.npy"))
y_test = np.load(os.path.join(outputs_dir, "y_test.npy"))
label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)

print(f"Test data: {X_test_input_ids.shape}")

# Evaluate
test_loss, test_acc = model.evaluate(
    {'input_ids': X_test_input_ids, 'attention_mask': X_test_attention_mask},
    y_test,
    verbose=0
)
print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred_logits = model.predict(
    {'input_ids': X_test_input_ids, 'attention_mask': X_test_attention_mask},
    verbose=0
)
y_pred_probs = tf.nn.softmax(y_pred_logits.logits, axis=-1).numpy()
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Calculate AUC-ROC for binary classification
try:
    auc_roc = roc_auc_score(y_test, y_pred_probs[:, 1])
except:
    auc_roc = None

print("\n" + "="*60)
print("Comprehensive Evaluation Metrics")
print("="*60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
if auc_roc:
    print(f"AUC-ROC:   {auc_roc:.4f} ({auc_roc*100:.2f}%)")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_classes))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_classes, yticklabels=label_classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - BERT Large')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'confusion_matrix.png'), dpi=300)
print("\nConfusion matrix saved!")

# ROC Curve (if binary classification)
if auc_roc:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - BERT Large')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, 'roc_curve.png'), dpi=300)
    print("ROC curve saved!")
