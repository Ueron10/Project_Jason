import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

print("="*70)
print("BUILD & TRAIN BERT LARGE MODEL (PyTorch)")
print("="*70)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# ============ BUILD MODEL ============
print("\n[1/4] Building BERT Large architecture...")

model_name = 'bert-large-uncased'
config = BertConfig.from_pretrained(
    model_name,
    num_labels=3,  # 3 classes: negative, neutral, positive
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
model = BertForSequenceClassification.from_pretrained(model_name, config=config)
model.to(device)

print("Model architecture built!")
print(f"Model: {model_name}")
print(f"Number of labels: {config.num_labels}")

# ============ LOAD DATA ============
print("\n[2/4] Loading training data...")

X_train_input_ids = np.load(os.path.join(outputs_dir, "X_train_input_ids.npy"))
X_train_attention_mask = np.load(os.path.join(outputs_dir, "X_train_attention_mask.npy"))
X_test_input_ids = np.load(os.path.join(outputs_dir, "X_test_input_ids.npy"))
X_test_attention_mask = np.load(os.path.join(outputs_dir, "X_test_attention_mask.npy"))
y_train = np.load(os.path.join(outputs_dir, "y_train.npy"))
y_test = np.load(os.path.join(outputs_dir, "y_test.npy"))

print(f"  Training: {X_train_input_ids.shape}")
print(f"  Testing:  {X_test_input_ids.shape}")

# Convert to PyTorch tensors
train_inputs = torch.tensor(X_train_input_ids, dtype=torch.long)
train_masks = torch.tensor(X_train_attention_mask, dtype=torch.long)
train_labels = torch.tensor(y_train, dtype=torch.long)

test_inputs = torch.tensor(X_test_input_ids, dtype=torch.long)
test_masks = torch.tensor(X_test_attention_mask, dtype=torch.long)
test_labels = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# ============ TRAIN MODEL ============
print("\n[3/4] Training model with optimized hyperparameters...")

# Optimized hyperparameters
epochs = 5
learning_rate = 3e-5
weight_decay = 0.01

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
total_steps = len(train_dataloader) * epochs
warmup_steps = int(total_steps * 0.1)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Training loop
training_stats = []
best_val_accuracy = 0

for epoch_i in range(0, epochs):
    print(f"\nEpoch {epoch_i + 1}/{epochs}")
    print("-" * 70)

    # Training
    model.train()
    total_train_loss = 0
    train_accuracy = 0

    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        outputs = model(b_input_ids,
                      attention_mask=b_input_mask,
                      labels=b_labels)

        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Calculate training accuracy
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        train_accuracy += (preds == b_labels).cpu().numpy().mean()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_accuracy = train_accuracy / len(train_dataloader)

    # Validation
    model.eval()
    total_val_accuracy = 0
    total_val_loss = 0

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(b_input_ids,
                          attention_mask=b_input_mask,
                          labels=b_labels)

        loss = outputs.loss
        total_val_loss += loss.item()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        total_val_accuracy += (preds == b_labels).cpu().numpy().mean()

    avg_val_accuracy = total_val_accuracy / len(test_dataloader)
    avg_val_loss = total_val_loss / len(test_dataloader)

    # Save best model
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        model.save_pretrained(os.path.join(models_dir, "bert_model"))

    training_stats.append({
        'epoch': epoch_i + 1,
        'train_loss': avg_train_loss,
        'train_acc': avg_train_accuracy,
        'val_loss': avg_val_loss,
        'val_acc': avg_val_accuracy
    })

    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Train Acc: {avg_train_accuracy:.4f} ({avg_train_accuracy*100:.2f}%)")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Acc: {avg_val_accuracy:.4f} ({avg_val_accuracy*100:.2f}%)")

# Plot training history
history_df = pd.DataFrame(training_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_df['epoch'], history_df['train_acc'], label='Training')
axes[0].plot(history_df['epoch'], history_df['val_acc'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_df['epoch'], history_df['train_loss'], label='Training')
axes[1].plot(history_df['epoch'], history_df['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'training_history.png'), dpi=300)
print("\nTraining plot saved!")

# ============ SAVE TOKENIZER ============
print("\n[4/4] Saving tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(os.path.join(models_dir, "bert_model"))

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Epochs completed: {len(training_stats)}")
print(f"Best val accuracy: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
print(f"Final train accuracy: {training_stats[-1]['train_acc']:.4f} ({training_stats[-1]['train_acc']*100:.2f}%)")
print(f"Model: BERT Large (bert-large-uncased)")
print(f"Learning rate: {learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Max sequence length: 256")
print(f"\nModel saved to: Models/bert_model")
print("="*70)
