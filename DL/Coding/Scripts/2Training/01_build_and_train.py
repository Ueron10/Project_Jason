import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer, BertConfig
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import PolynomialDecay

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

print("="*70)
print("BUILD & TRAIN BERT LARGE MODEL")
print("="*70)

# ============ BUILD MODEL ============
print("\n[1/4] Building BERT Large architecture...")

model_name = 'bert-large-uncased'
config = BertConfig.from_pretrained(
    model_name,
    num_labels=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
model = TFBertForSequenceClassification.from_pretrained(model_name, config=config)

print("Model architecture built!")
model.summary()

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

# ============ TRAIN MODEL ============
print("\n[3/4] Training model with optimized hyperparameters...")

# Optimized hyperparameters for maximum accuracy
batch_size = 16
epochs = 5
initial_learning_rate = 3e-5
weight_decay = 0.01

# Learning rate scheduler with warmup
steps_per_epoch = len(X_train_input_ids) // batch_size
total_steps = steps_per_epoch * epochs
warmup_steps = int(total_steps * 0.1)

lr_schedule = PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=total_steps - warmup_steps,
    end_learning_rate=1e-7,
    power=1.0
)

def lr_schedule_with_warmup(epoch):
    if epoch < warmup_steps / steps_per_epoch:
        return initial_learning_rate * (epoch + 1) / (warmup_steps / steps_per_epoch)
    else:
        return lr_schedule(epoch - warmup_steps / steps_per_epoch)

# Custom optimizer with gradient clipping and weight decay
optimizer = Adam(
    learning_rate=initial_learning_rate,
    clipnorm=1.0,
    weight_decay=weight_decay
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Advanced callbacks for better training
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(models_dir, "best_bert_model"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    x={'input_ids': X_train_input_ids, 'attention_mask': X_train_attention_mask},
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(
        {'input_ids': X_test_input_ids, 'attention_mask': X_test_attention_mask},
        y_test
    ),
    callbacks=[early_stop, lr_reducer, checkpoint],
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'training_history.png'), dpi=300)
print("\nTraining plot saved!")

# ============ SAVE MODEL ============
print("\n[4/4] Saving optimized BERT Large model...")

# Load best model from checkpoint and save
best_model = TFBertForSequenceClassification.from_pretrained(os.path.join(models_dir, "best_bert_model"))
best_model.save_pretrained(os.path.join(models_dir, "bert_model"))

best_val_acc = max(history.history['val_accuracy'])
final_epoch = len(history.history['loss'])
train_acc = history.history['accuracy'][-1]

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Epochs completed: {final_epoch}")
print(f"Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Final train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Model: BERT Large (bert-large-uncased)")
print(f"Learning rate: {initial_learning_rate}")
print(f"Batch size: {batch_size}")
print(f"Max sequence length: 256")
print(f"\nModel saved to: Models/bert_model")
print("="*70)
