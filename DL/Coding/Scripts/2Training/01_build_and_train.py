import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

print("="*70)
print("BUILD & TRAIN LSTM MODEL (TensorFlow)")
print("="*70)

print(f"\nUsing device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

print("\n[1/4] Building LSTM architecture...")

with open(os.path.join(outputs_dir, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 128
embedding_dim = 64
lstm_units = 64
num_classes = 3

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    Bidirectional(LSTM(lstm_units, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(lstm_units)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model architecture built!")
print(f"Vocabulary size: {vocab_size}")
print(f"Max sequence length: {max_length}")
print(f"Number of classes: {num_classes}")
model.summary()

print("\n[2/4] Loading training data...")

X_train_sequences = np.load(os.path.join(outputs_dir, "X_train_sequences.npy"))
X_test_sequences = np.load(os.path.join(outputs_dir, "X_test_sequences.npy"))
y_train = np.load(os.path.join(outputs_dir, "y_train.npy"))
y_test = np.load(os.path.join(outputs_dir, "y_test.npy"))

print(f"  Training: {X_train_sequences.shape}")
print(f"  Testing:  {X_test_sequences.shape}")

print("\n[3/4] Training model with optimized hyperparameters...")

epochs = 10
batch_size = 64

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    mode='max'
)

checkpoint = ModelCheckpoint(
    os.path.join(models_dir, "lstm_model.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_format='h5'
)

history = model.fit(
    X_train_sequences, y_train,
    validation_data=(X_test_sequences, y_test),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

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

print("\n[4/4] Saving model...")
model.save(os.path.join(models_dir, "lstm_model.h5"))

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Epochs completed: {len(history.history['loss'])}")
print(f"Best val accuracy: {max(history.history['val_accuracy']):.4f} ({max(history.history['val_accuracy'])*100:.2f}%)")
print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f} ({history.history['accuracy'][-1]*100:.2f}%)")
print(f"Model: Bidirectional LSTM")
print(f"Learning rate: 0.001 (Adam default)")
print(f"Batch size: {batch_size}")
print(f"Max sequence length: {max_length}")
print(f"\nModel saved to: Models/lstm_model.h5")
print("="*70)
