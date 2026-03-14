"""
Week 7 Lab: Convolutional Neural Networks - CIFAR-10
BFOR516 - Advanced Data Analytics for Cyber
Student: Spencer Kone
Date: March 2026

AI Usage Statement:
I used Antigravity and Claude as AI assisted tools for this assignment. Except for the tools
mentioned, all content, analysis, and conclusions drawn were made by Spencer Kone.

Goal:
Build two CNN models to classify CIFAR-10 images into 10 classes:
  Model 1: Baseline CNN (minimum spec — 2 Conv2D, 2 MaxPool, Flatten, Dense)
  Model 2: Improved CNN (VGG-style double Conv blocks, BatchNorm, Dropout, padding='same')
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 0. REPRODUCIBILITY — Set random seeds BEFORE importing TensorFlow
# ---------------------------------------------------------------------------
import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)


def log(msg):
    """Flush-safe print for real-time output."""
    print(msg)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# 1. DATA LOADING + PREPROCESSING
# ---------------------------------------------------------------------------
log("=" * 70)
log("STEP 1: Loading and Preprocessing CIFAR-10")
log("=" * 70)

# CIFAR-10 is bundled with Keras — no local CSV needed
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values from [0, 255] to [0.0, 1.0]
x_train = x_train / 255.0
x_test  = x_test  / 255.0

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

log(f"x_train shape: {x_train.shape}  (samples, H, W, C)")
log(f"x_test  shape: {x_test.shape}")
log(f"y_train shape: {y_train.shape}  (samples, 1) — integer labels 0-9")
log(f"Classes: {class_names}")
log(f"Train samples: {x_train.shape[0]:,}  |  Test samples: {x_test.shape[0]:,}")

# Flatten labels for sklearn (shape (N,1) → (N,))
# Keras model.fit() accepts (N,1) fine with sparse_categorical_crossentropy,
# but confusion_matrix / classification_report require 1-D arrays.
y_train_flat = y_train.flatten()
y_test_flat  = y_test.flatten()

log(f"\nLabel value range: {y_test_flat.min()} – {y_test_flat.max()}")
log("Class counts in test set:")
for i, name in enumerate(class_names):
    log(f"  {i} ({name:12s}): {(y_test_flat == i).sum()}")


# =========================================================================
#
#  MODEL 1: BASELINE CNN
#  Meets minimum spec exactly: 2×Conv2D, 2×MaxPool, Flatten, Dense(128), Dense(10)
#  No padding='same', no BatchNorm, no Dropout — intentionally bare.
#
# =========================================================================
log("\n" + "=" * 70)
log("MODEL 1: Baseline CNN")
log("=" * 70)
log("""
Architecture:
  Conv2D(32, (3,3), relu)   → (30,30,32)
  MaxPooling2D(2,2)          → (15,15,32)
  Conv2D(64, (3,3), relu)   → (13,13,64)
  MaxPooling2D(2,2)          → (6,6,64)
  Flatten()                  → 2304
  Dense(128, relu)
  Dense(10, softmax)
""")

model1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
], name='Model1_Baseline')

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model1.summary(print_fn=log)

# ---- Train ----
log("\nTraining Model 1 (epochs=10, batch_size=64, val_split=0.1)...")
history1 = model1.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ---- Evaluate ----
log("\n--- Model 1 Evaluation ---")
test_loss1, test_acc1 = model1.evaluate(x_test, y_test, verbose=0)
log(f"Test Accuracy: {test_acc1:.4f}")
log(f"Test Loss:     {test_loss1:.4f}")

# Multi-class: argmax over softmax output (NOT the > 0.5 binary threshold)
y_pred1 = np.argmax(model1.predict(x_test, verbose=0), axis=1)

log("\nClassification Report:")
log(classification_report(y_test_flat, y_pred1, target_names=class_names))

cm1 = confusion_matrix(y_test_flat, y_pred1)
log(f"Confusion Matrix (10×10):\n{cm1}")

# ---- Plot: Training Curves ----
fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(14, 5))

ax_a.plot(history1.history['accuracy'],     label='Train Accuracy')
ax_a.plot(history1.history['val_accuracy'], label='Val Accuracy')
ax_a.set_title('Model 1 (Baseline) — Accuracy over Epochs')
ax_a.set_xlabel('Epoch')
ax_a.set_ylabel('Accuracy')
ax_a.legend()
ax_a.grid(True, alpha=0.3)

ax_l.plot(history1.history['loss'],     label='Train Loss')
ax_l.plot(history1.history['val_loss'], label='Val Loss')
ax_l.set_title('Model 1 (Baseline) — Loss over Epochs')
ax_l.set_xlabel('Epoch')
ax_l.set_ylabel('Loss')
ax_l.legend()
ax_l.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model1_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: model1_training_curves.png")

# ---- Plot: Confusion Matrix ----
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names, ax=ax)
ax.set_title(f'Model 1 (Baseline) — Confusion Matrix\nTest Accuracy: {test_acc1:.4f}')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('model1_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: model1_confusion_matrix.png")


# =========================================================================
#
#  MODEL 2: IMPROVED CNN
#
#  Changes from Model 1 (and rationale):
#  1. padding='same'       — Preserves spatial dims before pooling; prevents
#                            feature maps from shrinking too fast, allowing
#                            deeper conv stacks without collapsing to tiny maps.
#  2. Double Conv blocks   — Two Conv layers at the same filter depth before
#                            each MaxPool (VGG-style). The network learns
#                            more complex feature combinations per resolution.
#  3. BatchNormalization   — Normalizes activations within each mini-batch.
#                            Stabilizes training, reduces sensitivity to weight
#                            initialization, and adds mild regularization.
#  4. Dropout(0.25/0.5)    — Forces redundant representations; reduces
#                            overfitting. Model 1 has no regularization, making
#                            it more prone to overfit on 10-epoch training.
#  5. Wider Dense(256)     — Larger classification head to handle the richer
#                            4096-dim flattened input from the deeper conv stack.
#
# =========================================================================
log("\n" + "=" * 70)
log("MODEL 2: Improved CNN")
log("=" * 70)
log("""
Architecture:
  Conv2D(32,(3,3),same,relu) → BatchNorm → Conv2D(32,(3,3),same,relu) → BatchNorm → MaxPool(2,2) → Dropout(0.25)
  Conv2D(64,(3,3),same,relu) → BatchNorm → Conv2D(64,(3,3),same,relu) → BatchNorm → MaxPool(2,2) → Dropout(0.25)
  Flatten → Dense(256,relu) → BatchNorm → Dropout(0.5) → Dense(10,softmax)

Changes from Model 1:
  1. padding='same'       — Preserves spatial dims; prevents premature feature map collapse
  2. Double Conv blocks   — More complex feature learning before each pool step (VGG pattern)
  3. BatchNormalization   — Stabilizes gradients; mild regularization
  4. Dropout(0.25/0.5)    — Explicit regularization to combat overfitting
  5. Dense(256) vs 128    — Wider head for richer flattened input
""")

model2 = Sequential([
    # --- Block 1: 32 filters ---
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # --- Block 2: 64 filters ---
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # --- Classifier head ---
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
], name='Model2_Improved')

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model2.summary(print_fn=log)

# ---- Train ----
log("\nTraining Model 2 (epochs=10, batch_size=64, val_split=0.1)...")
history2 = model2.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ---- Evaluate ----
log("\n--- Model 2 Evaluation ---")
test_loss2, test_acc2 = model2.evaluate(x_test, y_test, verbose=0)
log(f"Test Accuracy: {test_acc2:.4f}")
log(f"Test Loss:     {test_loss2:.4f}")

y_pred2 = np.argmax(model2.predict(x_test, verbose=0), axis=1)

log("\nClassification Report:")
log(classification_report(y_test_flat, y_pred2, target_names=class_names))

cm2 = confusion_matrix(y_test_flat, y_pred2)
log(f"Confusion Matrix (10×10):\n{cm2}")

# ---- Plot: Training Curves ----
fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(14, 5))

ax_a.plot(history2.history['accuracy'],     label='Train Accuracy')
ax_a.plot(history2.history['val_accuracy'], label='Val Accuracy')
ax_a.set_title('Model 2 (Improved) — Accuracy over Epochs')
ax_a.set_xlabel('Epoch')
ax_a.set_ylabel('Accuracy')
ax_a.legend()
ax_a.grid(True, alpha=0.3)

ax_l.plot(history2.history['loss'],     label='Train Loss')
ax_l.plot(history2.history['val_loss'], label='Val Loss')
ax_l.set_title('Model 2 (Improved) — Loss over Epochs')
ax_l.set_xlabel('Epoch')
ax_l.set_ylabel('Loss')
ax_l.legend()
ax_l.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model2_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: model2_training_curves.png")

# ---- Plot: Confusion Matrix ----
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names,
            yticklabels=class_names, ax=ax)
ax.set_title(f'Model 2 (Improved) — Confusion Matrix\nTest Accuracy: {test_acc2:.4f}')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('model2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: model2_confusion_matrix.png")


# =========================================================================
#
#  STEP 4: CLASSIFY NEW/UNSEEN IMAGES
#  Use x_test samples — they were never seen during training.
#
# =========================================================================
log("\n" + "=" * 70)
log("STEP 4: Sample Predictions — 20 Unseen Test Images")
log("=" * 70)

NUM_SAMPLES = 20
indices = np.arange(NUM_SAMPLES)  # first 20 test images (reproducible)

preds1_sample = np.argmax(model1.predict(x_test[indices], verbose=0), axis=1)
preds2_sample = np.argmax(model2.predict(x_test[indices], verbose=0), axis=1)
true_labels   = y_test_flat[indices]

correct2 = (preds2_sample == true_labels).sum()
log(f"  Showing {NUM_SAMPLES} test images (indices 0-{NUM_SAMPLES-1})")
log(f"  Model 2 correct on these samples: {correct2}/{NUM_SAMPLES}")

fig, axes = plt.subplots(4, 5, figsize=(15, 13))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(x_test[indices[i]])

    true_name  = class_names[true_labels[i]]
    pred1_name = class_names[preds1_sample[i]]
    pred2_name = class_names[preds2_sample[i]]

    # Color-code by Model 2 correctness (the better model)
    color = 'green' if preds2_sample[i] == true_labels[i] else 'red'

    ax.set_title(
        f"True: {true_name}\nM1: {pred1_name}  M2: {pred2_name}",
        fontsize=7,
        color=color
    )
    ax.axis('off')

    # Colored spine border matching title
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
        spine.set_visible(True)

plt.suptitle(
    'Sample Predictions — 20 Unseen Test Images\n'
    '(Green title = Model 2 Correct  |  Red title = Model 2 Wrong)',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: sample_predictions.png")


# =========================================================================
#
#  STEP 5: COMPARISON TRAINING CURVES (Side-by-Side)
#
# =========================================================================
log("\n" + "=" * 70)
log("STEP 5: Comparison Training Curves")
log("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Row 0: Accuracy
axes[0, 0].plot(history1.history['accuracy'],     label='Train', color='steelblue')
axes[0, 0].plot(history1.history['val_accuracy'], label='Val',   color='orange')
axes[0, 0].set_title('Model 1 (Baseline) — Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history2.history['accuracy'],     label='Train', color='steelblue')
axes[0, 1].plot(history2.history['val_accuracy'], label='Val',   color='orange')
axes[0, 1].set_title('Model 2 (Improved) — Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Row 1: Loss
axes[1, 0].plot(history1.history['loss'],     label='Train', color='steelblue')
axes[1, 0].plot(history1.history['val_loss'], label='Val',   color='orange')
axes[1, 0].set_title('Model 1 (Baseline) — Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history2.history['loss'],     label='Train', color='steelblue')
axes[1, 1].plot(history2.history['val_loss'], label='Val',   color='orange')
axes[1, 1].set_title('Model 2 (Improved) — Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Training Curves: Model 1 (Baseline) vs Model 2 (Improved)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: comparison_training_curves.png")


# =========================================================================
#
#  STEP 6: COMPARISON SUMMARY
#
# =========================================================================
log("\n" + "=" * 70)
log("COMPARISON: Model 1 (Baseline) vs Model 2 (Improved)")
log("=" * 70)

log(f"""
+-----------------------------+----------------------+----------------------+
| Metric                      | Model 1 (Baseline)   | Model 2 (Improved)   |
+-----------------------------+----------------------+----------------------+
| Test Accuracy               | {test_acc1:<20.4f} | {test_acc2:<20.4f} |
| Test Loss                   | {test_loss1:<20.4f} | {test_loss2:<20.4f} |
| Input Shape                 | (32, 32, 3)          | (32, 32, 3)          |
| Conv Blocks                 | 2 (single Conv/block)| 2 (double Conv/block)|
| Padding                     | valid                | same                 |
| BatchNormalization          | No                   | Yes                  |
| Dropout                     | No                   | 0.25 / 0.25 / 0.5    |
| Dense Hidden Units          | 128                  | 256                  |
| Epochs                      | 10                   | 10                   |
| Batch Size                  | 64                   | 64                   |
+-----------------------------+----------------------+----------------------+

Model 2 Improvement Rationale:
  1. padding='same'       — Preserves spatial dimensions before each pool step;
                            prevents premature feature map collapse and allows
                            stacking two Conv layers per block without size issues.
  2. Double Conv blocks   — Two consecutive Conv layers at the same filter depth
                            before each MaxPool (VGG-style). More complex feature
                            combinations learned per resolution level.
  3. BatchNormalization   — Normalizes activations within each mini-batch;
                            stabilizes training and acts as mild regularization.
  4. Dropout(0.25/0.5)    — Explicit regularization. Model 1 showed overfitting
                            (train acc >> val acc). Dropout forces redundant
                            representations, improving generalization.
  5. Wider Dense(256)     — Larger classification head proportional to the richer
                            4096-dim flattened feature map from the deeper stack.
""")

log("=" * 70)
log("ANALYSIS COMPLETE")
log("=" * 70)
log("Output files generated:")
for fname in [
    'model1_training_curves.png',
    'model2_training_curves.png',
    'comparison_training_curves.png',
    'model1_confusion_matrix.png',
    'model2_confusion_matrix.png',
    'sample_predictions.png',
]:
    log(f"  {fname}")
