# Lab Report: Convolutional Neural Networks — CIFAR-10
**Course:** BFOR516 - Advanced Data Analytics for Cyber
**Student:** Spencer Kone
**Date:** March 2026

## AI Usage Statement
I used Antigravity and Claude as AI assisted tools for this assignment. Except for the tools mentioned, all content, analysis, and conclusions drawn were made by Spencer Kone.

---

## 1. Objective

Build and iterate on CNN models to classify images from the CIFAR-10 dataset into 10 classes. The lab requires demonstrating the effect of architectural decisions on accuracy by comparing a baseline model against an improved design, and using the trained model to classify unseen test images.

---

## 2. Dataset

**CIFAR-10** is a benchmark image classification dataset developed by the Canadian Institute for Advanced Research.

| Property | Value |
|:---|:---|
| Total images | 60,000 |
| Image dimensions | 32 × 32 pixels, 3 channels (RGB) |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Training set | 50,000 images |
| Test set | 10,000 images |
| Samples per class (test) | 1,000 each (perfectly balanced) |

The dataset was loaded directly from `tf.keras.datasets.cifar10` and required no CSV handling. Pixel values were normalized from [0, 255] to [0.0, 1.0] by dividing by 255. No further preprocessing was needed — the dataset is pre-split, balanced, and contains no missing values.

---

## 3. Model Architectures

Two CNN models were built, both using `adam`, `sparse_categorical_crossentropy` (appropriate for integer-encoded class labels), and trained for 10 epochs with batch size 64 and a 10% validation split.

### Model 1 — Baseline CNN

Designed to meet the minimum lab specification exactly: two Conv2D layers, two MaxPooling layers, one Flatten, one Dense hidden layer, and a Dense(10, softmax) output. No regularization, no padding preservation.

```
Input: (32, 32, 3)
Conv2D(32, 3×3, relu, valid)   → (30, 30, 32)
MaxPooling2D(2×2)               → (15, 15, 32)
Conv2D(64, 3×3, relu, valid)   → (13, 13, 64)
MaxPooling2D(2×2)               → (6, 6, 64)
Flatten()                       → 2,304
Dense(128, relu)
Dense(10, softmax)
```

**Total parameters: 315,722 (1.20 MB)**

### Model 2 — Improved CNN

Designed to address the overfitting and limited feature learning observed in Model 1. Five targeted changes were made:

```
Input: (32, 32, 3)
Conv2D(32, 3×3, same, relu) → BN → Conv2D(32, 3×3, same, relu) → BN → MaxPool(2×2) → Dropout(0.25)
Conv2D(64, 3×3, same, relu) → BN → Conv2D(64, 3×3, same, relu) → BN → MaxPool(2×2) → Dropout(0.25)
Flatten()                   → 4,096
Dense(256, relu) → BatchNorm → Dropout(0.5)
Dense(10, softmax)
```

**Total parameters: 1,118,762 (4.27 MB)**

| Change | Rationale |
|:---|:---|
| `padding='same'` | Preserves the 32×32 spatial dimensions through each Conv layer before pooling. Without it, every 3×3 Conv shrinks the feature map by 2px per side, limiting how many Conv layers can be stacked before the map collapses. |
| Double Conv blocks | Two Conv layers at the same filter depth before each MaxPool (VGG-style). This forces the network to learn more complex feature combinations per resolution level before discarding spatial detail via pooling. |
| BatchNormalization | Normalizes activations within each mini-batch. Stabilizes the gradient signal, reduces sensitivity to weight initialization, and provides mild regularization. Applied after each Conv layer and after the Dense hidden layer. |
| Dropout(0.25/0.5) | Explicit regularization. Model 1 showed clear overfitting (training accuracy climbed to 80% while validation accuracy stalled at ~70% and declined). Dropout forces the network to learn redundant, robust representations. |
| Dense(256) vs Dense(128) | The flattened output from the deeper conv stack is 4,096-dimensional (vs 2,304 in Model 1). A wider Dense(256) head is proportionate to this richer input. |

---

## 4. Training Results

### 4.1 Accuracy and Loss Summary

| Metric | Model 1 (Baseline) | Model 2 (Improved) |
|:---|:---:|:---:|
| **Test Accuracy** | **66.21%** | **78.37%** |
| Test Loss | 1.0848 | 0.6404 |
| Final Train Accuracy | 80.45% | 82.58% |
| Final Val Accuracy (epoch 10) | 68.18% | 79.98% |
| Parameters | 315,722 | 1,118,762 |

Model 2 improved test accuracy by **+12.16 percentage points** over the baseline.

### 4.2 Training Curve Analysis

**Model 1** exhibits a classic overfitting signature. Training accuracy climbs continuously to 80.45% by epoch 10, while validation accuracy peaks around epoch 5–6 (~70%) and then declines to 68.18%. The validation loss follows the same pattern — it bottoms out around epoch 4 (~0.90) and rises to 1.04 by epoch 10 as the model memorizes training data rather than generalizing. Without any regularization, the model's 315K parameters are sufficient to overfit the 45,000 training samples.

**Model 2** shows healthy generalization. Training and validation accuracy track closely throughout all 10 epochs. Validation accuracy ends at 79.98%, only 2.6 points below training accuracy (82.58%). Validation loss continues declining through the final epoch (0.60), with no divergence from training loss. BatchNormalization and Dropout collectively prevent the overfitting seen in Model 1 even as the model parameter count increases to 1.1M.

### 4.3 Per-Class Performance (Model 2)

| Class | Precision | Recall | F1 | Notes |
|:---|:---:|:---:|:---:|:---|
| automobile | 0.92 | 0.87 | 0.89 | Easiest — distinctive boxy shape |
| ship | 0.84 | 0.91 | 0.88 | Strong ocean background cue |
| truck | 0.85 | 0.90 | 0.87 | Similar structural cues to automobile |
| frog | 0.86 | 0.79 | 0.83 | Distinctive color and shape |
| horse | 0.78 | 0.87 | 0.82 | Consistent silhouette |
| airplane | 0.84 | 0.81 | 0.82 | Recognizable outline |
| deer | 0.78 | 0.74 | 0.76 | Some confusion with horse |
| bird | 0.83 | 0.60 | 0.70 | Low recall — misclassified as dog/frog |
| dog | 0.59 | 0.81 | 0.68 | High recall but low precision (catches many cat) |
| cat | 0.62 | 0.54 | 0.58 | Hardest class — visually similar to dog |

The **cat–dog confusion** is the most notable failure pattern in both models. In Model 2, 267 of 1,000 actual cats were predicted as dogs — the single largest off-diagonal value in the confusion matrix. This is a well-documented challenge in CIFAR-10: at 32×32 resolution, cats and dogs share similar textures (fur), color distributions, and general proportions. The model struggles to distinguish them without higher resolution or more training data.

**Bird** also has below-average recall (60%), with misclassifications distributed across dog (96), deer (67), and frog (60). At 32×32, birds in various positions (perched, in flight) can look similar to small quadrupeds or frogs.

### 4.4 Confusion Matrix Observations

**Model 1 confusion highlights:**
- Airplane → Ship: 140 misclassifications (both have broad, flat shapes)
- Cat → Dog: 207 (texture similarity)
- Bird → Dog: 141 (small animal confusion at low resolution)

**Model 2 confusion highlights:**
- Cat → Dog: 267 (improved cat recall but dog precision suffers)
- Bird → Dog: 96 (reduced from 141)
- Airplane → Ship: 64 (reduced from 140 — padding='same' preserves more spatial detail)

The reduction in airplane→ship confusion from 140 to 64 is a direct effect of `padding='same'` — by preserving the full 32×32 spatial detail through each Conv layer, the network retains finer shape cues that help it distinguish the elongated airplane silhouette from the broader ship profile.

---

## 5. Sample Predictions — Unseen Test Images

Twenty images from the test set (never seen during training) were classified by both models. Model 2 correctly classified **17 of 20** (85%), consistent with its overall 78.37% test accuracy on the full 10,000-image test set.

The sample prediction visualization (`sample_predictions.png`) shows each image labeled with the true class and both model predictions, color-coded green (Model 2 correct) or red (Model 2 wrong). The three errors in the sample reflect the expected failure modes: visually ambiguous small animals at low resolution.

---

## 6. Conclusion

The experiment demonstrates a clear relationship between architectural choices and model performance on image classification:

1. **Regularization is essential.** Model 1 achieves 80% training accuracy but only 66% on the test set — a 14-point gap that indicates overfitting. Adding Dropout and BatchNormalization in Model 2 closes this gap to 4 points (83% train vs 78% test) while simultaneously improving absolute test accuracy by 12 points.

2. **Deeper feature extraction improves generalization.** The VGG-style double Conv blocks allow the network to learn hierarchical features — edges and textures in the first block, shapes and parts in the second — before spatial information is compressed by MaxPooling. This structural improvement is reflected in higher recall across nearly every class.

3. **Spatial preservation matters.** `padding='same'` prevents the feature map from shrinking on every Conv pass, preserving fine spatial details that help distinguish visually similar classes (airplane vs. ship).

4. **Some confusion is irreducible at this resolution.** Cat and dog remain the hardest pair to separate at 32×32 pixels. The ambiguity is fundamentally a data resolution constraint, not a model design failure. Addressing it would require higher-resolution training data or learned augmentations.

Model 2's 78.37% accuracy in 10 epochs on CPU-only hardware represents a strong result for this architecture class. Further gains would require more epochs, data augmentation (random flips, crops), or a third conv block — all viable next iterations beyond the scope of this lab.

---

## 7. Output Files

| File | Description |
|:---|:---|
| `model1_training_curves.png` | Model 1 accuracy and loss curves over 10 epochs |
| `model2_training_curves.png` | Model 2 accuracy and loss curves over 10 epochs |
| `comparison_training_curves.png` | Side-by-side 2×2 comparison of both models |
| `model1_confusion_matrix.png` | 10×10 confusion matrix — Baseline CNN (66.21%) |
| `model2_confusion_matrix.png` | 10×10 confusion matrix — Improved CNN (78.37%) |
| `sample_predictions.png` | 20 unseen test images with true and predicted labels |

---

## Appendix — Full Console Output

```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1773457467.544637   13976 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
======================================================================
STEP 1: Loading and Preprocessing CIFAR-10
======================================================================
Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step
x_train shape: (50000, 32, 32, 3)  (samples, H, W, C)
x_test  shape: (10000, 32, 32, 3)
y_train shape: (50000, 1)  (samples, 1) — integer labels 0-9
Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
Train samples: 50,000  |  Test samples: 10,000

Label value range: 0 – 9
Class counts in test set:
  0 (airplane    ): 1000
  1 (automobile  ): 1000
  2 (bird        ): 1000
  3 (cat         ): 1000
  4 (deer        ): 1000
  5 (dog         ): 1000
  6 (frog        ): 1000
  7 (horse       ): 1000
  8 (ship        ): 1000
  9 (truck       ): 1000

======================================================================
MODEL 1: Baseline CNN
======================================================================

Architecture:
  Conv2D(32, (3,3), relu)   → (30,30,32)
  MaxPooling2D(2,2)          → (15,15,32)
  Conv2D(64, (3,3), relu)   → (13,13,64)
  MaxPooling2D(2,2)          → (6,6,64)
  Flatten()                  → 2304
  Dense(128, relu)
  Dense(10, softmax)

I0000 00:00:1773457488.302644   13976 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
Model: "Model1_Baseline"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 30, 30, 32)          │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 15, 15, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 13, 13, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 6, 6, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 2304)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │         295,040 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 315,722 (1.20 MB)
 Trainable params: 315,722 (1.20 MB)
 Non-trainable params: 0 (0.00 B)

Training Model 1 (epochs=10, batch_size=64, val_split=0.1)...
Epoch 1/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 7s 7ms/step - accuracy: 0.4630 - loss: 1.5019 - val_accuracy: 0.5612 - val_loss: 1.2314
Epoch 2/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 4s 6ms/step - accuracy: 0.5947 - loss: 1.1509 - val_accuracy: 0.6376 - val_loss: 1.0421
Epoch 3/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 4s 6ms/step - accuracy: 0.6466 - loss: 1.0089 - val_accuracy: 0.6676 - val_loss: 0.9759
Epoch 4/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 5s 6ms/step - accuracy: 0.6823 - loss: 0.9135 - val_accuracy: 0.6862 - val_loss: 0.9257
Epoch 5/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 4s 6ms/step - accuracy: 0.7103 - loss: 0.8370 - val_accuracy: 0.6982 - val_loss: 0.8996
Epoch 6/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 4s 6ms/step - accuracy: 0.7348 - loss: 0.7678 - val_accuracy: 0.7026 - val_loss: 0.8842
Epoch 7/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 4s 6ms/step - accuracy: 0.7546 - loss: 0.7091 - val_accuracy: 0.6998 - val_loss: 0.9094
Epoch 8/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 5s 6ms/step - accuracy: 0.7721 - loss: 0.6571 - val_accuracy: 0.6932 - val_loss: 0.9316
Epoch 9/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 5s 6ms/step - accuracy: 0.7895 - loss: 0.6093 - val_accuracy: 0.6924 - val_loss: 0.9773
Epoch 10/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 5s 7ms/step - accuracy: 0.8045 - loss: 0.5644 - val_accuracy: 0.6818 - val_loss: 1.0436

--- Model 1 Evaluation ---
Test Accuracy: 0.6621
Test Loss:     1.0848

Classification Report:
              precision    recall  f1-score   support

    airplane       0.80      0.62      0.70      1000
  automobile       0.76      0.81      0.79      1000
        bird       0.70      0.46      0.55      1000
         cat       0.44      0.48      0.46      1000
        deer       0.63      0.60      0.62      1000
         dog       0.51      0.61      0.56      1000
        frog       0.84      0.60      0.70      1000
       horse       0.69      0.75      0.72      1000
        ship       0.69      0.84      0.76      1000
       truck       0.67      0.83      0.74      1000

    accuracy                           0.66     10000
   macro avg       0.67      0.66      0.66     10000
weighted avg       0.67      0.66      0.66     10000

Confusion Matrix (10×10):
[[623  41  34  28  24  10   4  10 140  86]
 [  9 812   4  10   1   5   5   1  35 118]
 [ 48  16 455  92  94 141  35  58  34  27]
 [ 14  17  26 482  79 207  32  63  40  40]
 [ 20  12  34  86 602  73  25 100  30  18]
 [  9   8  31 177  35 613   8  74  23  22]
 [  4  26  25 140  59  75 603  16  28  24]
 [  8  10  22  39  51  63   3 753  14  37]
 [ 35  32  11  15   4   6   1   9 845  42]
 [  7  89   6  16   4   2   1  14  28 833]]
Saved: model1_training_curves.png
Saved: model1_confusion_matrix.png

======================================================================
MODEL 2: Improved CNN
======================================================================

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

Model: "Model2_Improved"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_2 (Conv2D)                    │ (None, 32, 32, 32)          │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization                  │ (None, 32, 32, 32)          │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 32, 32, 32)          │           9,248 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_1                │ (None, 32, 32, 32)          │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 16, 16, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 16, 16, 32)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_4 (Conv2D)                    │ (None, 16, 16, 64)          │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_2                │ (None, 16, 16, 64)          │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_5 (Conv2D)                    │ (None, 16, 16, 64)          │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_3                │ (None, 16, 16, 64)          │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (MaxPooling2D)       │ (None, 8, 8, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 8, 8, 64)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_1 (Flatten)                  │ (None, 4096)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 256)                 │       1,048,832 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_4                │ (None, 256)                 │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 10)                  │           2,570 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,118,762 (4.27 MB)
 Trainable params: 1,117,866 (4.26 MB)
 Non-trainable params: 896 (3.50 KB)

Training Model 2 (epochs=10, batch_size=64, val_split=0.1)...
Epoch 1/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 28s 37ms/step - accuracy: 0.4864 - loss: 1.5123 - val_accuracy: 0.5824 - val_loss: 1.1891
Epoch 2/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 26s 37ms/step - accuracy: 0.6404 - loss: 1.0181 - val_accuracy: 0.6868 - val_loss: 0.9093
Epoch 3/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 27s 38ms/step - accuracy: 0.6934 - loss: 0.8775 - val_accuracy: 0.6838 - val_loss: 0.9282
Epoch 4/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 26s 37ms/step - accuracy: 0.7256 - loss: 0.7843 - val_accuracy: 0.7264 - val_loss: 0.8111
Epoch 5/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 27s 38ms/step - accuracy: 0.7460 - loss: 0.7212 - val_accuracy: 0.7666 - val_loss: 0.6699
Epoch 6/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 27s 38ms/step - accuracy: 0.7656 - loss: 0.6640 - val_accuracy: 0.7356 - val_loss: 0.7682
Epoch 7/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 26s 37ms/step - accuracy: 0.7795 - loss: 0.6268 - val_accuracy: 0.7566 - val_loss: 0.7012
Epoch 8/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 26s 37ms/step - accuracy: 0.8000 - loss: 0.5658 - val_accuracy: 0.7634 - val_loss: 0.6954
Epoch 9/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 27s 39ms/step - accuracy: 0.8109 - loss: 0.5320 - val_accuracy: 0.7558 - val_loss: 0.7370
Epoch 10/10
704/704 ━━━━━━━━━━━━━━━━━━━━ 28s 39ms/step - accuracy: 0.8258 - loss: 0.4937 - val_accuracy: 0.7998 - val_loss: 0.6005

--- Model 2 Evaluation ---
Test Accuracy: 0.7837
Test Loss:     0.6404

Classification Report:
              precision    recall  f1-score   support

    airplane       0.84      0.81      0.82      1000
  automobile       0.92      0.87      0.89      1000
        bird       0.83      0.60      0.70      1000
         cat       0.62      0.54      0.58      1000
        deer       0.78      0.74      0.76      1000
         dog       0.59      0.81      0.68      1000
        frog       0.86      0.79      0.83      1000
       horse       0.78      0.87      0.82      1000
        ship       0.84      0.91      0.88      1000
       truck       0.85      0.90      0.87      1000

    accuracy                           0.78     10000
   macro avg       0.79      0.78      0.78     10000
weighted avg       0.79      0.78      0.78     10000

Confusion Matrix (10×10):
[[808  12  25  19  10   8   3  16  64  35]
 [  9 870   2   6   0   5   2   1  21  84]
 [ 54   3 599  55  67  96  60  45  16   5]
 [ 16   5  28 543  49 267  30  34  15  13]
 [  9   1  28  47 738  50  29  83  14   1]
 [  6   1  10  99  23 809   3  45   1   3]
 [  5   2  17  75  32  57 793   5   9   5]
 [  7   1   9  18  22  68   0 870   1   4]
 [ 38  19   4   5   1   5   1   6 911  10]
 [ 14  36   1   7   1   3   1  13  28 896]]
Saved: model2_training_curves.png
Saved: model2_confusion_matrix.png

======================================================================
STEP 4: Sample Predictions — 20 Unseen Test Images
======================================================================
  Showing 20 test images (indices 0-19)
  Model 2 correct on these samples: 17/20
Saved: sample_predictions.png

======================================================================
STEP 5: Comparison Training Curves
======================================================================
Saved: comparison_training_curves.png

======================================================================
COMPARISON: Model 1 (Baseline) vs Model 2 (Improved)
======================================================================

+-----------------------------+----------------------+----------------------+
| Metric                      | Model 1 (Baseline)   | Model 2 (Improved)   |
+-----------------------------+----------------------+----------------------+
| Test Accuracy               | 0.6621               | 0.7837               |
| Test Loss                   | 1.0848               | 0.6404               |
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

======================================================================
ANALYSIS COMPLETE
======================================================================
Output files generated:
  model1_training_curves.png
  model2_training_curves.png
  comparison_training_curves.png
  model1_confusion_matrix.png
  model2_confusion_matrix.png
  sample_predictions.png
```
