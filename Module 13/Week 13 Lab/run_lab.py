"""
BFOR 516 — Week 13 Lab: Transformer Model for Spam Email Detection
Standalone runner — captures all output to raw_output.txt
Run with: python run_lab.py
"""

import os
import sys
import io
import contextlib

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # suppress TF C++ info logs

# ── Redirect stdout to both terminal and file ─────────────────────────────────
class SafeStream:
    """Wraps a stream and replaces unencodable characters instead of crashing."""
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        try:
            self.stream.write(data)
        except (UnicodeEncodeError, UnicodeDecodeError):
            safe = data.encode(self.stream.encoding or "utf-8", errors="replace").decode(
                self.stream.encoding or "utf-8", errors="replace"
            )
            self.stream.write(safe)
        self.stream.flush()
    def flush(self):
        self.stream.flush()

class Tee:
    def __init__(self, *streams):
        self.streams = [SafeStream(s) for s in streams]
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

log_file = open("raw_output.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)

def section(title):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD LIBRARIES
# ─────────────────────────────────────────────────────────────────────────────
section("1. LOAD LIBRARIES")

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI window needed)
import matplotlib.pyplot as plt
import seaborn as sns
import re

np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version : {tf.__version__}")
print(f"GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD AND INSPECT DATA
# ─────────────────────────────────────────────────────────────────────────────
section("2. LOAD AND INSPECT DATA")

df = pd.read_csv("SpamAssasin.csv", encoding="utf-8", on_bad_lines="skip")

print(f"Dataset shape      : {df.shape}")
print(f"\nColumns            : {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())
print("\nLabel distribution (0=ham, 1=spam):")
print(df["label"].value_counts())
print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True).round(3))

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────
section("3. CLEAN DATA")

df["email_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\S+@\S+", "[EMAIL]", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

df["email_text"] = df["email_text"].apply(clean_text)
df = df.dropna(subset=["label"])
df = df[df["email_text"].str.strip() != ""]
df["label"] = df["label"].astype(int)

print(f"Final dataset shape after cleaning: {df.shape}")
print("\nSample cleaned email (first 300 chars):")
print(df["email_text"].iloc[0][:300])

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
section("4. MODEL PARAMETERS")

MAX_VOCAB            = 8000
MAX_SEQUENCE_LEN     = 200
EMBED_DIM            = 64
NUM_HEADS            = 4
FF_DIM               = 128
NUM_TRANSFORMER_BLOCKS = 2
DROPOUT_RATE         = 0.3
BATCH_SIZE           = 32
EPOCHS               = 15

params = {
    "MAX_VOCAB": MAX_VOCAB, "MAX_SEQUENCE_LEN": MAX_SEQUENCE_LEN,
    "EMBED_DIM": EMBED_DIM, "NUM_HEADS": NUM_HEADS, "FF_DIM": FF_DIM,
    "NUM_TRANSFORMER_BLOCKS": NUM_TRANSFORMER_BLOCKS,
    "DROPOUT_RATE": DROPOUT_RATE, "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS,
}
for k, v in params.items():
    print(f"  {k:26s} = {v}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TEXT VECTORIZATION
# ─────────────────────────────────────────────────────────────────────────────
section("5. TEXT VECTORIZATION")

text_vectorizer = layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=MAX_SEQUENCE_LEN,
    standardize="lower_and_strip_punctuation",
)
text_vectorizer.adapt(df["email_text"].values)

print(f"Vocabulary size                    : {len(text_vectorizer.get_vocabulary())}")
print(f"Sample vocabulary (first 20 tokens): {text_vectorizer.get_vocabulary()[:20]}")

X = text_vectorizer(df["email_text"].values).numpy()
y = df["label"].values.astype(np.float32)

print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}, dtype: {y.dtype}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples  : {len(X_train)}")
print(f"Testing samples   : {len(X_test)}")
print(f"Train class dist  (ham/spam): {np.bincount(y_train.astype(int))}")
print(f"Test  class dist  (ham/spam): {np.bincount(y_test.astype(int))}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASS IMBALANCE
# ─────────────────────────────────────────────────────────────────────────────
section("6. CLASS IMBALANCE")

HAM_LABEL  = "Ham (0)"
SPAM_LABEL = "Spam (1)"

counts = np.bincount(y_train.astype(int))
class_weight = {
    0: len(y_train) / (2 * counts[0]),
    1: len(y_train) / (2 * counts[1]),
}

print(f"Ham  count : {counts[0]}  | weight: {class_weight[0]:.4f}")
print(f"Spam count : {counts[1]}  | weight: {class_weight[1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar([HAM_LABEL, SPAM_LABEL], counts, color=["steelblue", "tomato"], edgecolor="black")
axes[0].set_title("Training Set Class Distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")
axes[1].bar([HAM_LABEL, SPAM_LABEL],
            [class_weight[0], class_weight[1]],
            color=["steelblue", "tomato"], edgecolor="black")
axes[1].set_title("Class Weights Applied During Training")
axes[1].set_ylabel("Weight")
for i, v in enumerate([class_weight[0], class_weight[1]]):
    axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: class_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. BUILD TRANSFORMER MODEL
# ─────────────────────────────────────────────────────────────────────────────
section("7. BUILD TRANSFORMER MODEL")

class PositionalEmbedding(layers.Layer):
    """Token embedding + learned positional embedding."""
    def __init__(self, seq_len, vocab_size, embed_dim, **kw):
        super().__init__(**kw)
        self.tok = layers.Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos = layers.Embedding(seq_len, embed_dim)

    def call(self, x):
        positions = tf.range(tf.shape(x)[-1])
        return self.tok(x) + self.pos(positions)


def transformer_block(x, embed_dim, num_heads, ff_dim, dropout):
    """One encoder block: self-attention + FFN with residual + LayerNorm."""
    a = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout,
    )(x, x)
    a = layers.Dropout(dropout)(a)
    x = layers.LayerNormalization(epsilon=1e-6)(x + a)
    f = layers.Dense(ff_dim, activation="relu")(x)
    f = layers.Dense(embed_dim)(f)
    f = layers.Dropout(dropout)(f)
    return layers.LayerNormalization(epsilon=1e-6)(x + f)


def build_model():
    inp = layers.Input(shape=(MAX_SEQUENCE_LEN,), name="token_ids")
    x = PositionalEmbedding(MAX_SEQUENCE_LEN, MAX_VOCAB, EMBED_DIM, name="pos_embedding")(inp)
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_block(x, EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT_RATE)
    x = layers.GlobalAveragePooling1D(name="avg_pool")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(32, activation="relu", name="classifier_dense")(x)
    out = layers.Dense(1, activation="sigmoid", name="spam_prob")(x)
    return Model(inp, out, name="SpamTransformer")


model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)
model.summary()

# ─────────────────────────────────────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────────────────────────────────────
section("8. TRAIN MODEL")

cbs = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cbs,
    class_weight=class_weight,
    verbose=1,
)

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Training History", fontsize=14, fontweight="bold")
metrics_to_plot = [
    ("loss",     "Loss",     "tomato"),
    ("accuracy", "Accuracy", "steelblue"),
    ("auc",      "AUC",      "seagreen"),
]
for ax, (metric, label, color) in zip(axes, metrics_to_plot):
    epochs_ran = range(1, len(history.history[metric]) + 1)
    ax.plot(epochs_ran, history.history[metric],
            label=f"Train {label}", color=color, linewidth=2)
    ax.plot(epochs_ran, history.history[f"val_{metric}"],
            label=f"Val {label}", color=color, linewidth=2, linestyle="--")
    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: training_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
section("9. EVALUATION")

y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

print("=" * 55)
print("Keras evaluate on test set:")
print("=" * 55)
test_results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
for name, val in zip(model.metrics_names, test_results):
    print(f"  {name:12s}: {val:.4f}")

print("\n" + "=" * 55)
print("Classification Report (threshold = 0.50):")
print("=" * 55)
print(classification_report(y_test, y_pred, target_names=[HAM_LABEL, SPAM_LABEL]))

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"], ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
axes[0].set_ylabel("True Label")
axes[0].set_xlabel("Predicted Label")
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
            xticklabels=["Predicted Ham", "Predicted Spam"],
            yticklabels=["Actual Ham", "Actual Spam"], ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized)", fontweight="bold")
axes[1].set_ylabel("True Label")
axes[1].set_xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrix.png")

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue  Negatives (ham correctly classified)  : {tn}")
print(f"False Positives (ham misclassified as spam) : {fp}")
print(f"False Negatives (spam misclassified as ham) : {fn}")
print(f"True  Positives (spam correctly classified) : {tp}")

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="steelblue", linewidth=2.5,
         label=f"Transformer (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC = 0.50)")
plt.fill_between(fpr, tpr, alpha=0.15, color="steelblue")
plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=12)
plt.title("ROC Curve -- Spam Detection", fontsize=13, fontweight="bold")
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: roc_curve.png")

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal decision threshold (Youden's J): {optimal_threshold:.4f}")
print(f"  TPR at optimal : {tpr[optimal_idx]:.4f}")
print(f"  FPR at optimal : {fpr[optimal_idx]:.4f}")

# Confidence distribution
plt.figure(figsize=(10, 5))
plt.hist(y_prob[y_test == 0], bins=50, alpha=0.65,
         color="steelblue", label="Ham (true label 0)", edgecolor="white")
plt.hist(y_prob[y_test == 1], bins=50, alpha=0.65,
         color="tomato", label="Spam (true label 1)", edgecolor="white")
plt.axvline(0.5, color="black", linestyle="--", label="Default threshold (0.50)")
plt.axvline(optimal_threshold, color="gold", linestyle="--",
            label=f"Optimal threshold ({optimal_threshold:.2f})")
plt.xlabel("Predicted Spam Probability", fontsize=12)
plt.ylabel("Email Count", fontsize=12)
plt.title("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("confidence_distribution.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: confidence_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# 10. PERFORMANCE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
section("10. PERFORMANCE SUMMARY TABLE")

y_pred_opt = (y_prob >= optimal_threshold).astype(int)
summary = pd.DataFrame({
    "Threshold" : [0.5, optimal_threshold],
    "Accuracy"  : [accuracy_score(y_test, y_pred),  accuracy_score(y_test, y_pred_opt)],
    "Precision" : [precision_score(y_test, y_pred), precision_score(y_test, y_pred_opt)],
    "Recall"    : [recall_score(y_test, y_pred),    recall_score(y_test, y_pred_opt)],
    "F1-Score"  : [f1_score(y_test, y_pred),        f1_score(y_test, y_pred_opt)],
    "ROC-AUC"   : [roc_auc, roc_auc],
})
summary = summary.round(4)
summary.index = ["Default (0.50)", f"Optimal ({optimal_threshold:.2f})"]
print("\nPerformance Summary:")
print(summary.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 11. NEW EMAIL PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
section("11. NEW EMAIL PREDICTIONS")

def predict_email(email_texts, threshold=0.5):
    if isinstance(email_texts, str):
        email_texts = [email_texts]
    cleaned    = [clean_text(t) for t in email_texts]
    vectorized = text_vectorizer(np.array(cleaned)).numpy()
    probs      = model.predict(vectorized, verbose=0).flatten()
    return pd.DataFrame({
        "email_preview"    : [t[:80] + "..." if len(t) > 80 else t for t in email_texts],
        "spam_probability" : probs.round(4),
        "prediction"       : ["SPAM" if p >= threshold else "HAM" for p in probs],
        "confidence"       : [f"{max(p, 1-p)*100:.1f}%" for p in probs],
    })

test_emails = [
    "CONGRATULATIONS! You've been selected as our WINNER! Click here now to claim "
    "your FREE prize of $1,000,000!!! Limited time offer -- ACT NOW!!!",
    "Hi John, just following up on our meeting from yesterday. Could you send me "
    "the agenda for next Tuesday's project review? Thanks.",
    "Dear valued customer, your account has been suspended. Verify your identity "
    "immediately by clicking the link below or your account will be permanently deleted.",
    "This week in security: patch Tuesday roundup, CVE analysis, and our monthly "
    "threat intelligence summary from the research team.",
    "Exclusive offer for our members -- 50% off all premium subscriptions this weekend only.",
]

print("\nPredictions — default threshold (0.50):")
print(predict_email(test_emails, threshold=0.5).to_string(index=False))

print(f"\nPredictions — optimal threshold ({optimal_threshold:.2f}):")
print(predict_email(test_emails, threshold=optimal_threshold).to_string(index=False))

section("COMPLETE")
print("raw_output.txt  — full console output")
print("class_distribution.png")
print("training_curves.png")
print("confusion_matrix.png")
print("roc_curve.png")
print("confidence_distribution.png")

log_file.close()
sys.stdout = sys.__stdout__
print("\nAll output saved to raw_output.txt")
