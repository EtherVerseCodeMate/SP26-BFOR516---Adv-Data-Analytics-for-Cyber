"""
Week 6 Lab: Neural Networks - Truth Seeker Dataset
BFOR516 - Advanced Data Analytics for Cyber
Student: Spencer Kone
Date: March 8, 2026

AI Usage Statement:
- AI tools (Antigravity/Gemini) were used for coding assistance only.

Goal:
Build two NN models to classify tweets as true or false:
  Model 1: Linguistic features (text + numeric)
  Model 2: User behavior features (numeric)
"""

import os
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

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, issparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def log(msg):
    """Flush-safe print for real-time output."""
    print(msg)
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# 1. DATA LOADING — Robust path with fallback
# ---------------------------------------------------------------------------
log("=" * 70)
log("STEP 1: Loading Truth Seeker Dataset")
log("=" * 70)

# Try multiple possible locations for the dataset
possible_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'Week 4', 'Truth_Seeker_Dataset.csv'),
    os.path.join(os.path.dirname(__file__), 'Truth_Seeker_Dataset.csv'),
    'Truth_Seeker_Dataset.csv',
    os.path.join('..', 'Week 4', 'Truth_Seeker_Dataset.csv'),
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        log(f"Loaded dataset from: {os.path.abspath(path)}")
        break

if df is None:
    log("ERROR: Could not find 'Truth_Seeker_Dataset.csv'. Searched:")
    for p in possible_paths:
        log(f"  - {os.path.abspath(p)}")
    sys.exit(1)

log(f"Dataset shape: {df.shape}")
log(f"Columns ({len(df.columns)}): {list(df.columns)}")

# ---------------------------------------------------------------------------
# 2. DATA CLEANING — Handle NaN values
# ---------------------------------------------------------------------------
log(f"\nMissing values before cleaning:")
missing = df.isnull().sum()
log(missing[missing > 0].to_string() if missing.any() else "  None")

# Target column
TARGET = 'BinaryNumTarget'
df = df.dropna(subset=[TARGET])
df = df.reset_index(drop=True)   # ensure 0-based positional indexing
log(f"\nTarget distribution:")
log(f"  True  (1.0): {(df[TARGET] == 1.0).sum()}")
log(f"  False (0.0): {(df[TARGET] == 0.0).sum()}")

# =========================================================================
#
#  MODEL 1: LINGUISTIC FEATURES (Text + Numeric)
#
# =========================================================================
log("\n" + "=" * 70)
log("MODEL 1: Neural Network on Linguistic Features")
log("=" * 70)

# ---- Feature lists (as specified in the lab) ----
text_features = ['tweet', 'statement']
numeric_linguistic_features = [
    'unique_count', 'total_count',
    'Max word length', 'Min word length', 'Average word length',
    'present_verbs', 'past_verbs', 'adjectives',
    'pronouns', 'TOs', 'determiners', 'conjunctions',
    'dots', 'exclamation', 'questions', 'ampersand',
    'capitals', 'digits', 'long_word_freq', 'short_word_freq'
]

# ---- 2a. Raw arrays + train/test split FIRST (prevents data leakage) ----
log("\nSplitting raw data before any fitting...")
df['tweet'] = df['tweet'].fillna('')
df['statement'] = df['statement'].fillna('')

from scipy.sparse import csr_matrix

y = df[TARGET].values
tweet_raw     = df['tweet'].values
statement_raw = df['statement'].values
X_numeric_ling = df[numeric_linguistic_features].fillna(0).values

# Split by UNIQUE STATEMENT — prevents the same fact-checked claim from appearing
# in both train and test (the dataset has many rows per statement; row-level splits
# allow identical statement text to leak across the boundary).
unique_stmts = np.unique(statement_raw)
train_stmts, test_stmts = train_test_split(
    unique_stmts, test_size=0.2, random_state=42
)
train_stmts_set = set(train_stmts)
test_stmts_set  = set(test_stmts)

stmt_series = pd.Series(statement_raw)
idx_train = stmt_series[stmt_series.isin(train_stmts_set)].index.values
idx_test  = stmt_series[stmt_series.isin(test_stmts_set)].index.values
y_train1  = y[idx_train]
y_test1   = y[idx_test]

log(f"  Unique statements total: {len(unique_stmts)}")
log(f"  Train statements:        {len(train_stmts)}  ({len(idx_train):,} rows)")
log(f"  Test statements:         {len(test_stmts)}  ({len(idx_test):,} rows)")
log(f"  Train class balance: True={y_train1.sum():.0f}, False={(1-y_train1).sum():.0f}")

# ---- 2b. TF-IDF fit on TRAINING data only, then transform test ----
log("\nFitting TF-IDF on training data only...")
tfidf_tweet     = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_statement = TfidfVectorizer(max_features=500, stop_words='english')

X_tweet_train = tfidf_tweet.fit_transform(tweet_raw[idx_train])
X_tweet_test  = tfidf_tweet.transform(tweet_raw[idx_test])

X_stmt_train = tfidf_statement.fit_transform(statement_raw[idx_train])
X_stmt_test  = tfidf_statement.transform(statement_raw[idx_test])

log(f"  tweet TF-IDF shape (train):     {X_tweet_train.shape}")
log(f"  statement TF-IDF shape (train): {X_stmt_train.shape}")

# ---- 2c. Scale numeric features on TRAINING data only ----
log("Fitting StandardScaler on training data only...")
scaler_m1 = StandardScaler()
X_num_train = scaler_m1.fit_transform(X_numeric_ling[idx_train])
X_num_test  = scaler_m1.transform(X_numeric_ling[idx_test])
log(f"  Numeric features shape: {X_num_train.shape}")

# ---- 2d. Combine TF-IDF + scaled numeric ----
X_train1 = hstack([X_tweet_train, X_stmt_train, csr_matrix(X_num_train)]).toarray()
X_test1  = hstack([X_tweet_test,  X_stmt_test,  csr_matrix(X_num_test)]).toarray()

log(f"\nModel 1 feature matrix: train={X_train1.shape}, test={X_test1.shape}")

# ---- 2e. Build Model 1 (4 hidden layers — sized for ~1,020 features) ----
log("\nBuilding Model 1 architecture...")
model1 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train1.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # Binary classification output
])

model1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model1.summary(print_fn=log)

# ---- 2f. Train Model 1 ----
log("\nTraining Model 1 (epochs=50, batch_size=64)...")
history1 = model1.fit(
    X_train1, y_train1,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ---- 2g. Evaluate Model 1 ----
log("\n--- Model 1 Evaluation ---")
test_loss1, test_acc1 = model1.evaluate(X_test1, y_test1, verbose=0)
y_pred1 = (model1.predict(X_test1, verbose=0) > 0.5).astype(int).flatten()

log(f"Test Accuracy: {test_acc1:.4f}")
log(f"Test Loss:     {test_loss1:.4f}")

log("\nClassification Report:")
log(classification_report(y_test1, y_pred1, target_names=['False/Fake', 'True']))

cm1 = confusion_matrix(y_test1, y_pred1)
log(f"\nConfusion Matrix:\n{cm1}")

# ---- 2h. Plot Model 1 — Confusion Matrix ----
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted False', 'Predicted True'],
            yticklabels=['Actual False', 'Actual True'], ax=ax)
ax.set_title(f'Model 1 (Linguistic) — Confusion Matrix\nAccuracy: {test_acc1:.4f}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('model1_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: model1_confusion_matrix.png")

# ---- 2i. Plot Model 1 — Training Curves ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history1.history['accuracy'], label='Train Accuracy')
ax1.plot(history1.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model 1 — Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history1.history['loss'], label='Train Loss')
ax2.plot(history1.history['val_loss'], label='Validation Loss')
ax2.set_title('Model 1 — Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model1_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: model1_training_curves.png")


# =========================================================================
#
#  MODEL 2: USER BEHAVIOR FEATURES (Numeric only)
#
# =========================================================================
log("\n" + "=" * 70)
log("MODEL 2: Neural Network on User Behavior Features")
log("=" * 70)

behavior_features = [
    'followers_count', 'friends_count', 'favourites_count',
    'statuses_count', 'BotScore', 'cred',
    'normalize_influence', 'replies', 'retweets'
]

# ---- 3a. Prepare features ----
log(f"\nBehavior features ({len(behavior_features)}): {behavior_features}")
X_behavior = df[behavior_features].fillna(0).values

# ---- 3b. Same statement-based split as Model 1 — consistent evaluation ----
# Scaler fit on training rows only (no leakage).
scaler_m2 = StandardScaler()
X_train2 = scaler_m2.fit_transform(X_behavior[idx_train])
X_test2  = scaler_m2.transform(X_behavior[idx_test])
y_train2 = y[idx_train]
y_test2  = y[idx_test]
log("StandardScaler fit on training data only.")
log(f"  Training samples: {X_train2.shape[0]:,}")
log(f"  Test samples:     {X_test2.shape[0]:,}")

# ---- 3c. Build Model 2 (4 hidden layers — scaled down for 9 features) ----
log("\nBuilding Model 2 architecture...")
model2 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train2.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model2.summary(print_fn=log)

# ---- 3d. Train Model 2 ----
log("\nTraining Model 2 (epochs=50, batch_size=64)...")
history2 = model2.fit(
    X_train2, y_train2,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ---- 3e. Evaluate Model 2 ----
log("\n--- Model 2 Evaluation ---")
test_loss2, test_acc2 = model2.evaluate(X_test2, y_test2, verbose=0)
y_pred2 = (model2.predict(X_test2, verbose=0) > 0.5).astype(int).flatten()

log(f"Test Accuracy: {test_acc2:.4f}")
log(f"Test Loss:     {test_loss2:.4f}")

log("\nClassification Report:")
log(classification_report(y_test2, y_pred2, target_names=['False/Fake', 'True']))

cm2 = confusion_matrix(y_test2, y_pred2)
log(f"\nConfusion Matrix:\n{cm2}")

# ---- 3f. Plot Model 2 — Confusion Matrix ----
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Predicted False', 'Predicted True'],
            yticklabels=['Actual False', 'Actual True'], ax=ax)
ax.set_title(f'Model 2 (Behavior) — Confusion Matrix\nAccuracy: {test_acc2:.4f}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('model2_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: model2_confusion_matrix.png")

# ---- 3g. Plot Model 2 — Training Curves ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history2.history['accuracy'], label='Train Accuracy')
ax1.plot(history2.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model 2 — Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history2.history['loss'], label='Train Loss')
ax2.plot(history2.history['val_loss'], label='Validation Loss')
ax2.set_title('Model 2 — Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model2_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: model2_training_curves.png")


# =========================================================================
#
#  COMPARISON SUMMARY
#
# =========================================================================
log("\n" + "=" * 70)
log("COMPARISON: Model 1 (Linguistic) vs Model 2 (User Behavior)")
log("=" * 70)

log(f"""
+---------------------------+------------------+------------------+
| Metric                    | Model 1 (Ling.)  | Model 2 (Behav.) |
+---------------------------+------------------+------------------+
| Test Accuracy             | {test_acc1:.4f}           | {test_acc2:.4f}           |
| Test Loss                 | {test_loss1:.4f}           | {test_loss2:.4f}           |
| Input Features            | {X_train1.shape[1]:>6}           | {X_train2.shape[1]:>6}           |
| Hidden Layers             |      4           |      4           |
| Architecture              | 256->128->64->32 | 64->32->16->8    |
| Epochs                    |     50           |     50           |
| Batch Size                |     64           |     64           |
+---------------------------+------------------+------------------+
""")

# Side-by-side confusion matrix plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred False', 'Pred True'],
            yticklabels=['Actual False', 'Actual True'], ax=ax1)
ax1.set_title(f'Model 1 (Linguistic)\nAccuracy: {test_acc1:.4f}')

sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Pred False', 'Pred True'],
            yticklabels=['Actual False', 'Actual True'], ax=ax2)
ax2.set_title(f'Model 2 (Behavior)\nAccuracy: {test_acc2:.4f}')

plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: comparison_confusion_matrices.png")

log("\n" + "=" * 70)
log("ANALYSIS COMPLETE")
log("=" * 70)
log("All plots saved. See Lab Report for interpretation.")
