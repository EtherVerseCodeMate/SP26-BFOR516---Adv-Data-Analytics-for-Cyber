import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

def log(msg):
    print(msg)
    sys.stdout.flush()

log("Loading dataset...")
df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')

target = 'BinaryNumTarget'
df = df.dropna(subset=[target])

# Feature classification
what_features = [
    'ORG_percentage', 'NORP_percentage', 'GPE_percentage', 'PERSON_percentage',
    'MONEY_percentage', 'DATE_percentage', 'CARDINAL_percentage', 'PERCENT_percentage',
    'ORDINAL_percentage', 'FAC_percentage', 'LAW_percentage', 'PRODUCT_percentage',
    'EVENT_percentage', 'TIME_percentage', 'LOC_percentage', 'WORK_OF_ART_percentage',
    'QUANTITY_percentage', 'LANGUAGE_percentage', 'unique_count', 'total_count'
]
how_features = [
    'Word count', 'Max word length', 'Min word length', 'Average word length',
    'present_verbs', 'past_verbs', 'adjectives', 'adverbs', 'adpositions',
    'pronouns', 'TOs', 'determiners', 'conjunctions', 'dots', 'exclamation',
    'questions', 'ampersand', 'capitals', 'digits', 'long_word_freq', 'short_word_freq'
]

X_what = df[what_features].fillna(0)
X_how = df[how_features].fillna(0)
y = df[target]

log("Splitting and scaling...")
X_w_train, X_w_test, y_train, y_test = train_test_split(X_what, y, test_size=0.3, random_state=42, stratify=y)
X_h_train, X_h_test, _, _ = train_test_split(X_how, y, test_size=0.3, random_state=42, stratify=y)

scaler_w = StandardScaler()
X_w_train_sc = scaler_w.fit_transform(X_w_train)
X_w_test_sc = scaler_w.transform(X_w_test)

scaler_h = StandardScaler()
X_h_train_sc = scaler_h.fit_transform(X_h_train)
X_h_test_sc = scaler_h.transform(X_h_test)

def run_model(model, X_train, X_test, y_train, y_test, name, feat_type):
    log(f"Running {name} on {feat_type}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        'name': f"{name} ({feat_type})",
        'acc': accuracy_score(y_test, y_pred),
        'prec': precision_score(y_test, y_pred),
        'rec': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'y_prob': y_prob,
        'cm': confusion_matrix(y_test, y_pred),
        'coef': model.coef_[0] if hasattr(model, 'coef_') else None,
        'feats': what_features if feat_type == 'WHAT' else how_features
    }

results = []
results.append(run_model(LogisticRegression(max_iter=500), X_w_train_sc, X_w_test_sc, y_train, y_test, "LR", "WHAT"))
results.append(run_model(LogisticRegression(max_iter=500), X_h_train_sc, X_h_test_sc, y_train, y_test, "LR", "HOW"))
results.append(run_model(GaussianNB(), X_w_train, X_w_test, y_train, y_test, "NB", "WHAT"))
results.append(run_model(GaussianNB(), X_h_train, X_h_test, y_train, y_test, "NB", "HOW"))

log("Generating plots...")
# 1. Plot ROC Curves
plt.figure(figsize=(10, 6))
for r in results:
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    plt.plot(fpr, tpr, label=f"{r['name']} (AUC = {r['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: WHAT vs HOW')
plt.legend()
plt.savefig('roc_comparison.png')
plt.close()

# 2. Plot Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, r in enumerate(results):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(r['name'])
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# 3. Feature Importance (LR)
lr_whats = results[0]
lr_hows = results[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
pd.Series(lr_whats['coef'], index=lr_whats['feats']).sort_values().plot(kind='barh', ax=ax1)
ax1.set_title("LR Coefficients (WHAT Features)")
pd.Series(lr_hows['coef'], index=lr_hows['feats']).sort_values().plot(kind='barh', ax=ax2)
ax2.set_title("LR Coefficients (HOW Features)")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

log("\nSUMMARY TABLE:")
res_df = pd.DataFrame(results).drop(columns=['y_prob', 'cm', 'coef', 'feats'])
print(res_df.to_string(index=False))
log("\nDone.")
