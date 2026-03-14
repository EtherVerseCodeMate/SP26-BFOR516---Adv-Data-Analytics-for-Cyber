import pandas as pd
import numpy as np
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score

def log(msg):
    print(msg)
    sys.stdout.flush()

log("Loading data...")
start = time.time()
df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')
log(f"Data loaded in {time.time() - start:.2f} seconds. Shape: {df.shape}")

target = 'BinaryNumTarget'
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

df = df.dropna(subset=[target])
X_what = df[what_features].fillna(0)
X_how = df[how_features].fillna(0)
y = df[target]

log("Splitting data...")
X_w_train, X_w_test, y_train, y_test = train_test_split(X_what, y, test_size=0.3, random_state=42, stratify=y)
X_h_train, X_h_test, _, _ = train_test_split(X_how, y, test_size=0.3, random_state=42, stratify=y)

log("Scaling data...")
scaler = StandardScaler()
X_w_train_sc = scaler.fit_transform(X_w_train)
X_w_test_sc = scaler.transform(X_w_test)
X_h_train_sc = scaler.fit_transform(X_h_train)
X_h_test_sc = scaler.transform(X_h_test)

log("Training LR (WHAT)...")
lr_w = LogisticRegression(max_iter=100, random_state=42).fit(X_w_train_sc, y_train)
log("Training LR (HOW)...")
lr_h = LogisticRegression(max_iter=100, random_state=42).fit(X_h_train_sc, y_train)

log("Training NB (WHAT)...")
nb_w = GaussianNB().fit(X_w_train, y_train)
log("Training NB (HOW)...")
nb_h = GaussianNB().fit(X_h_train, y_train)

def report(model, X, y_true, name):
    log(f"Evaluating {name}...")
    y_prob = model.predict_proba(X)[:, 1]
    return roc_auc_score(y_true, y_prob)

log("\nRESULTS (ROC AUC):")
print(f"LR WHAT: {report(lr_w, X_w_test_sc, y_test, 'LR WHAT'):.4f}")
print(f"LR HOW:  {report(lr_h, X_h_test_sc, y_test, 'LR HOW'):.4f}")
print(f"NB WHAT: {report(nb_w, X_w_test, y_test, 'NB WHAT'):.4f}")
print(f"NB HOW:  {report(nb_h, X_h_test, y_test, 'NB HOW'):.4f}")
sys.stdout.flush()
