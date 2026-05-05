"""
BFOR 516 Group Project — Credit Card Default Prediction
Milestone 3: Generate all figures and metrics for the presentation.
Run with: python run_project.py
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
sns.set_style("whitegrid")
sns.set_palette("Set2")

OUT = "images"
os.makedirs(OUT, exist_ok=True)

def save(fname):
    path = os.path.join(OUT, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
NAMED_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

print("\n[1] Loading dataset ...")
try:
    from ucimlrepo import fetch_ucirepo
    repo = fetch_ucirepo(id=350)
    df = pd.concat([repo.data.features, repo.data.targets], axis=1)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[-1]: "default"})
    # ucimlrepo returns X1-X23; rename to descriptive names
    if "X1" in df.columns:
        rename_map = {f"X{i+1}": name for i, name in enumerate(NAMED_COLS)}
        df = df.rename(columns=rename_map)
    print("    Loaded via ucimlrepo.")
except Exception as e:
    print(f"    ucimlrepo failed ({e}), using direct XLS ...")
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00350/default%20of%20credit%20card%20clients.xls")
    df = pd.read_excel(url, header=1, index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"default payment next month": "default"})

if "ID" in df.columns:
    df = df.drop(columns=["ID"])

print(f"    Shape: {df.shape}")
print(f"    Default rate: {df['default'].mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] EDA figures ...")

# Fig 01 – Class distribution
counts = df["default"].value_counts()
labels = ["No Default (0)", "Default (1)"]
colors = ["steelblue", "tomato"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
axes[0].set_title("Class Distribution (Count)")
axes[0].set_ylabel("Number of Clients")
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f"{val:,}\n({val/len(df):.1%})", ha="center", fontsize=11)
axes[1].pie(counts.values, labels=["No Default", "Default"],
            autopct="%1.1f%%", colors=colors, startangle=90,
            explode=(0, 0.06), shadow=True)
axes[1].set_title("Class Distribution (Proportion)")
plt.suptitle("Target Variable: Credit Card Default Next Month",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig01 = save("fig_01_class_distribution.png")

imbalance_ratio = counts[0] / counts[1]
print(f"    Imbalance ratio: {imbalance_ratio:.1f}:1")

# Fig 02 – Demographic default rates
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sex_rate  = df.groupby("SEX")["default"].mean()
sex_labels = {1: "Male", 2: "Female"}
axes[0].bar([sex_labels.get(k, k) for k in sex_rate.index], sex_rate.values,
            color=["#4472C4", "#ED7D31"], edgecolor="black")
axes[0].set_title("Default Rate by Gender"); axes[0].set_ylabel("Default Rate")
axes[0].set_ylim(0, 0.30)
for i, v in enumerate(sex_rate.values):
    axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11)

edu_rate   = df.groupby("EDUCATION")["default"].mean().sort_index()
edu_labels = {1: "Grad\nSchool", 2: "University", 3: "High\nSchool",
              4: "Other", 5: "Unknown5", 6: "Unknown6", 0: "Unknown0"}
axes[1].bar([edu_labels.get(k, str(k)) for k in edu_rate.index], edu_rate.values,
            color="steelblue", edgecolor="black")
axes[1].set_title("Default Rate by Education"); axes[1].set_ylabel("Default Rate")
axes[1].set_ylim(0, 0.35)
for i, v in enumerate(edu_rate.values):
    axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10)

mar_rate   = df.groupby("MARRIAGE")["default"].mean().sort_index()
mar_labels = {0: "Unknown", 1: "Married", 2: "Single", 3: "Other"}
axes[2].bar([mar_labels.get(k, str(k)) for k in mar_rate.index], mar_rate.values,
            color="seagreen", edgecolor="black")
axes[2].set_title("Default Rate by Marital Status"); axes[2].set_ylabel("Default Rate")
axes[2].set_ylim(0, 0.30)
for i, v in enumerate(mar_rate.values):
    axes[2].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11)
plt.suptitle("Default Rate by Demographic Features", fontsize=13, fontweight="bold")
plt.tight_layout()
fig02 = save("fig_02_demographic_default_rates.png")

# Fig 03 – Age analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df[df["default"]==0]["AGE"], bins=40, alpha=0.6,
             color="steelblue", label="No Default", density=True)
axes[0].hist(df[df["default"]==1]["AGE"], bins=40, alpha=0.6,
             color="tomato", label="Default", density=True)
axes[0].set_title("Age Distribution by Default Status")
axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel("Density"); axes[0].legend()

bins_age = [20, 30, 40, 50, 60, 80]
labels_age = ["21-30", "31-40", "41-50", "51-60", "61+"]
df["AGE_GROUP"] = pd.cut(df["AGE"], bins=bins_age, labels=labels_age)
age_default = df.groupby("AGE_GROUP", observed=False)["default"].mean()
axes[1].bar(age_default.index.astype(str), age_default.values,
            color="steelblue", edgecolor="black")
axes[1].set_title("Default Rate by Age Group")
axes[1].set_xlabel("Age Group"); axes[1].set_ylabel("Default Rate")
axes[1].set_ylim(0, 0.35)
for i, v in enumerate(age_default.values):
    axes[1].text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=11)
plt.suptitle("Age Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
fig03 = save("fig_03_age_analysis.png")
df = df.drop(columns=["AGE_GROUP"])

# Fig 04 – Payment history
pay_cols   = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
month_labels_pay = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
pay_defaults = df.groupby("default")[pay_cols].mean()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, color, name in zip([0, 1], ["steelblue", "tomato"], ["No Default", "Default"]):
    axes[0].plot(month_labels_pay, pay_defaults.loc[label].values,
                 marker="o", color=color, linewidth=2, label=name)
axes[0].set_title("Avg Payment Delay Status Over 6 Months")
axes[0].set_xlabel("Month"); axes[0].set_ylabel("Avg PAY Status")
axes[0].legend(); axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
pay0_default = df.groupby("PAY_0")["default"].mean()
axes[1].bar(pay0_default.index, pay0_default.values, color="steelblue", edgecolor="black")
axes[1].set_title("Default Rate by Most Recent Payment Status (PAY_0)")
axes[1].set_xlabel("PAY_0 Value"); axes[1].set_ylabel("Default Rate")
plt.suptitle("Payment History Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
fig04 = save("fig_04_payment_history.png")

# Fig 05 – Correlation heatmap
plt.figure(figsize=(16, 12))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm",
            center=0, vmin=-1, vmax=1, linewidths=0.3,
            cbar_kws={"label": "Pearson Correlation"})
plt.title("Feature Correlation Matrix (Lower Triangle)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig05 = save("fig_05_correlation_heatmap.png")
target_corr = corr["default"].drop("default").abs().sort_values(ascending=False)
print("    Top 5 features correlated with default:")
for feat, val in target_corr.head(5).items():
    print(f"      {feat}: {val:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEANING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Cleaning ...")
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})
print(f"    Missing values: {df.isnull().sum().sum()}")
print(f"    Duplicates: {df.duplicated().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE PREP
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Feature preparation ...")
X = df.drop(columns=["default"])
y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"    Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"    Train default rate: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PCA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] PCA ...")
pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)
explained = pca_full.explained_variance_ratio_
cumvar    = np.cumsum(explained)
n_90 = int(np.argmax(cumvar >= 0.90)) + 1
n_95 = int(np.argmax(cumvar >= 0.95)) + 1
n_99 = int(np.argmax(cumvar >= 0.99)) + 1
print(f"    Components for 90%: {n_90} | 95%: {n_95} | 99%: {n_99}")
print(f"    Top 5 PCs explain: {cumvar[4]:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, len(explained)+1), explained, color="steelblue", alpha=0.8, edgecolor="white")
axes[0].plot(range(1, len(explained)+1), explained, "o-", color="tomato", markersize=4, linewidth=1.5)
axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Scree Plot"); axes[0].set_xlim(0, 24)
axes[1].plot(range(1, len(cumvar)+1), cumvar, "o-", color="steelblue", markersize=5, linewidth=2)
axes[1].axhline(0.90, color="green",  linestyle="--", alpha=0.7, label="90%")
axes[1].axhline(0.95, color="orange", linestyle="--", alpha=0.7, label="95%")
axes[1].axhline(0.99, color="red",    linestyle="--", alpha=0.7, label="99%")
axes[1].axvline(n_95, color="orange", linestyle=":",  alpha=0.6)
axes[1].fill_between(range(1, len(cumvar)+1), cumvar, alpha=0.15, color="steelblue")
axes[1].set_xlabel("Number of Components"); axes[1].set_ylabel("Cumulative Variance")
axes[1].set_title("Cumulative Explained Variance"); axes[1].legend(title="Threshold")
axes[1].set_xlim(0, 24)
plt.suptitle("PCA – Explained Variance Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
fig06 = save("fig_06_pca_variance.png")

pca_2d = PCA(n_components=2, random_state=42)
X_pca_train = pca_2d.fit_transform(X_train_scaled)
plt.figure(figsize=(10, 7))
for label, color, name in zip([0, 1], ["steelblue", "tomato"], ["No Default", "Default"]):
    mask = y_train == label
    plt.scatter(X_pca_train[mask, 0], X_pca_train[mask, 1],
                c=color, alpha=0.25, s=6, label=f"{name} (n={mask.sum():,})")
var1 = pca_2d.explained_variance_ratio_[0]; var2 = pca_2d.explained_variance_ratio_[1]
plt.xlabel(f"PC1 ({var1:.1%} explained variance)")
plt.ylabel(f"PC2 ({var2:.1%} explained variance)")
plt.title(f"PCA 2D Projection (PC1+PC2 = {var1+var2:.1%} of total variance)")
plt.legend(markerscale=4, fontsize=11)
plt.tight_layout()
fig07 = save("fig_07_pca_2d_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. MODELS
# ─────────────────────────────────────────────────────────────────────────────
all_results = []
all_probs   = {}
model_colors = ["#4472C4", "#70AD47", "#ED7D31"]

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, fname_prefix):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, y_pred), 4),
        "F1-Score":  round(f1_score(y_te, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_te, y_prob), 4) if y_prob is not None else None,
    }
    print(f"\n    {name}:")
    print(f"      Acc={metrics['Accuracy']} | Prec={metrics['Precision']} "
          f"| Rec={metrics['Recall']} | F1={metrics['F1-Score']} | AUC={metrics['ROC-AUC']}")
    print(classification_report(y_te, y_pred, target_names=["No Default", "Default"],
                                 zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, y_pred,
        display_labels=["No Default", "Default"],
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{name} – Confusion Matrix")
    plt.tight_layout()
    save(f"fig_cm_{fname_prefix}.png")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
    metrics["CV_AUC_Mean"] = round(cv_scores.mean(), 4)
    metrics["CV_AUC_Std"]  = round(cv_scores.std(), 4)
    print(f"      CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    return metrics, model, y_prob

print("\n[6] Training models ...")

# Logistic Regression
lr_metrics, trained_lr, prob_lr = evaluate_model(
    "Logistic Regression",
    LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                       solver="lbfgs", random_state=42),
    X_train_scaled, y_train, X_test_scaled, y_test, "logistic_regression"
)
all_results.append(lr_metrics); all_probs["Logistic Regression"] = prob_lr

# Naive Bayes
nb_metrics, trained_nb, prob_nb = evaluate_model(
    "Naive Bayes", GaussianNB(),
    X_train_scaled, y_train, X_test_scaled, y_test, "naive_bayes"
)
all_results.append(nb_metrics); all_probs["Naive Bayes"] = prob_nb

# Decision Tree
dt_metrics, trained_dt, prob_dt = evaluate_model(
    "Decision Tree",
    DecisionTreeClassifier(max_depth=5, min_samples_split=50,
                            min_samples_leaf=20, class_weight="balanced",
                            criterion="gini", random_state=42),
    X_train.values, y_train, X_test.values, y_test, "decision_tree"
)
all_results.append(dt_metrics); all_probs["Decision Tree"] = prob_dt

# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPARISON CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Comparison charts ...")
results_df = pd.DataFrame(all_results).set_index("Model")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
metrics_bar = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics_bar)); width = 0.22
for i, (model_name, row) in enumerate(results_df.iterrows()):
    vals = [row[m] for m in metrics_bar]
    axes[0].bar(x + i*width, vals, width, label=model_name,
                color=model_colors[i], edgecolor="black", alpha=0.88)
axes[0].set_xticks(x + width); axes[0].set_xticklabels(metrics_bar)
axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("Score")
axes[0].set_title("Model Performance Comparison"); axes[0].legend()
axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.4)
axes[0].axhline(0.8, color="green", linestyle=":", alpha=0.4)

for (model_name, prob), color in zip(all_probs.items(), model_colors):
    if prob is not None:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        axes[1].plot(fpr, tpr, color=color, lw=2.5, label=f"{model_name} (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.500)")
axes[1].fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves"); axes[1].legend(loc="lower right", fontsize=10)
plt.suptitle("Model Comparison – Results", fontsize=14, fontweight="bold")
plt.tight_layout()
fig08 = save("fig_08_model_comparison_roc.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Feature importance ...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
dt_importance = pd.Series(
    trained_dt.feature_importances_, index=X.columns
).sort_values(ascending=True).tail(15)
dt_importance.plot(kind="barh", color="#ED7D31", edgecolor="black", ax=axes[0])
axes[0].set_title("Decision Tree – Top 15 Feature Importances (Gini)")
axes[0].set_xlabel("Importance")

lr_importance = pd.Series(
    np.abs(trained_lr.coef_[0]), index=X.columns
).sort_values(ascending=True).tail(15)
lr_importance.plot(kind="barh", color="#4472C4", edgecolor="black", ax=axes[1])
axes[1].set_title("Logistic Regression – Top 15 |Coefficients|")
axes[1].set_xlabel("|Coefficient|")
plt.suptitle("Feature Importance Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
fig09 = save("fig_09_feature_importance.png")

# Decision Tree Visualization
plt.figure(figsize=(22, 8))
plot_tree(trained_dt, feature_names=X.columns.tolist(),
          class_names=["No Default", "Default"],
          filled=True, max_depth=3, fontsize=8, impurity=True, proportion=False)
plt.title("Decision Tree Visualization (Top 3 Levels of 5)",
          fontsize=13, fontweight="bold")
plt.tight_layout()
fig_dt_viz = save("fig_decision_tree_viz.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE METRICS JSON
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Saving metrics ...")
best_f1  = results_df["F1-Score"].idxmax()
best_auc = results_df["ROC-AUC"].idxmax()

metrics_out = {
    "dataset": {
        "total_records": int(len(df)),
        "features": int(X.shape[1]),
        "default_rate": f"{df['default'].mean():.2%}",
        "no_default_count": int(counts[0]),
        "default_count": int(counts[1]),
        "imbalance_ratio": f"{imbalance_ratio:.1f}:1",
        "avg_credit_limit": int(df["LIMIT_BAL"].mean()),
        "avg_age": int(df["AGE"].mean()),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "missing_values": 0,
        "duplicate_rows": int(df.duplicated().sum()),
    },
    "pca": {
        "components_90pct": n_90,
        "components_95pct": n_95,
        "components_99pct": n_99,
        "top5_variance": f"{cumvar[4]:.1%}",
    },
    "models": {m["Model"]: {k: v for k, v in m.items() if k != "Model"}
               for m in all_results},
    "best": {"f1_model": best_f1, "auc_model": best_auc},
    "figures": {
        "fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04,
        "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08,
        "fig09": fig09, "fig_dt_viz": fig_dt_viz,
    }
}

with open("project_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
print(f"\n  Dataset : {len(df):,} records | {X.shape[1]} features | "
      f"{df['default'].mean():.2%} default rate")
print(f"\n  {'Model':<22} {'Accuracy':>9} {'Recall':>8} {'F1':>8} {'AUC':>8} {'CV-AUC':>8}")
print(f"  {'-'*65}")
for _, row in results_df.iterrows():
    print(f"  {row.name:<22} {row['Accuracy']:>9.4f} {row['Recall']:>8.4f} "
          f"{row['F1-Score']:>8.4f} {row['ROC-AUC']:>8.4f} {row['CV_AUC_Mean']:>8.4f}")

print(f"\n  Best F1-Score  : {best_f1}  ({results_df.loc[best_f1,'F1-Score']})")
print(f"  Best ROC-AUC   : {best_auc}  ({results_df.loc[best_auc,'ROC-AUC']})")
print(f"\n  All figures saved to: {OUT}/")
print("  Metrics saved to   : project_metrics.json")
print("  Done.")
