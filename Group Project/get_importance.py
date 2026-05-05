import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
np.random.seed(42)
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

NAMED_COLS = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]

repo = fetch_ucirepo(id=350)
df = pd.concat([repo.data.features, repo.data.targets], axis=1)
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={df.columns[-1]: "default"})
if "X1" in df.columns:
    df = df.rename(columns={f"X{i+1}": n for i, n in enumerate(NAMED_COLS)})
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})
X = df.drop(columns=["default"]); y = df["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=50, min_samples_leaf=20,
                             class_weight="balanced", criterion="gini", random_state=42)
dt.fit(X_train.values, y_train)

imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 DT Feature Importances:")
for feat, val in imp.head(10).items():
    print(f"  {feat}: {val:.4f} ({val*100:.2f}%)")

pay0_pct = imp["PAY_0"] * 100
top3 = imp.head(3).index.tolist()
print(f"\nPAY_0 importance: {pay0_pct:.2f}%")
print(f"Top 3 features: {top3}")

# Save to metrics
with open("project_metrics.json") as f:
    M = json.load(f)
M["feature_importance"] = {feat: round(float(val), 4) for feat, val in imp.items()}
M["pay0_importance_pct"] = round(pay0_pct, 2)
M["top3_features"] = top3
with open("project_metrics.json", "w") as f:
    json.dump(M, f, indent=2)
print("Saved to project_metrics.json")
