# BFOR 516 – Group Project Milestone 2: Progress Report
**Title:** Predicting Credit Card Defaults using Machine Learning Techniques
**Team Members:** [Names]
**Date:** March 2026

---

> ⚠️ **ACADEMIC INTEGRITY NOTE**
> Sections marked **[YOUR WRITING]** must be written entirely in your own words.
> The factual data below (numbers, tables, code snippets) comes from running the notebook.
> Any analysis, interpretation, or conclusion paragraphs written by AI will be penalized.

---

## 1. Detailed Description

### 1a. Project Objective (and any evolution from Milestone 1)

**[YOUR WRITING]** — Write in your own words:
- Core goal: predict whether a Taiwan credit card client will default on their payment next month
- What stayed the same from Milestone 1 (LR + NB as primary models)
- What evolved: a third model (Decision Tree Classifier) was added to meet the requirement
  of ~3 models for rigorous comparison, and because DT provides built-in feature importance
  (Gini impurity) that neither LR nor NB offer in the same way
- Brief mention of PCA as a supporting analysis technique (not a prediction model)

---

### 1b. Dataset Description

**Dataset:** Default of Credit Card Clients
**Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
**Original Paper:** Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.

**Confirmed Dataset Facts (from notebook output):**
- **Shape:** 30,000 rows × 24 columns (23 features + 1 target)
- **All columns:** integer type (int64), no null values
- **Memory:** 5.7 MB
- **Target distribution:** 23,364 no default (77.88%) vs. 6,636 default (22.12%)
- **Class imbalance ratio:** 3.5:1 (no default : default)
- **Credit limit range:** NT$10,000 – NT$1,000,000 (mean: NT$167,484)
- **Age range:** 21–79 years (mean: 35.49)

**[YOUR WRITING]** — Describe in your own words why this dataset was selected:
(real-world financial data, well-documented provenance, directly applicable to
financial risk and fraud detection in a cybersecurity context, clean and publicly available)

---

### 1c. Model Selection Rationale

**Model 1: Logistic Regression**

**[YOUR WRITING]** — Address: Why LR for credit default prediction? (industry standard,
interpretable coefficients directly show which features increase/decrease default probability,
widely used in real credit scoring systems — FICO etc.). What are its limitations with this data?
(assumes linear decision boundary — the PCA scatter confirms limited linear separability;
class imbalance handled via `class_weight='balanced'`)

**Model 2: Gaussian Naive Bayes**

**[YOUR WRITING]** — Address: Why GNB as a benchmark? (probabilistic approach, completely
different assumptions than LR). Important limitation to address: the "naive" independence
assumption is clearly violated here — the correlation matrix shows BILL_AMT features are
correlated with each other at ~0.9+, and PAY_0 through PAY_6 are also strongly correlated
with each other and with the target. Discuss whether this violation appears to hurt GNB's
performance compared to the other two models.

**Model 3: Decision Tree Classifier**

**[YOUR WRITING]** — Address: Why DT as the third model? (captures non-linear interactions
that LR cannot, produces visualizable decision rules, built-in Gini feature importance,
scale-invariant so no standardization needed). PAY_0 alone accounts for 74.61% of the DT's
feature importance — discuss what this reveals about the problem structure.

**Comparison Strategy**

**[YOUR WRITING]** — Explain your evaluation framework and why each metric matters for this
specific problem:
- **Recall** is especially critical: a missed default (False Negative) means a bank extends
  credit to someone who will not pay — a direct financial loss
- **Precision** matters too: flagging too many false alarms (False Positives) means denying
  credit to creditworthy customers — a business cost
- **ROC-AUC** captures overall discriminative ability across all decision thresholds
- **5-Fold CV AUC** validates that results generalize, not just memorized on one split

---

## 2. Dataset Preparation

### 2a. Initial Inspection

**Confirmed facts (notebook output):**
- 30,000 rows, 24 columns — all `int64`, no missing values
- **35 duplicate rows** found (noted; not dropped — rows represent distinct clients
  who happen to share the same feature values)
- Default rate: **22.12%** (6,636 defaults out of 30,000)
- Credit limit (LIMIT_BAL): right-skewed, mean NT$167,484, max NT$1,000,000
- PAY status columns: values range from -2 to +8; mean near 0 (most clients pay on time)
- BILL_AMT columns: mean ~NT$40k–51k with high standard deviation (some clients have
  very large balances including negatives, indicating credits/refunds)

**[YOUR WRITING]** — Describe in your own words what stood out from the initial inspection
and how it informed your cleaning and preprocessing decisions.

---

### 2b. Data Cleaning

**Issue 1 — Invalid EDUCATION values:**

The original dataset documentation (Yeh & Lien, 2009) defines only:
- 1 = graduate school, 2 = university, 3 = high school, 4 = others

Values `0`, `5`, `6` are present but undocumented.

**Actual counts from notebook:**
- EDUCATION values before cleaning: `[0, 1, 2, 3, 4, 5, 6]`
- **345 rows** contained undocumented values (0, 5, or 6)
- Action: remapped all three to `4` (others)
- EDUCATION values after cleaning: `[1, 2, 3, 4]`

**Issue 2 — Invalid MARRIAGE values:**

Documentation defines: 1 = married, 2 = single, 3 = others. Value `0` is undocumented.

- MARRIAGE values before cleaning: `[0, 1, 2, 3]`
- **54 rows** contained undocumented value `0`
- Action: remapped to `3` (others)
- MARRIAGE values after cleaning: `[1, 2, 3]`

**Total rows affected by cleaning: 399 rows out of 30,000 (1.3%)**

```python
# Fix undocumented EDUCATION values (0, 5, 6 → 4 = Other)
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
# Fix undocumented MARRIAGE values (0 → 3 = Other)
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
```

**[YOUR WRITING]** — Explain in your own words: how were these issues discovered?
(EDA step — examining `value_counts()` on categorical columns, then cross-referencing
with the dataset documentation to identify undocumented values). Why remap rather than
drop these rows? (1.3% of data is too significant to discard; remapping to "Other" is
semantically appropriate and preserves the records)

---

### 2c. Feature Engineering & Preprocessing

**Pipeline summary (confirmed from notebook):**

| Step | Detail | Confirmed Output |
|------|--------|-----------------|
| Feature/target split | X: 23 features, y: default | X shape: (30000, 23) |
| Stratified train/test split | 80/20, random_state=42 | Train: 24,000 / Test: 6,000 |
| Train default rate | Preserved by stratification | 22.12% (matches full dataset) |
| Test default rate | Preserved by stratification | 22.12% (matches full dataset) |
| StandardScaler | Fit on train, transform both | Train mean: 0.000000, std: 1.000000 |
| DT scaling | Not applied (scale-invariant) | Raw X_train / X_test used |

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # transform only — no data leakage
```

**[YOUR WRITING]** — Justify these choices in your own words:
1. Why stratified split? (preserves 22.12% default rate in both subsets — critical for
   imbalanced data; a random split could place more/fewer defaults in test)
2. Why fit scaler on train only? (if we used test statistics, we'd be leaking information
   about the test set into the training pipeline — a form of data leakage)
3. Why keep SEX/EDUCATION/MARRIAGE as ordinal integers vs. one-hot encoding?
4. Why `class_weight='balanced'` in LR and DT? (automatically adjusts sample weights
   inversely proportional to class frequency — compensates for 3.5:1 imbalance without
   resampling)

---

## 3. Model Building

### 3a. Logistic Regression

```python
lr_model = LogisticRegression(
    C=1.0,              # L2 regularization strength (inverse); default = no extra penalty
    max_iter=1000,      # ensures solver convergence on 24k samples
    class_weight='balanced',  # compensates for 3.5:1 class imbalance
    solver='lbfgs',     # efficient for binary classification, supports L2
    random_state=42
)
```

**[YOUR WRITING]** — Explain each hyperparameter choice. Also note: LR requires scaled
features because the lbfgs solver is gradient-based and sensitive to feature magnitude
(LIMIT_BAL in tens of thousands vs. SEX as 1 or 2 would dominate without scaling).

---

### 3b. Gaussian Naive Bayes

```python
nb_model = GaussianNB()
# No hyperparameters set — GNB estimates Gaussian parameters (mean, var)
# per feature per class from training data automatically
```

**[YOUR WRITING]** — Describe the Gaussian assumption: for each feature, GNB models its
distribution within each class as a Gaussian (normal distribution). After StandardScaler,
continuous features are closer to normally distributed. Note the known limitation:
categorical features (SEX=1/2, EDUCATION=1-4, MARRIAGE=1-3) are treated as continuous —
this is an approximation. Also: GNB does not support `class_weight` parameter, so class
imbalance is not explicitly handled — this is a design difference worth discussing.

---

### 3c. Decision Tree Classifier

```python
dt_model = DecisionTreeClassifier(
    max_depth=5,           # limits tree depth to prevent overfitting
    min_samples_split=50,  # node must have ≥50 samples to split
    min_samples_leaf=20,   # leaf must have ≥20 samples (stable estimates)
    class_weight='balanced',
    criterion='gini',      # Gini impurity for split quality
    random_state=42
)
# Decision Tree is scale-invariant — uses raw (unscaled) X_train / X_test
```

**[YOUR WRITING]** — Explain each hyperparameter. Specifically address `max_depth=5`:
why not unlimited? (on 30,000 samples, an unconstrained tree would memorize training data,
achieving near 100% train accuracy but poor test generalization). The tree visualization
shows the top 3 levels — what does the root split on? (Almost certainly PAY_0, consistent
with its 74.61% Gini importance weight)

---

### 3d. Evaluation Framework

All models evaluated using:
- Classification report (precision, recall, F1 per class)
- Confusion matrix visualization
- 5-Fold Stratified Cross-Validation (AUC scoring)
- ROC curve with AUC

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc')
```

**[YOUR WRITING]** — Explain why StratifiedKFold is used instead of regular KFold
(preserves class ratio in each fold), and why CV is important (single train/test split
results can vary depending on which 20% ends up in test — CV averages over 5 splits
for a more reliable estimate).

---

## 4. Preliminary Results and Analysis

### 4a. Results Table (Actual Output)

**Metrics on 6,000-sample test set (stratified 20% holdout):**

| Model | Accuracy | Precision* | Recall* | F1-Score* | ROC-AUC | CV-AUC (5-fold) |
|-------|----------|-----------|--------|----------|---------|----------------|
| Logistic Regression | 0.6795 | 0.3671 | **0.6202** | 0.4612 | 0.7084 | 0.7264 ± 0.0106 |
| Naive Bayes | 0.7518 | 0.4504 | 0.5539 | 0.4968 | 0.7248 | 0.7365 ± 0.0102 |
| **Decision Tree** | **0.7723** | **0.4870** | 0.5516 | **0.5173** | **0.7589** | **0.7577 ± 0.0067** |

*\*Precision, Recall, F1 reported for the **Default (1)** class*

**Best F1-Score: Decision Tree (0.5173)**
**Best ROC-AUC: Decision Tree (0.7589)**

**Full classification reports:**

**Logistic Regression:**
```
              precision    recall  f1-score   support
  No Default       0.87      0.70      0.77      4673
     Default       0.37      0.62      0.46      1327
    accuracy                           0.68      6000
   macro avg       0.62      0.66      0.62      6000
weighted avg       0.76      0.68      0.70      6000
5-Fold CV ROC-AUC: 0.7264 ± 0.0106
```

**Naive Bayes:**
```
              precision    recall  f1-score   support
  No Default       0.86      0.81      0.84      4673
     Default       0.45      0.55      0.50      1327
    accuracy                           0.75      6000
   macro avg       0.66      0.68      0.67      6000
weighted avg       0.77      0.75      0.76      6000
5-Fold CV ROC-AUC: 0.7365 ± 0.0102
```

**Decision Tree:**
```
              precision    recall  f1-score   support
  No Default       0.87      0.84      0.85      4673
     Default       0.49      0.55      0.52      1327
    accuracy                           0.77      6000
   macro avg       0.68      0.69      0.68      6000
weighted avg       0.78      0.77      0.78      6000
5-Fold CV ROC-AUC: 0.7577 ± 0.0067
```

---

### 4b. Feature Importance Results

**Decision Tree — Gini Feature Importance (top 10):**

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | PAY_0 | **0.7461** |
| 2 | PAY_AMT2 | 0.0737 |
| 3 | PAY_4 | 0.0372 |
| 4 | LIMIT_BAL | 0.0276 |
| 5 | PAY_3 | 0.0225 |
| 6 | PAY_2 | 0.0200 |
| 7 | PAY_AMT4 | 0.0150 |
| 8 | PAY_AMT3 | 0.0148 |
| 9 | PAY_AMT1 | 0.0105 |
| 10 | BILL_AMT2 | 0.0089 |

**Logistic Regression — |Coefficient| (top 10):**

| Rank | Feature | |Coefficient| |
|------|---------|-------------|
| 1 | PAY_0 | 0.5903 |
| 2 | BILL_AMT1 | 0.2853 |
| 3 | PAY_AMT1 | 0.1843 |
| 4 | PAY_AMT2 | 0.1612 |
| 5 | LIMIT_BAL | 0.1369 |
| 6 | PAY_2 | 0.1010 |
| 7 | PAY_3 | 0.0909 |
| 8 | EDUCATION | 0.0887 |
| 9 | MARRIAGE | 0.0809 |
| 10 | AGE | 0.0761 |

**Correlation with target (from EDA):**

| Feature | |Pearson r| with default |
|---------|-------------------------|
| PAY_0 | 0.3248 |
| PAY_2 | 0.2636 |
| PAY_3 | 0.2353 |
| PAY_4 | 0.2166 |
| PAY_5 | 0.2041 |
| PAY_6 | 0.1869 |
| LIMIT_BAL | 0.1535 |
| PAY_AMT1 | 0.0729 |

---

### 4c. PCA Results

| Threshold | Components Required (of 23) |
|-----------|---------------------------|
| 90% variance | 13 |
| 95% variance | 15 |
| 99% variance | 19 |
| Top 5 components | 64.17% |
| Top 10 components | 83.09% |

---

### 4d. Critical Analysis

**[YOUR WRITING]** — This is the most important section. Write your own analysis of
the numbers above. Suggested points to address:

**Logistic Regression (Accuracy 0.68, Recall 0.62, AUC 0.71):**
- Has the highest Recall (0.62) — catches the most actual defaults, but at the cost of
  low precision (0.37) meaning many false alarms
- Accuracy of 68% is the lowest — `class_weight='balanced'` causes it to sacrifice
  overall accuracy to improve detection of the minority class
- The top coefficient is PAY_0 (0.59) — interpret what this means
- BILL_AMT1 has a surprisingly high coefficient (0.29) despite low correlation with
  target (not in top 10 of Pearson r) — what might this suggest?

**Naive Bayes (Accuracy 0.75, Recall 0.55, AUC 0.72):**
- Middle ground on most metrics — better accuracy than LR but lower recall
- GNB does not use `class_weight` — does this explain lower recall vs. LR?
- The independence assumption is violated: BILL_AMT1–6 have near-perfect correlations
  (~0.9+) with each other. Does this appear to hurt or help NB? Discuss.
- CV AUC (0.7365) is stable (±0.0102) suggesting it generalizes consistently

**Decision Tree (Accuracy 0.77, F1 0.52, AUC 0.76):**
- Best on every metric except Recall (where LR wins narrowly)
- PAY_0 alone accounts for 74.61% of Gini importance — the tree is heavily dominated
  by this single feature. What are the implications? (model may be fragile if PAY_0
  is unavailable or noisy; also suggests a simple rule-based system could work well)
- CV AUC of 0.7577 ± 0.0067 has the tightest confidence interval — most stable model
- max_depth=5 appears appropriate: compare train vs. test performance (if overfitting,
  train accuracy would be much higher than test)

**PCA Analysis:**
- 15 components needed for 95% of variance out of 23 features — moderate redundancy
- The 2D scatter (PC1+PC2) shows significant class overlap — confirming that the
  problem is not linearly separable in a 2D projection
- This finding supports using non-linear methods (Decision Tree, or future Random Forest)
  over purely linear models

**Class imbalance impact:**
- Test set has 4,673 "No Default" vs. 1,327 "Default" — a model predicting "No Default"
  for everything would achieve ~78% accuracy. All 3 models beat this but the margin
  for LR is slim (68%)
- Discuss the confusion matrices: how many actual defaults are missed by each model?
  (1327 × (1 - recall)) — calculate and interpret the real-world meaning

---

### 4e. Relevant Plots
*(Insert plots from the Group Project folder into your final document)*

- `fig_01_class_distribution.png` — target class imbalance
- `fig_02_demographic_default_rates.png` — default rate by sex, education, marriage
- `fig_03_age_analysis.png` — age distribution and default rate by age group
- `fig_04_payment_history.png` — PAY_0–PAY_6 analysis
- `fig_05_correlation_heatmap.png` — feature correlations
- `fig_06_pca_variance.png` — scree plot + cumulative variance
- `fig_07_pca_2d_scatter.png` — 2D PCA class separation
- `fig_cm_logistic_regression.png` — confusion matrix
- `fig_cm_naive_bayes.png` — confusion matrix
- `fig_cm_decision_tree.png` — confusion matrix
- `fig_08_model_comparison_roc.png` — side-by-side bar chart + ROC curves
- `fig_09_feature_importance.png` — DT Gini + LR coefficient comparison
- `fig_decision_tree_viz.png` — tree visualization (top 3 levels)

---

## 5. Plan for Remainder of the Project & Conclusion

### 5a. Plan & Timeline

| Week | Planned Task |
|------|-------------|
| ~~Week of Mar 17~~ | ✅ Run notebook; collect outputs and plots |
| Week of Mar 24 | Hyperparameter tuning: GridSearchCV for LR (vary C), DT (vary max_depth, min_samples_leaf) |
| Week of Mar 31 | Address class imbalance with SMOTE (imblearn); re-evaluate all models |
| Week of Apr 7 | Feature engineering: payment utilization ratios (PAY_AMT / BILL_AMT); optionally test Random Forest |
| Week of Apr 14 | Final model selection with complete analysis; write final report; prepare presentation |

**[YOUR WRITING]** — Motivate each step from the preliminary results:
- Why tune hyperparameters? (e.g., DT at max_depth=5 — is it underfitting or is there
  room to improve? LR at C=1.0 — would more/less regularization help?)
- Why SMOTE? (class_weight handles imbalance during training but SMOTE creates synthetic
  minority class samples — may improve recall further)
- Why payment utilization ratios? (PAY_AMT1/BILL_AMT1 = how much of the bill was paid
  that month — this ratio may be more informative than the raw amounts individually)

---

### 5b. Team Member Roles

**[YOUR WRITING]** — Fill in actual names and contributions. Be specific.

| Team Member | Contribution to Date | Planned Role (Remainder) |
|-------------|---------------------|--------------------------|
| [Souhimbou Kone] | Project setup, notebook development, GitHub repo management | Hyperparameter tuning, final report |
| [Muhammad H Bahar] | Ran notebook on JupyterHub, shared output results | Model evaluation, SMOTE exploration |
| [Name 3] | [contribution] | [planned role] |
| [Name 4] | [contribution] | [planned role] |

---

### 5c. Concluding Remarks

**[YOUR WRITING]** — 3–4 sentences in your own words. Suggested points:
- Preliminary results show Decision Tree outperforms LR and NB on this dataset
- PAY_0 (most recent payment status) is consistently the strongest predictor across
  all three models — a finding consistent with the original Yeh & Lien (2009) paper
- The class overlap observed in PCA suggests that more powerful non-linear methods
  (Random Forest, Gradient Boosting) may significantly improve results in the final phase
- All three models achieve AUC > 0.70, indicating meaningful discriminative ability
  above random chance, providing a solid foundation for the final milestone

---

## 6. AI Declaration and Citations

### AI Usage Declaration

**[YOUR WRITING]** — Required by rubric to be transparent and specific. Suggested content
(write in your own voice):

Declare that:
- **Tool used:** Claude Code (Anthropic) — accessed via claude.ai / Claude Code CLI
- **What was AI-assisted:** The Python Jupyter notebook code was generated with Claude
  Code assistance — specifically: the data loading pipeline, StandardScaler/train-test
  split boilerplate, model training/evaluation loop structure, visualization code
  (matplotlib/seaborn), and the initial report outline structure
- **What was NOT AI-assisted:** All written analysis, interpretation of results,
  model selection rationale discussion, team role descriptions, conclusions — these
  were written by team members
- **Evidence:** The notebook file `CreditDefault_ML_Analysis.ipynb` in the GitHub repo
  was developed with AI coding assistance; the analysis paragraphs in this report
  were authored by the team

---

### Citations

1. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020

2. UCI Machine Learning Repository. (2016). *Default of Credit Card Clients* (Dataset ID 350). https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://scikit-learn.org

4. Pandas Development Team. (2024). *pandas: powerful Python data analysis toolkit* (v2.x). https://pandas.pydata.org

5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

6. Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021. https://doi.org/10.21105/joss.03021

7. Harris, C. R., et al. (2020). Array programming with NumPy. *Nature*, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2
