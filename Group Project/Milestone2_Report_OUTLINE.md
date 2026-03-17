# BFOR 516 – Group Project Milestone 2: Progress Report
**Title:** Predicting Credit Card Defaults using Machine Learning Techniques
**Team Members:** [Names]
**Date:** March 2026

---

> ⚠️ **IMPORTANT – Academic Integrity Note**
> This document is a **structural outline only**. All analysis paragraphs, interpretations,
> and conclusions **must be written in your own words**. AI-generated writing will be penalized.
> Use the notebook output (metrics, plots) to inform your analysis, then write it yourself.

---

## 1. Detailed Description

### 1a. Project Objective (and any evolution from Milestone 1)

[YOUR WRITING — Describe the goal: predicting credit card payment default using ML.
Note whether the objective has changed since Milestone 1. In our case: we added a
third model (Decision Tree) to the originally planned Logistic Regression + Naive Bayes
to get deeper feature interpretation and stronger comparison. Explain WHY this decision
was made — what motivated adding a third model.]

### 1b. Dataset Description

**Dataset:** Default of Credit Card Clients
**Source:** UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
**Original Paper:** Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.

[YOUR WRITING — Describe the dataset in your own words:
- 30,000 Taiwan credit card clients (April–September 2005)
- 23 features: demographics (SEX, EDUCATION, MARRIAGE, AGE), credit limit (LIMIT_BAL),
  6 months of payment status (PAY_0–PAY_6), bill amounts (BILL_AMT1–6), payment amounts (PAY_AMT1–6)
- Target: binary default indicator (1 = default, 0 = no default)
- Why selected: real-world financial dataset, clean, well-documented, directly applicable
  to cybersecurity/fraud analytics and financial risk]

### 1c. Model Selection Rationale

**Model 1: Logistic Regression**
[YOUR WRITING — Why LR? Interpretable, strong linear baseline, widely used in credit scoring.
How does it help analyze this dataset? Coefficients show which features push toward/away from default.
What are its limitations given this data?]

**Model 2: Gaussian Naive Bayes**
[YOUR WRITING — Why GNB? Probabilistic approach, fast, different assumptions than LR.
The "naive" independence assumption is a known limitation — do the features actually correlate?
(See correlation heatmap — they do, especially BILL_AMT features.) This makes for an
interesting comparison.]

**Model 3: Decision Tree Classifier**
[YOUR WRITING — Why DT? Captures non-linear relationships that LR misses. Provides
feature importance via Gini impurity. Decisions can be visualized as rules. Does not
require scaling. Complements the linear (LR) and probabilistic (NB) approaches.]

**Comparison Strategy**
[YOUR WRITING — How will you compare the 3 models? Metrics used (Accuracy, Precision,
Recall, F1, ROC-AUC, CV-AUC). Why is Recall important for default prediction — what is
the real-world cost of a False Negative (missing an actual default) vs. a False Positive?]

---

## 2. Dataset Preparation

### 2a. Initial Inspection

[YOUR WRITING — Describe what you found when first loading the data:
- Shape: 30,000 rows × 24 columns (including target)
- No missing values (confirm from notebook output)
- Data types: all numeric (integers/floats)
- Notable: class imbalance (~22% default, ~78% no default)]

### 2b. Data Cleaning

[YOUR WRITING — Describe the two cleaning steps and WHY they were necessary:

1. EDUCATION values 0, 5, 6: The original dataset documentation defines only values
   1 (graduate school), 2 (university), 3 (high school), 4 (other). Values 0, 5, 6 appear
   in the data but are undocumented. Remapped to 4 (Other).
   → How many rows were affected? (check notebook output)

2. MARRIAGE value 0: Only 1, 2, 3 are documented. 0 is undocumented.
   → How many rows were affected? (check notebook output)

Justify: How did you discover these issues? (EDA — checking value_counts() on categorical columns)]

### 2c. Feature Engineering & Preprocessing

[YOUR WRITING — Describe the preprocessing pipeline:

1. Feature/target split: X (23 features), y (default)
2. Categorical features (SEX, EDUCATION, MARRIAGE) kept as ordinal integers —
   justify this choice vs. one-hot encoding
3. Stratified 80/20 train/test split — explain why stratified is important given imbalance
4. StandardScaler: fit ONLY on training data, then transform both sets — explain why
   (prevent data leakage — test set statistics would contaminate the scaler)
5. Decision Tree: uses unscaled data — explain why DT is scale-invariant
6. Class imbalance handling: class_weight='balanced' in LR and DT — explain what this does]

**Code Snippet — Data Cleaning:**
```python
# Fix undocumented EDUCATION values
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
# Fix undocumented MARRIAGE values
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
```

**Code Snippet — Train/Test Split + Scaling:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # fit on train only
```

---

## 3. Model Building

### 3a. Logistic Regression

[YOUR WRITING — Key hyperparameter choices and reasoning:
- C=1.0: default regularization (L2); prevents overfitting
- max_iter=1000: ensures solver convergence on this dataset
- class_weight='balanced': adjusts for 78/22 imbalance
- solver='lbfgs': efficient for binary classification]

```python
lr_model = LogisticRegression(
    C=1.0, max_iter=1000,
    class_weight='balanced',
    solver='lbfgs', random_state=42
)
```

### 3b. Gaussian Naive Bayes

[YOUR WRITING — GNB has no hyperparameters beyond the prior. Describe the assumption:
features within each class follow a Gaussian distribution. After StandardScaler,
this is a reasonable approximation for continuous features. Note: categorical features
(SEX, EDUCATION, MARRIAGE) treated as continuous here — acknowledge this limitation.]

```python
nb_model = GaussianNB()
```

### 3c. Decision Tree Classifier

[YOUR WRITING — Key hyperparameter choices and reasoning:
- max_depth=5: limits tree complexity, prevents memorizing training data (overfitting)
- min_samples_split=50: internal node must have at least 50 samples to split further
- min_samples_leaf=20: leaf node must have at least 20 samples
- class_weight='balanced': same class imbalance handling as LR
- criterion='gini': Gini impurity for split quality
Explain: why not max_depth=None? (would overfit on 30k samples)]

```python
dt_model = DecisionTreeClassifier(
    max_depth=5, min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    criterion='gini', random_state=42
)
```

### 3d. Evaluation Framework

[YOUR WRITING — Describe: classification report, confusion matrix, 5-fold stratified CV.
Why 5-fold CV? Gives more reliable generalization estimate than single train/test split.
Why StratifiedKFold? Preserves class ratio in each fold.]

---

## 4. Preliminary Results and Analysis

### 4a. Results Table

[INSERT YOUR ACTUAL RESULTS TABLE from notebook — copy the output here]

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV-AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| Logistic Regression | ... | ... | ... | ... | ... | ... |
| Naive Bayes | ... | ... | ... | ... | ... | ... |
| Decision Tree | ... | ... | ... | ... | ... | ... |

### 4b. Analysis

[YOUR WRITING — Critical analysis. For each model:

**Logistic Regression:**
Discuss accuracy, recall, F1, AUC. Is the model identifying true defaults well?
What do the top coefficients (PAY_0, PAY_2, etc.) tell us about which features drive default?

**Naive Bayes:**
How does it compare? The independence assumption is violated here (BILL_AMT features
are highly correlated ~0.9+). Does this hurt performance? Where does NB fail?

**Decision Tree:**
How does it compare to LR? What does the tree visualization reveal about decision logic?
What are the top features (likely PAY_0 at the root)? Does max_depth=5 seem right,
or is the model underfitting/overfitting?

**PCA Observations:**
What does the 2D PCA scatter show? Can the classes be separated linearly?
How many components capture 95% of variance? What does this suggest about
feature redundancy?

**Class imbalance impact:**
The ~22% default rate — how does this affect the confusion matrices?
Discuss false negatives (missed defaults) vs. false positives (wrongly flagged)
in the context of real-world credit risk.]

### 4c. Relevant Plots
*(Insert plots generated by the notebook — see saved .png files in the project folder)*

- fig_01_class_distribution.png
- fig_02_demographic_default_rates.png
- fig_03_age_analysis.png
- fig_04_payment_history.png
- fig_05_correlation_heatmap.png
- fig_06_pca_variance.png
- fig_07_pca_2d_scatter.png
- fig_cm_logistic_regression.png
- fig_cm_naive_bayes.png
- fig_cm_decision_tree.png
- fig_08_model_comparison_roc.png
- fig_09_feature_importance.png
- fig_decision_tree_viz.png

---

## 5. Plan for Remainder of the Project & Conclusion

### 5a. Plan & Timeline

| Week | Task |
|------|------|
| Week of Mar 17 | Run notebook on JupyterHub; collect all metric outputs and plots |
| Week of Mar 24 | Hyperparameter tuning (GridSearchCV for LR C values, DT max_depth) |
| Week of Mar 31 | Explore SMOTE for class imbalance; consider Random Forest as 4th model |
| Week of Apr 7  | Feature engineering (payment utilization ratios); re-evaluate all models |
| Week of Apr 14 | Final model selection; write final report; prepare presentation |

[YOUR WRITING — Elaborate on each step. What specific improvements are planned?
What do the preliminary results suggest needs the most work?]

### 5b. Team Member Roles

[YOUR WRITING — Describe each member's contribution to date and planned role.
Be specific: who built which model, who did EDA, who wrote which section, etc.]

| Team Member | Contribution to Date | Planned Role |
|-------------|---------------------|--------------|
| [Name 1] | ... | ... |
| [Name 2] | ... | ... |
| [Name 3] | ... | ... |

### 5c. Concluding Remarks

[YOUR WRITING — 2-3 sentences: what is the project trajectory? Are preliminary results
promising? What is the most interesting finding so far?]

---

## 6. AI Declaration and Citations

### AI Usage Declaration

[YOUR WRITING — Be transparent and specific. Required elements per rubric:
- Tool name: Claude (claude.ai / Claude Code)
- What was AI-assisted: generating the Python notebook code (data loading, model pipeline,
  visualization code), structuring the report outline, debugging pandas/sklearn syntax
- What was NOT AI-assisted: all analysis writing, interpretation of results, model
  selection rationale, dataset selection decision, team contributions
- Evidence: you can reference that this report outline itself was AI-generated,
  but all analysis paragraphs were written by team members]

### Citations

1. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020

2. UCI Machine Learning Repository. (2016). *Default of Credit Card Clients Dataset* (Dataset ID 350). https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. https://scikit-learn.org

4. Pandas Development Team. (2024). *pandas: powerful Python data analysis toolkit*. https://pandas.pydata.org

5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.

6. Waskom, M. L. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021. https://doi.org/10.21105/joss.03021

[Add any additional papers or resources your team referenced]
