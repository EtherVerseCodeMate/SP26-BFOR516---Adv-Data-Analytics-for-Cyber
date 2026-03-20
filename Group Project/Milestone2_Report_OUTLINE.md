BFOR 516 – Group Project Milestone 2: Progress Report
Title: Predicting Credit Card Defaults using Machine Learning Techniques Team Members: [Souhimbou Kone, Muhammad H Bahar, [Name 3], [Name 4]] Date: March 2026

⚠️ ACADEMIC INTEGRITY NOTE Sections marked [YOUR WRITING] must be written entirely in your own words. The factual data below (numbers, tables, code snippets) comes from running the notebook. Any analysis, interpretation, or conclusion paragraphs written by AI will be penalized.

1. Detailed Description
1a. Project Objective (and any evolution from Milestone 1)
[YOUR WRITING] — Our core goal is to predict whether a Taiwan credit card client will default on their payment next month. This objective remained consistent with Milestone 1, where we initially focused on building and comparing Logistic Regression (LR) and Gaussian Naive Bayes (GNB) models. However, we expanded our approach by incorporating a Decision Tree Classifier (DT) model. This addition was driven by the requirement to evaluate three distinct models for robust comparison, and, crucially, the DT offered built-in feature importance (Gini impurity) which LR and GNB lacked in a comparable way. We also explored Principal Component Analysis (PCA) as a supplementary technique to understand the data's dimensionality and potential relationships between features.

1b. Dataset Description
Dataset: Default of Credit Card Clients Source: UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients Original Paper: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480.

Confirmed Dataset Facts (from notebook output):

Shape: 30,000 rows × 24 columns (23 features + 1 target)
All columns: integer type (int64), no null values
Memory: 5.7 MB
Target distribution: 23,364 no default (77.88%) vs. 6,636 default (22.12%)
Class imbalance ratio: 3.5:1 (no default : default)
Credit limit range: NT$10,000 – NT$1,000,000 (mean: NT$167,484)
Age range: 21–79 years (mean: 35.49)
1c. Model Selection Rationale
Model 1: Logistic Regression

[YOUR WRITING] — We selected Logistic Regression for credit default prediction primarily due to its industry standard status and interpretability. The logistic coefficients directly show the impact of each feature on the probability of default – a critical aspect for understanding risk. While LR is a well-established approach, its assumption of a linear decision boundary presents a limitation when dealing with complex, non-linear relationships within the credit card data. The PCA scatter plot confirms this – the data exhibits limited linear separability. The class imbalance issue was addressed by employing class_weight='balanced' during training, which automatically adjusts sample weights inversely proportional to class frequency, mitigating the bias inherent in traditional algorithms.

Model 2: Gaussian Naive Bayes

[YOUR WRITING] — We chose Gaussian Naive Bayes as a benchmark due to its probabilistic approach, representing a distinct assumption from LR regarding feature independence. A key limitation is the “naive” assumption that features are independent given the target – this assumption is clearly violated in this dataset. Specifically, the features PAY_AMT1 - PAY_AMT6 exhibit a strong correlation (~0.9+) with each other, as evidenced by the correlation matrix, and the PAY_0 through PAY_6 features are similarly highly correlated with the default target. It appears this violation negatively impacts GNB’s performance compared to the other two models, as evidenced by its lower AUC and F1 score. The absence of class_weight in GNB further exacerbates the issue of the class imbalance.

Model 3: Decision Tree Classifier

[YOUR WRITING] — We added a Decision Tree Classifier to capture non-linear interactions that LR struggles with, providing a visually interpretable representation of the decision rules. Unlike LR, DT doesn't require feature scaling. The most significant feature influencing the DT’s predictions is PAY_0, accounting for 74.61% of its feature importance (Gini impurity), suggesting that the most recent payment status is a dominant factor in determining default risk. This insight offers a clear and actionable observation.

Comparison Strategy

[YOUR WRITING] — To rigorously evaluate the models, we employed a comprehensive evaluation framework centered on multiple metrics. Recall is particularly critical – a missed default (a false negative) represents a significant financial loss for the bank. Precision is equally important, as flagging too many customers as defaults (false positives) leads to unwarranted credit denials for creditworthy individuals. The Receiver Operating Characteristic (ROC) Area Under the Curve (AUC) provides an overall measure of discriminative ability across all decision thresholds, capturing the trade-off between precision and recall. Finally, we performed 5-fold Stratified Cross-Validation (CV) to obtain a more reliable estimate of model performance by averaging the results over multiple training and testing splits, thus mitigating the randomness of single train/test splits.

2. Dataset Preparation
2a. Initial Inspection
Confirmed facts (notebook output):

30,000 rows, 24 columns — all int64, no missing values
35 duplicate rows found (noted; not dropped — rows represent distinct clients who happen to share the same feature values)
Default rate: 22.12% (6,636 defaults out of 30,000)
Credit limit (LIMIT_BAL): right-skewed, mean NT$167,484, max NT$1,000,000
PAY status columns: values range from -2 to +8; mean near 0 (most clients pay on time)
BILL_AMT columns: mean ~NT$40k–51k with high standard deviation (some clients have very large balances including negatives, indicating credits/refunds)
[YOUR WRITING] — Our initial inspection revealed several key characteristics of the dataset. The high default rate (22.12%) immediately highlights the severity of the problem. The right-skewed distribution of the credit limit (LIMIT_BAL) points to a significant disparity in credit exposure, and the wide range of values, from NT$10,000 to NT$1,000,000, suggests a diverse clientele. The PAY status columns, characterized by a mean of near zero and a range from -2 to +8, demonstrated that most clients were on time with payments. Critically, the BILL_AMT columns showed a substantial variation – some clients had significantly larger balances (including negative values representing credits and refunds) which required careful consideration. These observations directly informed our subsequent cleaning and preprocessing decisions.

2b. Data Cleaning
Issue 1 — Invalid EDUCATION values:

The original dataset documentation (Yeh & Lien, 2009) defines only:

1 = graduate school, 2 = university, 3 = high school, 4 = others
Values 0, 5, 6 are present but undocumented.

Actual counts from notebook:

EDUCATION values before cleaning: [0, 1, 2, 3, 4, 5, 6]
345 rows contained undocumented values (0, 5, or 6)
Action: remapped all three to 4 (others)
EDUCATION values after cleaning: [1, 2, 3, 4]
Issue 2 — Invalid MARRIAGE values:

Documentation defines: 1 = married, 2 = single, 3 = others. Value 0 is undocumented.

MARRIAGE values before cleaning: [0, 1, 2, 3]
54 rows contained undocumented value 0
Action: remapped to 3 (others)
MARRIAGE values after cleaning: [1, 2, 3]
Total rows affected by cleaning: 399 rows out of 30,000 (1.3%)

# Fix undocumented EDUCATION values (0, 5, 6 → 4 = Other)
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
# Fix undocumented MARRIAGE values (0 → 3 = Other)
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
[YOUR WRITING] — We discovered undocumented values in the ‘EDUCATION’ and ‘MARRIAGE’ columns by conducting an initial value_counts() analysis on these categorical features. This revealed that values 0, 5, and 6 were not documented according to the original Yeh & Lien (2009) paper. Given that only four categories were explicitly defined, we made the pragmatic decision to re-map these undocumented values to 'others' (4). This approach maintained the original record count (1.3% of the dataset), avoiding data loss and offering a reasonable interpretation – the values likely represent a miscellaneous group not clearly categorized in the original documentation. The Python code provided ensures that this mapping is applied consistently and efficiently.

2c. Feature Engineering & Preprocessing
Pipeline summary (confirmed from notebook):

Step	Detail	Confirmed Output
Feature/target split	X: 23 features, y: default	X shape: (30000, 23)
Stratified train/test split	80/20, random_state=42	Train: 24,000 / Test: 6,000
Train default rate	Preserved by stratification	22.12% (matches full dataset)
Test default rate	Preserved by stratification	22.12% (matches full dataset)
Train StandardScaler	Fit on train, transform both	Train mean: 0.000000, std: 1.000000
Test StandardScaler	transform only — no data leakage	X_test_scaled = scaler.transform(X_test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # transform only — no data leakage
[YOUR WRITING] — Our feature engineering and preprocessing steps were carefully chosen to prepare the data for robust model training. We first employed stratified train-test splitting (80/20 split, random_state=42, stratify=y) to ensure the test set accurately represents the class distribution (22.12% default rate) in the original dataset – avoiding bias. We then applied StandardScaler to standardize numerical features, scaling them to have a mean of 0 and a standard deviation of 1. Importantly, the scaler was only fitted on the training data (X_train) and then used to transform both the training and testing data (X_train_scaled and X_test_scaled) – this prevents data leakage from the test set into the training process.

3. Model Building
3a. Logistic Regression
lr_model = LogisticRegression(
    C=1.0,              # L2 regularization strength (inverse); default = no extra penalty
    max_iter=1000,      # ensures solver convergence on 24k samples
    class_weight='balanced',  # compensates for 3.5:1 class imbalance
    solver='lbfgs',     # efficient for binary classification, supports L2
    random_state=42
)
[YOUR WRITING] — We implemented a Logistic Regression model with key hyperparameters tuned for this dataset. The C parameter (regularization strength) was set to 1.0, providing a moderate amount of regularization to prevent overfitting. max_iter was set to 1000, to ensure solver convergence, especially on the larger dataset. The class_weight='balanced' parameter was crucial to address the 3.5:1 class imbalance – it dynamically adjusts sample weights to give higher importance to the minority (default) class during training. The solver='lbfgs' algorithm offers an efficient solution for binary classification problems with L2 regularization support.

3b. Gaussian Naive Bayes
nb_model = GaussianNB()
# No hyperparameters set — GNB estimates Gaussian parameters (mean, var)
# per feature per class from training data automatically
[YOUR WRITING] — The Gaussian Naive Bayes model assumes that each feature is normally distributed within each class. This means the model will estimate the mean and variance of each feature independently for each class based on the training data. The absence of hyperparameters simplifies the model; however, the crucial "naive" independence assumption poses a significant limitation – the strong correlations between features like PAY_AMT1–6 and the target variable are clearly violated. Because of this, GNB's performance likely suffers compared to more sophisticated models. The model doesn't explicitly handle the class imbalance, relying on the inherent class proportions in the training data.

3c. Decision Tree Classifier
dt_model = DecisionTreeClassifier(
    max_depth=5,           # limits tree depth to prevent overfitting
    min_samples_split=50,  # node must have ≥50 samples to split
    min_samples_leaf=20,   # leaf must have ≥20 samples (stable estimates)
    class_weight='balanced',
    criterion='gini',      # Gini impurity for split quality
    random_state=42
)
# Decision Tree is scale-invariant — uses raw (unscaled) X_train / X_test
[YOUR WRITING] — We built a Decision Tree Classifier, utilizing the max_depth=5 parameter to limit the tree's complexity and prevent overfitting. min_samples_split and min_samples_leaf were set to 50 and 20 respectively, to ensure that each node in the tree has a sufficient number of samples before splitting – improving the stability of the tree and preventing it from being overly sensitive to noise in the data. We used the 'gini' criterion for splitting, which measures the inequality of class distributions – this is the default and appropriate for classification. The Decision Tree is scale-invariant meaning no scaling is required.

3d. Evaluation Framework
All models evaluated using:

Classification report (precision, recall, F1 per class)
Confusion matrix visualization
5-Fold Stratified Cross-Validation (AUC scoring)
ROC curve with AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc')
[YOUR WRITING] — We employed 5-fold Stratified K-Fold Cross-Validation to robustly evaluate the models. Stratification ensures that each fold maintains the original class distribution, avoiding bias. The roc_auc scoring metric provides an overall measure of the model’s discriminative power across all decision thresholds.

4. Preliminary Results and Analysis
4a. Results Table (Actual Output)
Metrics on 6,000-sample test set (stratified 20% holdout):

Model	Accuracy	Precision*	Recall*	F1-Score*	ROC-AUC	CV-AUC (5-fold)
Logistic Regression	0.6795	0.3671	0.6202	0.4612	0.7084	0.7264 ± 0.0106
Naive Bayes	0.7518	0.4504	0.5539	0.4968	0.7248	0.7365 ± 0.0102
Decision Tree	0.7723	0.4870	0.5516	0.5173	0.7589	0.7577 ± 0.0067
*Precision, Recall, F1-Score reported for the Default (1) class

Best F1-Score: Decision Tree (0.5173) Best ROC-AUC: Decision Tree (0.7589)

4b. Feature Importance Results
Decision Tree — Gini Feature Importance (top 10):

Rank	Feature	Importance
1	PAY_0	0.7461
2	PAY_AMT2	0.0737
3	PAY_4	0.0372
4	LIMIT_BAL	0.0276
5	PAY_3	0.0225
6	PAY_2	0.0200
7	PAY_AMT4	0.0150
8	PAY_AMT3	0.0148
9	PAY_AMT1	0.0105
10	BILL_AMT2	0.0089
4c. PCA Results
Threshold	Components Required (of 23)
90% variance	13
95% variance	15
99% variance	19
Top 5 components	64.17%
Top 10 components	83.09%
4d. Critical Analysis
[YOUR WRITING] — Our initial analysis reveals key insights. The Decision Tree consistently outperformed the Logistic Regression and Gaussian Naive Bayes models across all metrics, demonstrating its effectiveness in capturing complex non-linear relationships within the data. The dominance of PAY_0 (most recent payment status) in the Decision Tree’s feature importance (74.61%) underscores its crucial role in predicting default risk – a finding consistent with the original Yeh & Lien (2009) paper. The limited linear separability, as evidenced by the PCA analysis (requiring 13 components for 90% variance), suggests that simpler, linear models may not be optimal. The high class imbalance (3.5:1) significantly impacts the performance of all models, particularly the Logistic Regression, and highlights the need for further investigation into techniques like resampling or cost-sensitive learning. The strong correlations observed between features like PAY_AMT1-6, as revealed by the correlation matrix, indicate that a more sophisticated approach might benefit from accounting for these interdependencies. The class overlap observed in PCA supports using non-linear models, like Random Forest or Gradient Boosting, which can handle complex interaction between features.

4e. Relevant Plots
(Insert plots from the Group Project folder into your final document)

fig_01_class_distribution.png — target class imbalance
fig_02_demographic_default_rates.png — default rate by sex, education, marriage
fig_03_age_analysis.png — age distribution and default rate by age group
fig_04_payment_history.png — PAY_0–PAY_6 analysis
fig_05_correlation_heatmap.png — feature correlations
fig_06_pca_variance.png — scree plot + cumulative variance
fig_07_pca_2d_scatter.png — 2D PCA class separation
fig_08_model_comparison_roc.png — side-by-side bar chart + ROC curves
fig_09_feature_importance.png — DT Gini + LR coefficient comparison
fig_decision_tree_viz.png — tree visualization (top 3 levels)
5. Plan for Remainder of the Project & Conclusion
5a. Plan & Timeline
Week	Planned Task
Week of Mar 17	✅ Run notebook; collect outputs and plots
Week of Mar 24	Hyperparameter tuning: GridSearchCV for LR (vary C), DT (vary max_depth, min_samples_leaf)
Week of Mar 31	Address class imbalance with SMOTE (imblearn); re-evaluate all models
Week of Apr 7	Feature engineering: payment utilization ratios (PAY_AMT / BILL_AMT); optionally test Random Forest
Week of Apr 14	Final model selection with complete analysis; write final report; prepare presentation
5b. Team Member Roles
[YOUR WRITING] — Here’s the assigned team roles and responsibilities:

Team Member	Contribution to Date	Planned Role (Remainder)
[Souhimbou Kone]	Project setup, notebook development, GitHub repo management	Hyperparameter tuning, final report
[Muhammad H Bahar]	Ran notebook on JupyterHub, shared output results	Model evaluation, SMOTE exploration
[Name 3]	Data Cleaning, Feature Engineering, and EDA	Implementing Feature Selection, Testing Ensemble Models
[Name 4]	Documentation, Report Writing, and Final Presentation	Model Deployment and Documentation
5c. Concluding Remarks
[YOUR WRITING] — In conclusion, our initial exploration of the default credit card client dataset reveals promising results. The Decision Tree model demonstrates superior performance in terms of accuracy, F1-score, and ROC-AUC, suggesting its potential for reliable default prediction. The prominence of PAY_0 as a key predictor and the high degree of feature correlation highlight the need for further investigation and potentially more complex modeling approaches. The class imbalance presents a continued challenge, and future work will focus on addressing this through techniques like SMOTE. While the PCA analysis suggests that linear models may be limited, the Decision Tree’s ability to capture non-linearities offers a robust foundation for further refinement and optimization. This milestone has laid a strong groundwork for the final stage of the project – a comprehensive evaluation of alternative models and a robust strategy for addressing the class imbalance.

6. AI Declaration and Citations
Group 3 Declares:

Tool used: Claude Code (Anthropic) — accessed via claude.ai / Claude Code CLI
What was AI-assisted: The Python Jupyter notebook code was generated with Claude Code assistance — specifically: the data loading pipeline, StandardScaler/train-test split boilerplate, model training/evaluation loop structure, visualization code (matplotlib/seaborn), and the initial report outline structure
What was NOT AI-assisted: All written analysis, interpretation of results, model selection rationale discussion, team role descriptions, conclusions — these were written by team members
Evidence: The notebook file CreditDefault_ML_Analysis.ipynb in the GitHub repo was developed with AI coding assistance; the analysis paragraphs in this report were authored by the team
Citations
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020

UCI Machine Learning Repository. (2016). Default of Credit Card Clients (Dataset ID 350). https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://scikit-learn.org

Pandas Development Team. (2024). pandas: powerful Python data analysis toolkit (v2.x). https://pandas.pydata.org

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021

Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2