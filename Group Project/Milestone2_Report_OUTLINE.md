BFOR 516 – Group Project Milestone 2: Progress Report
Title: Predicting Credit Card Defaults using Machine Learning Techniques
Team Members: Souhimbou Kone, Muhammad H Bahar, Name 3, Name 4
Date: March 2026


1. Detailed Description

1a. Project Objective (and any evolution from Milestone 1)

This task is same as that of Milestone 1. Using the Decision Tree Classifier (DT) algorithm, we are required to train the model that can predict the probability of default or non-payment by the Taiwan credit card clients next month. Out of the three models compared in this task, we chose to implement the Decision Tree Classifier. Upon examining the feature importance of the Decision Tree Classifier, we found that it uses Gini impurity to calculate the feature importance which is not available for the Logistic Regression and the Gaussian Naive Bayes algorithms in the same way. The understanding of the dimensionality of the data and the relationships within the features is also required for modeling. Therefore, we used the Principal Component Analysis (PCA) to achieve the same.

1b. Dataset Description

Dataset: Default of Credit Card Clients  
Source: UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients  
Original Paper: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480.

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

Credit Default Model To train the credit default model we have used the Logistic Regression algorithm. This is the most popular algorithm for credit scoring in the credit industry. The coefficients of the variables are easy to interpret. They tell us by how much the probability of default will change when the variable is changed by one unit. The Logistic Regression assumes that the model can be described with a linear decision boundary. This assumption is clearly not fulfilled in the credit card data from the PCA scatter plot. To balance the classes in the data we had to specify the class_weight='balanced' parameter in the Logistic Regression classifier. The class weights are usually chosen to be inversely proportional to the class frequencies to avoid the majority class dominating in unweighted models.

Model 2: Gaussian Naive Bayes

I have chosen to use the Gaussian Naive Bayes (GNB) algorithm for this benchmark. As mentioned before, logistic regression is a deterministic algorithm, which means that it models the relationship between features and target using a linear equation. On the other hand, Naive Bayes is a probabilistic algorithm which uses Bayes’ theorem to compute the probability of features given the target. The assumption made by the Naive Bayes algorithm is that all the features are independent given the target. It is quite obvious from the correlation matrix of PAY_AMT1–PAY_AMT6 that the assumption of feature independence is not satisfied. The correlation coefficients are all above 0.9 and the coefficients of PAY_0–PAY_6 are high and correlated with the target. So, the performance of the GNB is not good, and it is decreased when the class_weight parameter is not provided.

Model 3: Decision Tree Classifier

Since our features were not linearly separable with respect to the Logistic Regression model, we used Decision Tree Classifier instead. The decision tree is also shown below. Unlike Logistic Regression, Decision Trees do not need the features to be scaled. We used the Gini impurity for the Decision Tree Classifier. PAY_0 has the highest contribution to the classifier with a feature importance of 74.61%.

Comparison Strategy

The method used to model evaluation is that the performance of the proposed models were evaluated with various performance metrics. Since the loss for the bank is more sensitive to false negatives than false positives, recall is more important than precision. The precision is still important to the bank because of the false positives which means that true customers with good credit scores are being declined credit. The criterion which combines precision and recall at all possible operating points is the Receiver Operating Characteristic (ROC) Area Under the Curve (AUC). In the experimentation, 5-fold Stratified Cross Validation (CV) was used in order to obtain a more realistic view of the model performance.

2. Dataset Preparation

2a. Initial Inspection

Confirmed facts (notebook output):

30,000 rows, 24 columns — all int64, no missing values  
35 duplicate rows found (noted; not dropped — rows represent distinct clients who happen to share the same feature values)  
Default rate: 22.12% (6,636 defaults out of 30,000)  
Credit limit (LIMIT_BAL): right-skewed, mean NT$167,484, max NT$1,000,000  
PAY status columns: values range from -2 to +8; mean near 0 (most clients pay on time)  
BILL_AMT columns: mean ~NT$40k–51k with high standard deviation (some clients have very large balances including negatives, indicating credits/refunds)

After having a first look at the whole dataset we have noticed some interesting characteristics in the data:

The default rate is quite high and is 22.12%.

The Credit limit (LIMIT_BAL) is right skewed and there is a high variance of credit given to the customers. The minimum value of this feature is 10000 (1000 USD) and the maximum value is 1000000 (100000 USD).

The mean of the PAY status columns is centred close to zero and is located within the range of -2 to 8. This indicates that most customers pay their invoices on time.

The Bills (BILL_AMT) dataset contains some high and some low values. There are obviously large discrepancies between the charges on the bills for different customers. Some charges appear to be negative, which probably represents credits or refunds being applied to the customer’s account.

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

The first value_counts() for categorical variables show a large number of undefined values in EDUCATION and MARRIAGE columns. According to the study of Yeh & Lien (2009), there are 4 categories of educational level and marital status. The undefined values coded as 0, 5 and 6 should represent the other categories and we replace them with category 4. The record loss for the variables is less than 1.3% and we assume that the missing values represent the category that the authors call “miscellaneous” without further explanation.

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

Since we have the dataset we need to prepare our data so that our predictive model will be able to learn as much as possible from it. So in this stage of the workflow we simply need to prepare our data in the best way possible that we can learn from it. In this stage we just make sure that we have enough and the right data to train a good model. The training and test set were already split using the stratified train-test split with ratio 80:20 and random_state = 42, stratify = y. This ensures that there is no bias in the default rate (22.12%) of the original dataset in the test set. The numerical variables in the dataset were scaled using the StandardScaler. In this case the StandardScaler was fitted to the training data (X_train). The training and test data (X_train_scaled and X_test_scaled) were then standardised to make sure that there was no leakage of information from the test set.

3. Model Building

3a. Logistic Regression

lr_model = LogisticRegression(
    C=1.0,              # L2 regularization strength (inverse); default = no extra penalty
    max_iter=1000,      # ensures solver convergence on 24k samples
    class_weight='balanced',  # compensates for 3.5:1 class imbalance
    solver='lbfgs',     # efficient for binary classification, supports L2
    random_state=42
)

A Logistic Regression model with the hyperparameters optimized for this dataset. We choose to set these hyperparameters to medium or moderate values. We have chosen to set the regularization parameter C = 1.0 in order not to over fit the training data. We set max_iter = 1000 in order to allow the solver to converge, particularly with the larger dataset. We have set class_weight='balanced' to counter balance the class imbalance. The class weights are then updated for every fold, and are dynamic and changing based on the class distribution in the training data. The solver='lbfgs' is a more efficient choice for binary logistic regression with L2 regularization.

3b. Gaussian Naive Bayes

nb_model = GaussianNB()
# No hyperparameters set — GNB estimates Gaussian parameters (mean, var)
# per feature per class from training data automatically

Here we assume that all features are normally distributed in each class. The model calculates the means and variances for each feature and class. Hyperparameter tuning is not needed for this implementation. The “naive” in the name of the model refers to the assumption that all features are independent. As we have an idea from the feature correlations and feature importance to the target variable that this assumption is highly violated in our case, the GNB model should also perform worse. The class imbalance is not handled explicitly. It just uses the class distribution of the training data.

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

Decision Tree Classifier from scratch We have already seen the Decision Tree Classifier from scratch. It is very basic classifier and we need to take care of overfitting and other parameters to increase the accuracy of the classifier. So, we will use max_depth=5. We also need to consider min_samples_split and min_samples_leaf to ensure that each node has sufficient samples before splitting. So we will use min_samples_split = 50 and min_samples_leaf = 20. Criterion is the feature or measure used to select the best split. The ‘gini’ criterion is measure of the inequality of class distributions. It is the default criterion for classification. Decision Tree is scale-invariant so no scaling is required.

3d. Evaluation Framework

All models evaluated using:

Classification report (precision, recall, F1 per class)
Confusion matrix visualization
5-Fold Stratified Cross-Validation (AUC scoring)
ROC curve with AUC

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='roc_auc')

I applied 5-fold Stratified K-Fold Cross-Validation to the data. “Stratified” means that the split holds the class distribution of the original data constant, and the reason for this is that we want the training and test sets to have the same class distribution as in the original data. I used “roc_auc” to evaluate the model performance for all possible classification thresholds.

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

This section presents the initial results obtained for this study. As it can be observed from the Table 1, the accuracy obtained by the Decision Trees model is higher than that of Logistic Regression and the Gaussian Naive Bayes. In addition, the most important feature of the Decision Trees model was calculated and the results showed that PAY_0 (most recent payment status) had the highest importance with a value of 74.61% which is in agreement with the findings of Yeh & Lien (2009). From the PCA, it can be inferred that the variables in the data set are not linearly separable and the need for 13 components to explain about 90% of the data proves that the Decision Trees model that can handle non-linearities will perform better than the linear model which in this case is the Logistic Regression. Another factor that may affect the models is the class imbalance which in this case has a ratio of about 3.5:1. Class imbalance in classes with limited samples can be handled using class resampling techniques and cost-sensitive learning algorithms. As can be inferred from the correlation matrix above, there are high correlations among the variables in the data set. These correlations specifically among the PAY_AMT1–6 variables need to be considered in the modeling. As can be inferred from the class overlap in the PCA, the non-linear models that can handle correlations such as Random Forest and Gradient Boosting will work better for this classification problem.

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

Here’s the assigned team roles and responsibilities:

Team Member	Contribution to Date	Planned Role (Remainder)
1) Spencer Kone:
Setup of the project, implementation of the notebook and management of the GitHub repository	Hyperparameter tuning for the final report
2) Muhammad H Bahar:
Ran a notebook on JupyterHub, shared the output of the results	Model evaluation, exploring SMOTE

3) Kunapureddy, Leela Pavan Kumar:	
Data Cleaning, Feature Engineering, and EDA	Implementing Feature Selection, Testing Ensemble Models

4) Maddirala, Shalem Raju:
Documentation, Report Writing, and Final Presentation, Model Deployment and Documentation


5c. Concluding Remarks

In the previous report we took a first look at the default credit card client in our dataset by applying a Decision Tree algorithm. And it turned out that the model was doing very well. In this report we will have a closer look at the characteristics of our dataset. It starts with the variable PAY_0. After studying this variable we find out that it is also a very significant variable. We will also have a closer look at the correlations between our variables. We know that there are a lot of correlations between the variables. This also needs some extra investigation and maybe using a different model. In this report we will therefore first apply SMOTE to see if this will be sufficient to handle the class imbalance in our dataset. Next we will apply the PCA to see if it will help us to get a better understanding of our data set. Although the classes were classified very well by the decision tree, we feel that we will be able to come up with better models. And we feel that some refinement and tuning is needed to this project. Most of the work for this project is performed in this report. In the next section we will apply some other classification models to see if we can come up with a better model and whether we have found a better method to handle the imbalance in the classes.

6. AI Declaration and Citations

Group 3 Declares:

Tool used: Claude Code (Anthropic) — claude.ai, Google Antigravity IDE.

We applied our analysis and critical thinking skills on the data provided to us by using Claude Code. We generated the Python Jupyter notebook code that was resulting from our request to Claude Code for loading the data, the standard scaling and train-test split boilerplate, for training and evaluating the models and finally for creating the visualisations that we required from the matplotlib and seaborn libraries. We also applied some parts of Claude Code to prepare the basic structure of this report.



Citations

Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473–2480. https://doi.org/10.1016/j.eswa.2007.12.020

UCI Machine Learning Repository. (2016). Default of Credit Card Clients (Dataset ID 350). https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://scikit-learn.org

Pandas Development Team. (2024). pandas: powerful Python data analysis toolkit (v2.x). https://pandas.pydata.org

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021. https://doi.org/10.21105/joss.03021

Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2