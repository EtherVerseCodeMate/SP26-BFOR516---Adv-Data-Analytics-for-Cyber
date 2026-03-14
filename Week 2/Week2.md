Rule of Engagement:
Please find attached the dataset you will work with, and instructions for this weeks lab. You are permitted to use AI tools to seek help with code generation/debugging. However, the interpretation and program logic must be your own. Please write detailed comments on how AI tools were used to help assist with this assignment.


Week 2 Lab
Principal Component Analysis
In this lab, you will perform Principal Component Analysis (PCA) on a dataset containing features for predicting fake vs. real news. You will explore how PCA reduces dimensionality, visualize the results, and interpret which features contribute most to distinguishing classes.
The dataset is called truth_seeker created by the University of New Brunswick: https://www.unb.ca/cic/datasets/truthseeker-2023.html
You can find the dataset on Brightspace under Assignments.
Instructions (You can find steps 1 and 2 under the “Starter Code” after the instructions):
1. Load and prepare the dataset
○ Load the file Features_For_Traditional_ML_Techniques.csv.
○ Remove non-numeric columns and the target column (BinaryNumTarget) to create your feature matrix X.
○ Fill any missing values with 0.
2. Standardize the features
○ Standardize the dataset using StandardScaler.
○ Explain why standardization is important before performing PCA.
3. Apply PCA
○ Reduce the dimensionality of your dataset using PCA while retaining 90% of variance.
○ Transform your standardized features into principal components.
4. Visualize the PCA results
○ Create a 2D scatter plot of the first two principal components.
○ Color the points by the target variable (0: Fake, 1: Real) to see if PCA separates the classes.
○ Save the plot as truth_seeker.png.
5. Analyze explained variance
○ Print the number of components kept.
○ Print the explained variance ratio of each component.
○ Compute and print the total variance captured.
○ Did PCA reduce the data effectively without throwing away too much information?
6. Interpret feature contributions
○ Examine the loadings of the first principal component (PC1).
i. Hint: use pca.components_[0] for getting loadings for PC1, pca.components_[1] for PC2 and so on…
Deliverables:
Copy your code, plot generated, and output (number of components kept, feature contribution analysis, and top 5 features with fake news) into a pdf file and submit by the deadline.
Started Code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# load the dataset
df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')
# prepare the data
# Remove non-numeric columns and target
cols_to_drop = ['majority_target', 'statement', 'BinaryNumTarget', 'tweet', 'embeddings', 'following']
X_raw = df.drop(columns=cols_to_drop, errors='ignore')
# X_raw is the feature matrix used for PCS
# PCA is undupervised, it must not see the target (BinaryNumTarget)
y = df['BinaryNumTarget'] # this is used to interpret PCA results later through viz and interpretation
# handle any nan values that might break the scaler
X_raw = X_raw.fillna(0)
# standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)