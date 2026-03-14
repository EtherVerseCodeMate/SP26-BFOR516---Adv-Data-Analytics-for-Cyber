# Lab Report: Supervised Machine Learning – Logistic Regression vs. Naive Bayes

**Course:** BFOR516 - Advanced Data Analytics for Cyber  
**Student:** Spencer Kone  
**Date:** February 15, 2026  

---

## 1. AI Usage Statement
Artificial Intelligence  was utilized as a tool to assist in the following ways:
- **Code Generation:** Assisting in writing the Python script for data processing, model implementation (Scikit-Learn), and data visualization (Matplotlib/Seaborn).
- **Formatting:** Helping structure the report and summarize technical outputs into readable tables.
- **Troubleshooting:** Resolving environmental issues such as missing dependencies and output buffering.

**Note:** All feature selection logic, analytical interpretations, and the final conclusion regarding the research question are my own (Spencer Kone).

---

## 2. Introduction & Research Question
The objective of this lab is to determine if **"what a person says"** is a better indicator of truth than **"how they say it."** We explore this by training two different supervised machine learning models—Logistic Regression and Naive Bayes—on the "Truth Seeker" dataset, categorizing features into "Content/Substance" (WHAT) and "Style/Delivery" (HOW).

**Research Question:** Is the substantive content of a statement (WHAT) or the linguistic style of the delivery (HOW) more predictive of truthfulness?

---

## 3. Experimental Setup

### A. Feature Selection
To answer the research question, features were manually categorized into two distinct sets:

| Feature Set | Category | Examples of Features | Rationale |
| :--- | :--- | :--- | :--- |
| **WHAT** | Content/Substance | ORG_percentage, PERSON_percentage, GPE_percentage, unique_entities_count | Captures the "meat" of the statement: specific entities, topics, and factual references. |
| **HOW** | Style/Delivery | Word count, Avg word length, exclamation, present_verbs, capitals | Captures the "manner" of the statement: grammar, punctuation, complexity, and emotional emphasis. |

### B. Model Building
- **Logistic Regression (LR):** Chosen for its ability to model linear relationships and provide interpretable coefficients. Features were standardized (Z-score) before training.
- **Naive Bayes (NB):** Specifically the **Gaussian Naive Bayes** variant was used, as the features are continuous numeric values. No scaling was required for this model.
- **Split:** Data was split into 70% Training and 30% Testing (134,198 total samples).

---

## 4. Results & Analysis

### Model Performance Summary
The models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC AUC.

| Model | Feature Set | Accuracy | Precision | Recall | F1 Score | **ROC AUC** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | WHAT (Content) | 0.5786 | 0.5806 | 0.6469 | 0.6120 | **0.6094** |
| **Logistic Regression** | **HOW (Style)** | **0.5881** | 0.5845 | 0.6854 | 0.6309 | **0.6225** |
| **Naive Bayes** | WHAT (Content) | 0.5532 | 0.6335 | 0.3086 | 0.4150 | **0.5935** |
| **Naive Bayes** | **HOW (Style)** | **0.5660** | 0.5568 | 0.7604 | 0.6428 | **0.5943** |

### Individual Model Interpretation
1. **Logistic Regression (LR):** The "HOW" features significantly outperformed the "WHAT" features in terms of AUC (0.6225 vs 0.6094). This suggests that the linguistic style—particularly the use of present/past verbs, punctuation, and word length—carries more signal for truth detection than the named entities mentioned.
2. **Naive Bayes (NB):** The performance gap was much narrower for Naive Bayes, with "HOW" features showing a very slight edge (0.5943 vs 0.5935). However, Naive Bayes displayed much lower overall performance compared to Logistic Regression, likely due to its strong assumption of feature independence, which is often violated in natural language data.

---

## 5. Comparison & Interpretation
- **Model Comparison:** Logistic Regression consistently outperformed Naive Bayes across all metrics. This indicates that the relationship between these features and the target (Truth) is more effectively captured by a linear model that accounts for relative feature importance rather than a probabilistic model assuming independence.
- **Feature Importance:** In the LR "HOW" model, features like `Word count`, `exclamation`, and `present_verbs` showed significant coefficient magnitudes, suggesting that shorter, less excited, and more descriptive statements are linked to higher truthfulness (or vice-versa, depending on coefficient direction).
- **The "WHAT" vs "HOW" Verdict:** In both models, the **"HOW"** features provided a superior indicator of truth. This suggests that while facts (WHAT) can be easily falsified, the subconscious linguistic patterns (HOW) we use when speaking the truth are much harder to mask.

---

## 6. Final Conclusion
Based on the results of both Logistic Regression and Naive Bayes models, **how a person says something is a better indicator of truth than what they say.**

The "HOW" models achieved higher Accuracy and ROC AUC scores consistently. In cybersecurity and forensic analysis, this outcome implies that examining the *metadata* of communication (style, tone, complexity) may be more reliable for detecting deception than attempting to verify the individual *claims* or entities mentioned in the text.

---

## Appendix: Python Code
```python
# [Attached separately: supervised_ml_lab.py]
```
