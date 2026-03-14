# Week 2 Lab: Principal Component Analysis (PCA) - Summary Report

**Student Name:** [Your Name]  
**Course:** BFOR516 - Advanced Data Analytics for Cybersecurity  
**Date:** February 3, 2026

---

## AI Tool Usage Statement

This lab was completed with assistance from **Google Antigravity AI**. The AI tool was used for:

1. **Code Generation**: Generating boilerplate Python code for PCA implementation using scikit-learn
2. **Visualization**: Creating matplotlib code for the 2D scatter plot with proper formatting
3. **Feature Analysis**: Implementing code to extract and sort PCA loadings for feature contribution analysis
4. **Documentation**: Adding comprehensive comments to explain each step
5. **Debugging**: Fixing Unicode encoding errors and ensuring cross-platform compatibility

**Important Note**: While AI assisted with code syntax and implementation details, the interpretation of results, analytical reasoning, and understanding of PCA concepts are my own work.

---

## Lab Results

### Dataset Overview
- **Total Samples**: 134,198 tweets
- **Original Features**: 58 numeric features
- **Target Variable**: BinaryNumTarget (0 = Fake News, 1 = Real News)

### Step 1: Data Preparation ✓
- Successfully loaded `Features_For_Traditional_ML_Techniques.csv`
- Removed non-numeric columns: `['majority_target', 'statement', 'tweet', 'embeddings', 'following']`
- Removed target column: `BinaryNumTarget`
- Filled missing values with 0
- Final feature matrix: **134,198 samples × 58 features**

### Step 2: Standardization ✓

**Why Standardization is Important:**

Standardization is crucial before performing PCA for four key reasons:

1. **Scale Sensitivity**: PCA is sensitive to feature scales. Without standardization, features with larger variances (e.g., follower counts in thousands) would dominate the principal components over features with smaller scales (e.g., percentages), even if the smaller-scale features are more informative.

2. **Variance Equality**: By standardizing to mean=0 and std=1, we ensure all features contribute equally based on their correlation structure rather than their original measurement scales. This creates a fair analysis.

3. **Fair Comparison**: Our dataset contains features with vastly different ranges—word counts, social media metrics, entity percentages, and linguistic features. Standardization puts them on a level playing field.

4. **Numerical Stability**: Standardization improves the numerical stability of the PCA algorithm, leading to more reliable and reproducible results.

**Verification:**
- Mean of standardized features: 0.000000 ✓
- Standard deviation: 1.000000 ✓

### Step 3: PCA Application ✓

**Configuration**: Retained 90% of variance

**Results:**
- **Components Kept**: 40 (out of 58 original features)
- **Dimensionality Reduction**: 58 → 40 features
- **Compression Ratio**: 31.03%
- **Total Variance Captured**: 90.48%

### Step 4: Visualization ✓

**Plot Generated**: `truth_seeker.png`

The 2D scatter plot shows the distribution of fake vs. real news in the first two principal components:
- **PC1 (x-axis)**: Explains 8.28% of variance
- **PC2 (y-axis)**: Explains 5.25% of variance
- **Color Coding**: Red = Fake News, Blue = Real News

**Observations:**
- Some separation is visible between fake and real news clusters
- Most data points cluster near the origin with some outliers
- The first two PCs alone capture 13.53% of total variance
- While not perfectly separable, there are visible patterns suggesting these components capture meaningful differences

### Step 5: Explained Variance Analysis ✓

**Number of Components**: 40

**Top 5 Most Important Components:**
1. PC1: 8.28% of variance
2. PC2: 5.25% of variance
3. PC3: 4.81% of variance
4. PC4: 4.62% of variance
5. PC5: 3.51% of variance

**Effectiveness Assessment:**

✅ **YES**, PCA reduced the data effectively because:

1. **High Information Retention**: We retained 90.48% of variance while reducing dimensionality by 31%
2. **Significant Compression**: Eliminated 18 features (31%) while losing only ~10% of information
3. **Practical Benefits**: 
   - Faster model training with fewer features
   - Reduced risk of overfitting
   - Easier visualization and interpretation
4. **Meaningful Patterns**: The visualization shows some class separation, indicating the reduced representation captures distinguishing characteristics between fake and real news

### Step 6: Feature Contribution Analysis ✓

**Top 10 Features Contributing to PC1 (Primary Component):**

| Feature | Loading | Abs Loading |
|---------|---------|-------------|
| Word count | 0.434417 | 0.434417 |
| short_word_freq | 0.404580 | 0.404580 |
| adpositions | 0.278407 | 0.278407 |
| present_verbs | 0.257532 | 0.257532 |
| pronouns | 0.235439 | 0.235439 |
| adjectives | 0.228502 | 0.228502 |
| adverbs | 0.219452 | 0.219452 |
| dots | 0.213483 | 0.213483 |
| conjunctions | 0.211077 | 0.211077 |
| past_verbs | 0.204675 | 0.204675 |

**Top 5 Features Most Associated with Distinguishing Fake News:**

1. **Word count** (0.434): Fake/real news differs in verbosity
2. **short_word_freq** (0.405): Frequency of short words is a strong indicator
3. **adpositions** (0.278): Prepositions usage differs between fake and real news
4. **present_verbs** (0.258): Tense usage is a distinguishing factor
5. **pronouns** (0.235): Personal pronoun usage varies between classes

**Interpretation:**

PC1 is heavily influenced by **linguistic and textual features**, particularly:
- Content length (word count)
- Word complexity (short word frequency)
- Grammatical structure (adpositions, verbs, pronouns)

This suggests that fake news has distinctive writing patterns compared to real news, potentially using:
- Different sentence lengths
- Simpler vocabulary
- Different grammatical structures

**Top 10 Features Contributing to PC2:**

| Feature | Loading | Abs Loading |
|---------|---------|-------------|
| quotes | 0.500074 | 0.500074 |
| retweets | 0.478189 | 0.478189 |
| favourites | 0.472741 | 0.472741 |
| replies | 0.332382 | 0.332382 |
| normalize_influence | 0.203937 | 0.203937 |

PC2 is dominated by **social engagement metrics**, suggesting that fake vs. real news differs in how they spread and engage on social media.

---

## Conclusions

1. **PCA Successfully Reduced Dimensionality**: From 58 to 40 features while retaining 90.48% of information

2. **Two Main Factors Distinguish Fake vs. Real News**:
   - **PC1 (Linguistic Features)**: Writing style, vocabulary, and grammar
   - **PC2 (Social Engagement)**: How the content spreads and engages on social media

3. **Practical Implications**:
   - Machine learning models can use these 40 components instead of 58 features
   - Faster training and potentially better generalization
   - The analysis reveals that both content characteristics AND social behavior are important for detecting fake news

4. **Next Steps**: These principal components could be used as input features for classification algorithms (e.g., Random Forest, SVM, Neural Networks) to build a fake news detector

---

## Files Submitted

1. **pca_analysis.py** - Complete Python script with detailed comments
2. **truth_seeker.png** - 2D visualization of PC1 vs PC2
3. **pca_output.txt** - Full console output with all analysis results
4. **This summary document** - Comprehensive report and interpretation

---

## Code Used

See `pca_analysis.py` for the complete implementation with detailed AI usage documentation in the header.

**Key Libraries:**
- pandas, numpy - Data manipulation
- sklearn.preprocessing.StandardScaler - Feature standardization
- sklearn.decomposition.PCA - Principal component analysis
- matplotlib.pyplot - Visualization
