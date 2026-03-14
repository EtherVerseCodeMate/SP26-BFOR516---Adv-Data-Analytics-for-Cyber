# Lab Report: Unsupervised Machine Learning - Clustering
**Course:** BFOR516 - Advanced Data Analytics for Cyber  
**Student:** Spencer Kone  
**Date:** February 23, 2026

## AI Usage Statement
AI tools were used ONLY for code generation and visualization assistance. All interpretations, data analysis, and conclusions presented in this report were developed and written by the student (Spencer Kone).

---

## 1. Research Question
**Are there linguistic patterns that we can observe in data labeled as 'Fake'?**

The objective of this analysis is to identify distinct stylistic groups within fake news data. Instead of looking at *what* is being said (entities, topics), we focus on *how* it is being said (linguistic style) using features like word count, part-of-speech frequencies, and punctuation usage.

---

## 2. Feature Selection
For this clustering analysis, I selected **21 linguistic structure ("HOW") features**. These features were chosen because they capture the "fingerprint" of the text's delivery rather than its content:

- **Lexical Stats:** Word count, Average word length, Max/Min word length.
- **Syntactic Features:** Frequency of present/past verbs, adjectives, adverbs, pronouns, and conjunctions.
- **Punctuation & Style:** Usage of exclamation marks, question marks, capitals, digits, and ampersands.
- **Complexity:** Frequency of long words vs. short words.

By scaling these features using `StandardScaler`, we ensure that features with large ranges (like word count) do not disproportionately influence the distance-based clustering algorithms.

---

## 3. Findings - Part 1: Fake Data Clustering

### 3.1 K-Means Clustering Results
The **Elbow Method** and **Silhouette Analysis** suggested an optimal **K=3**. Three distinct linguistic "styles" emerged within the fake-labeled data:

1.  **Cluster 0 - "The Formal Outliers" (n=438):** A small but highly distinct group. These segments are very long (avg. 79 words), use highly complex vocabulary (avg. word length 9.0), and extreme capitalization (avg. 76 capitals). These likely represent long snippets of text or technical propaganda.
2.  **Cluster 1 - "The Brief/Minimalist" (n=28,839):** Short, simple statements (avg. 24 words). Low use of adjectives and verbs. This represents "headline-style" or "punchy" fake news.
3.  **Cluster 2 - "The Elaborate Sensationalist" (n=35,991):** The largest group. Medium length (avg. 45 words), with high punctuation (exclamations/questions) and a high frequency of adjectives and present-tense verbs. This style matches the "sensationalist" tone often associated with misinformation.

### 3.2 Hierarchical Clustering Results
Using **Ward's Linkage**, the dendrogram confirmed three primary branches. The results broadly mirrored K-Means but highlighted the hierarchical nature of the data:
- The "Formal Outliers" (Cluster 2 in HC) were the first to be separated from the rest, confirming their high distance from typical "tweet-style" content.
- The two larger clusters (Short vs. Medium length) showed significant overlap but were distinct based on their complexity scores.

### 3.3 Method Comparison
| Metric | K-Means (K=3) | Hierarchical (K=3) |
| :--- | :--- | :--- |
| **Silhouette Score** | 0.1620 | 0.0990 |
| **Stability** | High | Medium (Sample-dependent) |

**Conclusion:** K-Means produced tighter, more mathematically distinct clusters for this high-dimensional linguistic data. However, Hierarchical clustering was invaluable for visualizing how the "Formal Outlier" group is fundamentally different from all other data points.

---

## 4. Findings - Part 2 (Practice Task): Tweet Linguistic Patterns
I repeated the analysis on the full dataset (containing both 'True' and 'Fake' labels) to see if 'Fake' tweets cluster together.

### 4.1 Cluster Characteristics
K-Means identified **K=2** as optimal for the full dataset:
-   **Cluster 0 (52% Fake):** "The Intensive Style." Defined by longer word counts (45), higher capitalization (14.6), and more frequent use of adjectives and punctuation.
-   **Cluster 1 (44% Fake):** "The Simple Style." Shorter (23 words) and grammatically simpler.

### 4.2 Interpretation
The data shows that **misinformation (Fake news) is more likely to fall into the "Intensive" linguistic cluster.** Fake news segments in this dataset tend to be slightly longer and use more stylistic markers (punctuation, capitals, descriptors) than true news segments. 

While the "Simple Style" also contains fake news, the higher concentration of misinformation in Cluster 0 suggests that a "verbose and emotional" delivery is a common linguistic pattern for fake data in this collection.

---

## 5. Overall Conclusion
The unsupervised clustering analysis reveals three distinct linguistic signatures in fake news:
1.  **Mass-Market Sensationalism:** Dense, adjective-heavy, and punctuated content.
2.  **Short-Form Misinformation:** Minimalist, headline-style statements.
3.  **Technical/Propaganda Outliers:** Highly complex, long-form content.

The comparative analysis shows that both K-Means and Hierarchical methods agree on the presence of these styles, but K-Means provides better grouping for large-scale linguistic feature sets. The findings suggest that truthfulness can be partially signaled by the **intensity and complexity** of the delivery style.
