"""
Week 4 Lab: Unsupervised Learning - Clustering
BFOR516 - Advanced Data Analytics for Cyber
Student: Spencer Kone
Date: February 23, 2026

AI Usage Statement:
- AI tools were used ONLY for code generation assistance.
- All interpretation, results analysis, and conclusions are from
  the student (Spencer Kone).

Research Question:
Are there linguistic patterns that we can observe in data labeled as 'Fake'?

Practice Question:
Are there linguistic patterns unique to tweets labeled as 'Fake'?
"""

import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

def log(msg):
    print(msg)
    sys.stdout.flush()

# =============================================================================
# 1. LOAD AND PREPARE DATA
# =============================================================================
log("Loading dataset...")
df = pd.read_csv('Truth_Seeker_Dataset.csv')
log(f"Dataset shape: {df.shape}")
log(f"Columns: {list(df.columns)}")

# Target column: BinaryNumTarget (1.0 = True, 0.0 = Fake)
target = 'BinaryNumTarget'
df = df.dropna(subset=[target])

log(f"\nTarget distribution:")
log(f"  True (1.0):  {(df[target] == 1.0).sum()}")
log(f"  Fake (0.0):  {(df[target] == 0.0).sum()}")

# =============================================================================
# 2. FEATURE SELECTION - Linguistic Structure (HOW) Features
# =============================================================================
# Selecting linguistic/stylistic features that describe HOW something is said
# (not WHAT is said). These capture grammar, punctuation, complexity, tone.
how_features = [
    'Word count', 'Max word length', 'Min word length', 'Average word length',
    'present_verbs', 'past_verbs', 'adjectives', 'adverbs', 'adpositions',
    'pronouns', 'TOs', 'determiners', 'conjunctions', 'dots', 'exclamation',
    'questions', 'ampersand', 'capitals', 'digits', 'long_word_freq', 'short_word_freq'
]

log(f"\nSelected HOW (Linguistic Structure) Features ({len(how_features)}):")
for f in how_features:
    log(f"  - {f}")

# =============================================================================
# 3. FILTER TO FAKE DATA ONLY & PREPARE
# =============================================================================
log("\n" + "="*70)
log("PART 1: Clustering linguistic patterns in FAKE-labeled data")
log("="*70)

df_fake = df[df[target] == 0.0].copy()
log(f"\nFake data subset: {df_fake.shape[0]} rows")

# Drop any rows with NaN in our features
df_fake = df_fake.dropna(subset=how_features)
log(f"After dropping NaN: {df_fake.shape[0]} rows")

# Extract & scale features
X_fake = df_fake[how_features].fillna(0)
scaler = StandardScaler()
X_fake_scaled = scaler.fit_transform(X_fake)
log("Features scaled using StandardScaler.")

# =============================================================================
# 4. K-MEANS CLUSTERING ON FAKE DATA
# =============================================================================
log("\n--- K-Means Clustering (Fake Data) ---")

# 4a. Elbow Method to find optimal K
inertia = []
sil_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_fake_scaled)
    inertia.append(kmeans.inertia_)
    sil = silhouette_score(X_fake_scaled, labels, sample_size=10000, random_state=42)
    sil_scores.append(sil)
    log(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil:.4f}")

# Elbow plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(k_range, inertia, 'bx-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method - Fake Data (Linguistic Features)')
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, sil_scores, 'rx-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score - Fake Data (Linguistic Features)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fake_kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: fake_kmeans_elbow.png")

# 4b. Run K-Means with best K (use K with highest silhouette)
best_k_idx = np.argmax(sil_scores)
best_k = list(k_range)[best_k_idx]
log(f"\nBest K by Silhouette Score: {best_k} (score={sil_scores[best_k_idx]:.4f})")

kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_fake['KMeans_Cluster'] = kmeans_best.fit_predict(X_fake_scaled)

# 4c. Analyze K-Means clusters
log(f"\nK-Means Cluster Analysis (K={best_k}):")
km_analysis = df_fake.groupby('KMeans_Cluster')[how_features].agg(['mean'])
km_counts = df_fake.groupby('KMeans_Cluster').size().reset_index(name='count')
log(km_counts.to_string(index=False))

# Print key feature means per cluster
log("\nKey feature means per K-Means cluster:")
key_feats = ['Word count', 'Average word length', 'exclamation', 'questions',
             'capitals', 'present_verbs', 'past_verbs', 'adjectives',
             'long_word_freq', 'short_word_freq']
summary = df_fake.groupby('KMeans_Cluster')[key_feats].mean()
log(summary.to_string())

# 4d. K-Means scatter plot
plt.figure(figsize=(10, 6))
for c in range(best_k):
    mask = df_fake['KMeans_Cluster'] == c
    plt.scatter(
        df_fake.loc[mask, 'Word count'],
        df_fake.loc[mask, 'Average word length'],
        label=f'Cluster {c} (n={mask.sum()})',
        alpha=0.5, s=10
    )
plt.xlabel('Word Count')
plt.ylabel('Average Word Length')
plt.title(f'K-Means Clusters (K={best_k}) - Fake Data Linguistic Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fake_kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: fake_kmeans_clusters.png")

# =============================================================================
# 5. HIERARCHICAL CLUSTERING ON FAKE DATA
# =============================================================================
log("\n--- Hierarchical Clustering (Fake Data) ---")

# 5a. Sample for dendrogram (full dataset too large for linkage)
sample_size = min(10000, len(df_fake))
df_fake_sample = df_fake.sample(n=sample_size, random_state=42)
X_fake_sample_scaled = scaler.fit_transform(df_fake_sample[how_features].fillna(0))
log(f"Sampled {sample_size} rows for hierarchical clustering.")

# 5b. Dendrogram
plt.figure(figsize=(15, 7))
plt.title('Dendrogram - Fake Data (Linguistic Features)')
dendrogram = sch.dendrogram(sch.linkage(X_fake_sample_scaled, method='ward'))
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.savefig('fake_hierarchical_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: fake_hierarchical_dendrogram.png")

# 5c. Merge distance plot
Z = sch.linkage(X_fake_sample_scaled, method='ward')
distances = sorted(Z[:, 2], reverse=True)
plt.figure(figsize=(10, 5))
plt.plot(range(1, min(51, len(distances)+1)), distances[:50], 'o-')
plt.title('Merge Distances - Fake Data (Hierarchical Clustering)')
plt.xlabel('Number of Merges')
plt.ylabel('Merge Distance')
plt.grid(True, alpha=0.3)
plt.savefig('fake_hierarchical_merge_distance.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: fake_hierarchical_merge_distance.png")

# 5d. Run Agglomerative Clustering with same K as K-Means for comparison
hc = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
df_fake_sample['HC_Cluster'] = hc.fit_predict(X_fake_sample_scaled)

log(f"\nHierarchical Cluster Analysis (K={best_k}):")
hc_counts = df_fake_sample.groupby('HC_Cluster').size().reset_index(name='count')
log(hc_counts.to_string(index=False))

log("\nKey feature means per Hierarchical cluster:")
hc_summary = df_fake_sample.groupby('HC_Cluster')[key_feats].mean()
log(hc_summary.to_string())

# 5e. Hierarchical scatter plot
plt.figure(figsize=(10, 6))
for c in range(best_k):
    mask = df_fake_sample['HC_Cluster'] == c
    plt.scatter(
        df_fake_sample.loc[mask, 'Word count'],
        df_fake_sample.loc[mask, 'Average word length'],
        label=f'Cluster {c} (n={mask.sum()})',
        alpha=0.5, s=10
    )
plt.xlabel('Word Count')
plt.ylabel('Average Word Length')
plt.title(f'Hierarchical Clusters (K={best_k}) - Fake Data Linguistic Features')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('fake_hierarchical_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: fake_hierarchical_clusters.png")

# =============================================================================
# 6. COMPARISON: K-Means vs Hierarchical on Fake Data
# =============================================================================
log("\n--- K-Means vs Hierarchical Comparison (Fake Data) ---")

# Run KMeans on same sample for apples-to-apples
km_sample = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_fake_sample['KM_Cluster_Sample'] = km_sample.fit_predict(X_fake_sample_scaled)

km_sil = silhouette_score(X_fake_sample_scaled, df_fake_sample['KM_Cluster_Sample'])
hc_sil = silhouette_score(X_fake_sample_scaled, df_fake_sample['HC_Cluster'])
log(f"K-Means Silhouette Score:      {km_sil:.4f}")
log(f"Hierarchical Silhouette Score:  {hc_sil:.4f}")

# Side-by-side cluster size comparison
km_sizes = df_fake_sample.groupby('KM_Cluster_Sample').size().values
hc_sizes = df_fake_sample.groupby('HC_Cluster').size().values
log(f"\nK-Means cluster sizes:      {sorted(km_sizes)}")
log(f"Hierarchical cluster sizes: {sorted(hc_sizes)}")

# Side-by-side scatter
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
for c in range(best_k):
    mask = df_fake_sample['KM_Cluster_Sample'] == c
    ax1.scatter(df_fake_sample.loc[mask, 'Word count'],
                df_fake_sample.loc[mask, 'Average word length'],
                label=f'Cluster {c}', alpha=0.5, s=10)
ax1.set_xlabel('Word Count')
ax1.set_ylabel('Average Word Length')
ax1.set_title(f'K-Means (K={best_k}) - Fake Data')
ax1.legend()
ax1.grid(True, alpha=0.3)

for c in range(best_k):
    mask = df_fake_sample['HC_Cluster'] == c
    ax2.scatter(df_fake_sample.loc[mask, 'Word count'],
                df_fake_sample.loc[mask, 'Average word length'],
                label=f'Cluster {c}', alpha=0.5, s=10)
ax2.set_xlabel('Word Count')
ax2.set_ylabel('Average Word Length')
ax2.set_title(f'Hierarchical (K={best_k}) - Fake Data')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fake_comparison_kmeans_vs_hierarchical.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: fake_comparison_kmeans_vs_hierarchical.png")

# ==============================================================================
# 7. PRACTICE: Repeat analysis for TWEET data
# ==============================================================================
log("\n" + "="*70)
log("PART 2 (Practice): Clustering linguistic patterns in tweets labeled 'Fake'")
log("="*70)

# The tweets have the same linguistic features since those columns describe the
# tweet text itself. We now run the exact same pipeline but use the full data
# with both label types and compare the results, focusing on Fake tweets.

# Re-load fresh for tweet analysis
df_all = pd.read_csv('Truth_Seeker_Dataset.csv')
df_all = df_all.dropna(subset=[target] + how_features)
log(f"Full dataset (after dropping NaN): {df_all.shape[0]} rows")

# Scale
X_all = df_all[how_features].fillna(0)
scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)

# K-Means on all data
log("\n--- K-Means on ALL data (Fake + True) ---")
inertia_all = []
sil_all = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_all_scaled)
    inertia_all.append(km.inertia_)
    sil = silhouette_score(X_all_scaled, labels, sample_size=10000, random_state=42)
    sil_all.append(sil)
    log(f"  K={k}: Inertia={km.inertia_:.2f}, Silhouette={sil:.4f}")

best_k_all_idx = np.argmax(sil_all)
best_k_all = list(k_range)[best_k_all_idx]
log(f"\nBest K for all data: {best_k_all} (Silhouette={sil_all[best_k_all_idx]:.4f})")

km_all = KMeans(n_clusters=best_k_all, random_state=42, n_init=10)
df_all['KMeans_Cluster'] = km_all.fit_predict(X_all_scaled)

# Elbow plot for all data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(k_range, inertia_all, 'bx-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method - All Data (Linguistic Features)')
ax1.grid(True, alpha=0.3)

ax2.plot(k_range, sil_all, 'rx-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score - All Data (Linguistic Features)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('all_kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: all_kmeans_elbow.png")

# Analyze: what % of each cluster is Fake?
log(f"\nK-Means cluster composition (All Data, K={best_k_all}):")
cluster_comp = df_all.groupby('KMeans_Cluster').agg(
    count=('BinaryNumTarget', 'count'),
    mean_target=('BinaryNumTarget', 'mean')
)
cluster_comp['FakeNewsPropensity'] = 1 - cluster_comp['mean_target']
log(cluster_comp.to_string())

log("\nKey linguistic features per cluster (All Data):")
all_summary = df_all.groupby('KMeans_Cluster')[key_feats].mean()
log(all_summary.to_string())

# Scatter: color by cluster, marker by label
plt.figure(figsize=(12, 7))
for c in range(best_k_all):
    mask = df_all['KMeans_Cluster'] == c
    plt.scatter(
        df_all.loc[mask, 'Word count'],
        df_all.loc[mask, 'Average word length'],
        label=f'Cluster {c} (n={mask.sum()}, Fake%={cluster_comp.loc[c,"FakeNewsPropensity"]*100:.1f}%)',
        alpha=0.3, s=8
    )
plt.xlabel('Word Count')
plt.ylabel('Average Word Length')
plt.title(f'K-Means (K={best_k_all}) - All Data (Fake + True)')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.savefig('all_kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: all_kmeans_clusters.png")

# Hierarchical on sampled all data
log("\n--- Hierarchical on ALL data ---")
sample_all_size = min(10000, len(df_all))
df_all_sample = df_all.sample(n=sample_all_size, random_state=42)
X_all_sample_scaled = scaler_all.fit_transform(df_all_sample[how_features].fillna(0))

hc_all = AgglomerativeClustering(n_clusters=best_k_all, linkage='ward')
df_all_sample['HC_Cluster'] = hc_all.fit_predict(X_all_sample_scaled)

# Dendrogram
plt.figure(figsize=(15, 7))
plt.title('Dendrogram - All Data (Linguistic Features)')
sch.dendrogram(sch.linkage(X_all_sample_scaled, method='ward'))
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.savefig('all_hierarchical_dendrogram.png', dpi=150, bbox_inches='tight')
plt.close()
log("Saved: all_hierarchical_dendrogram.png")

# Hierarchical composition
hc_comp = df_all_sample.groupby('HC_Cluster').agg(
    count=('BinaryNumTarget', 'count'),
    mean_target=('BinaryNumTarget', 'mean')
)
hc_comp['FakeNewsPropensity'] = 1 - hc_comp['mean_target']
log(f"\nHierarchical cluster composition (All Data, K={best_k_all}):")
log(hc_comp.to_string())

log("\nKey linguistic features per HC cluster (All Data):")
hc_all_summary = df_all_sample.groupby('HC_Cluster')[key_feats].mean()
log(hc_all_summary.to_string())

# Comparison on all data sample
km_s = KMeans(n_clusters=best_k_all, random_state=42, n_init=10)
df_all_sample['KM_Cluster_Sample'] = km_s.fit_predict(X_all_sample_scaled)

km_sil_all = silhouette_score(X_all_sample_scaled, df_all_sample['KM_Cluster_Sample'])
hc_sil_all = silhouette_score(X_all_sample_scaled, df_all_sample['HC_Cluster'])
log(f"\nAll Data Comparison:")
log(f"K-Means Silhouette Score:      {km_sil_all:.4f}")
log(f"Hierarchical Silhouette Score:  {hc_sil_all:.4f}")

log("\n" + "="*70)
log("ANALYSIS COMPLETE")
log("="*70)
log("All plots saved. See Lab Report for interpretation.")
