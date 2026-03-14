# Lab Report: Neural Networks — Truth Seeker Dataset
**Course:** BFOR516 - Advanced Data Analytics for Cyber  
**Student:** Spencer Kone  
**Date:** March 8, 2026

## AI Usage Statement
AI tools (Antigravity/Gemini) were used ONLY for code generation assistance. All model design decisions, interpretation, results analysis, and conclusions in this report were developed and written by the student (Spencer Kone).

---

## 1. Objective
Build two neural network models to classify tweets as **true or false** using the Truth Seeker Dataset:
- **Model 1:** Linguistic features (text content + writing style metrics)
- **Model 2:** User behavior features (follower counts, bot scores, etc.)

---

## 2. Data Preparation

### 2.1 Dataset
- Total samples: 134,198
- True (1.0): 68,930 (51.4%)
- False (0.0): 65,268 (48.6%)

### 2.2 Feature Engineering
**Model 1** combined three sources of information:
- TF-IDF of `tweet` text (500 features)
- TF-IDF of `statement` text (500 features)
- 20 numeric linguistic features (word count, verb frequencies, punctuation, etc.)
- Total: ~1,020 input features

**Model 2** used 9 numeric features:
- `followers_count`, `friends_count`, `favourites_count`, `statuses_count`
- `BotScore`, `cred`, `normalize_influence`, `replies`, `retweets`

### 2.3 Preprocessing
- **NaN handling:** Missing values filled with 0 (numeric) or empty string (text)
- **StandardScaler:** Applied to all numeric features to normalize magnitude differences
- **Train/Test split:** 80/20, stratified by target variable

---

## 3. Model Architecture

| Component | Model 1 (Linguistic) | Model 2 (User Behavior) |
|:---|:---|:---|
| Input features | ~1,020 | 9 |
| Hidden Layer 1 | Dense(256, ReLU) + Dropout(0.3) | Dense(64, ReLU) + Dropout(0.3) |
| Hidden Layer 2 | Dense(128, ReLU) + Dropout(0.3) | Dense(32, ReLU) + Dropout(0.2) |
| Hidden Layer 3 | Dense(64, ReLU) + Dropout(0.2) | Dense(16, ReLU) + Dropout(0.1) |
| Hidden Layer 4 | Dense(32, ReLU) | Dense(8, ReLU) |
| Output | Dense(1, Sigmoid) | Dense(1, Sigmoid) |
| Optimizer | Adam | Adam |
| Loss | Binary Crossentropy | Binary Crossentropy |
| Epochs | 50 | 50 |
| Batch Size | 64 | 64 |

### Architecture Rationale
- Model 1 uses a wider network (256 neurons first layer) because it receives ~1,020 TF-IDF and numeric inputs. A wide first layer allows the network to learn diverse associations across a large, sparse feature space before progressively compressing them.
- Model 2 uses a narrower network (64 neurons first layer) because it only has 9 input features. Using a wide first layer like 256 on only 9 inputs would create roughly 2,300 parameters per feature — extreme overparameterization that would memorize training noise rather than learn patterns.
- Dropout layers were added to randomly deactivate neurons during training, preventing any single neuron from becoming over-reliant on specific features. This is especially important for Model 1's high-dimensional TF-IDF input, where rare tokens could cause overfitting.

---

## 4. Results

### 4.1 Model 1: Linguistic Features

#### Run 1 — Initial Results (before remediation)
- **Test Accuracy:** 0.9973 (99.73%)
- **Test Loss:** 0.0190

**Confusion Matrix (Run 1):**

|  | Predicted False | Predicted True |
|:---|:---|:---|
| **Actual False** | 13,010 (TN) | 44 (FP) |
| **Actual True** | 29 (FN) | 13,757 (TP) |

Upon reviewing these results, a methodological issue was identified: the TF-IDF vectorizers were being fit on the **entire dataset** before the train/test split. This means the vocabulary and term-frequency statistics were computed using tokens from what would later become the test set — a form of data leakage. The model had indirect exposure to test-set text during feature construction.

**Decision:** Correct the pipeline so that TF-IDF vectorizers and StandardScaler are fit exclusively on training data, then applied (transform only) to the test set. The train/test split was moved to before any fitting step.

#### Run 2 — Corrected Results (after remediation)
- **Test Accuracy:** 0.9972 (99.72%)
- **Test Loss:** 0.0274

**Confusion Matrix (Run 2 — Final):**

|  | Predicted False | Predicted True |
|:---|:---|:---|
| **Actual False** | 13,010 (TN) | 44 (FP) |
| **Actual True** | 31 (FN) | 13,755 (TP) |

- True Negatives: 13,010 — correctly identified false claims
- False Positives: 44 — false claims predicted as true
- False Negatives: 31 — true claims predicted as false
- True Positives: 13,755 — correctly identified true claims

The accuracy changed by only 0.01 percentage points (99.73% → 99.72%). This revealed a deeper structural issue in the dataset: the same `statement` text appears across many rows (multiple tweets referencing the same fact-checked claim). Because the row-level 80/20 split does not account for this, the same statement phrases appear in both train and test sets regardless of TF-IDF fitting order. The high accuracy persists because the model is partially recognizing repeated statements rather than generalizing to truly unseen claims.

**Decision:** Rerun with a statement-level split — split the 1,058 unique statements 80/20, then assign all rows belonging to a training statement to train, and all rows belonging to a test statement to test. No statement text seen during training appears in the test set.

#### Run 3 — Statement-Level Split (Most Rigorous)
- **Test Accuracy:** 0.7722 (77.22%)
- **Test Loss:** 15.6946

**Confusion Matrix (Run 3 — Final):**

|  | Predicted False | Predicted True |
|:---|:---|:---|
| **Actual False** | 8,849 (TN) | 4,158 (FP) |
| **Actual True** | 1,826 (FN) | 11,435 (TP) |

- True Negatives: 8,849 — correctly identified false claims
- False Positives: 4,158 — false claims predicted as true (32% of actual false)
- False Negatives: 1,826 — true claims predicted as false (14% of actual true)
- True Positives: 11,435 — correctly identified true claims
- Precision (False): 0.83 | Recall (False): 0.68
- Precision (True): 0.73 | Recall (True): 0.86

The accuracy drop from 99.72% → 77.22% confirms the earlier hypothesis: most of the prior accuracy came from the model recognizing repeated statement text, not from genuine linguistic pattern learning. The true generalization accuracy on unseen claims is 77.22%, and the very high test loss (15.69) indicates the model produces overconfident probability scores on claims it has never seen — a classic sign of overfitting to training statement vocabulary.

**Training Curves Observation:**
Training accuracy rose sharply to ~99% within the first 5 epochs and remained flat near 99–100% for the rest of training. Validation accuracy, by contrast, plateaued around 62–67% with high variance throughout all 50 epochs — never tracking the training curve. Training loss converged toward ~0 and stayed flat, while validation loss began around 3, climbed to ~14 by epoch 20, and continued oscillating between 6 and 27, ending near ~26 at epoch 50. This divergence between training and validation loss is the signature of extreme overfitting: the model memorized the training statement vocabulary and produced overconfident, miscalibrated predictions on unseen statements.

### 4.2 Model 2: User Behavior

Run 3 uses the same statement-level split as Model 1 for consistent evaluation. Model 2 also had a scaler leakage issue (StandardScaler was fit on the full dataset); this is corrected in Run 3.

- **Test Accuracy:** 0.5084 (50.84%)
- **Test Loss:** 0.7379

**Confusion Matrix (Run 3 — Final):**

|  | Predicted False | Predicted True |
|:---|:---|:---|
| **Actual False** | 363 (TN) | 12,644 (FP) |
| **Actual True** | 269 (FN) | 12,992 (TP) |

- True Negatives: 363 — correctly identified false claims (only 2.8% of actual false)
- False Positives: 12,644 — false claims predicted as true (97.2% miss rate)
- False Negatives: 269 — true claims predicted as false
- True Positives: 12,992 — correctly identified true claims (98% recall for true)
- The model predicts "True" for nearly every sample — effectively a majority-class predictor

**Training Curves Observation:**
Both training and validation accuracy were flat from epoch 1 — training accuracy hovered near 64% and validation accuracy near 2–5%, with no improvement across all 50 epochs. Training loss declined slightly from ~0.68 to ~0.62, indicating the model slowly learned to predict "True" more confidently on training data. Validation loss remained flat around 1.03–1.05 throughout, never decreasing. The near-zero validation accuracy (2–5%) confirms what the confusion matrix showed: the model collapsed to predicting "True" for virtually every test sample. No learning of generalizable patterns occurred — the curves showed that user behavior features simply do not carry enough signal for the network to distinguish true from false claims on unseen statements.

### 4.3 Comparison Across All Runs

| Metric | Run 1 (Row split, leaky TF-IDF) | Run 2 (Row split, fixed TF-IDF) | Run 3 (Statement split — Final) |
|:---|:---|:---|:---|
| **Model 1 Accuracy** | 99.73% | 99.72% | **77.22%** |
| **Model 1 Loss** | 0.0190 | 0.0274 | 15.6946 |
| **Model 2 Accuracy** | 57.07% | 57.01% | **50.84%** |
| **Model 2 Loss** | 0.6744 | 0.6744 | 0.7379 |

The progression from Run 1 → Run 2 → Run 3 isolates the source of the inflated accuracy: fixing TF-IDF fitting order (Run 2) had almost no effect, while fixing the split to unseen statements (Run 3) reduced Model 1 accuracy by ~22 percentage points. Model 2 converged to near-random performance in all runs, confirming user behavior features carry no meaningful predictive signal.

---

## 5. Model Interpretation (Critical Analysis)

### 5.1 What Worked Well
Even at its honest 77.22% accuracy (Run 3, statement-level split), Model 1 (Linguistic) substantially outperformed Model 2 (User Behavior) at 50.84% — a gap of over 26 percentage points. TF-IDF captured discriminative vocabulary patterns between true and false claims (e.g., specific political language, hedging phrases, sensational wording), while the 20 numeric linguistic features contributed style signals like punctuation frequency, word diversity, and POS distributions. The model learned real patterns from content — 77% accuracy on genuinely unseen claims is a meaningful result for a simple feedforward network with no pre-trained language model.

The three-run iteration process itself was the most valuable part of the analysis. Each run sharpened the understanding: Run 1 revealed suspiciously high accuracy → Run 2 fixed the TF-IDF fit order (minimal effect) → Run 3 fixed the split strategy (major effect). This progression demonstrated that a single result cannot be trusted at face value, and that rigorous evaluation design matters as much as model architecture.

### 5.2 What Went Wrong
**Model 1 — Statement Repetition Leakage (Resolved):** The initial accuracy of 99.73% was inflated because the same `statement` text appeared in both train and test rows. Fixing the TF-IDF fitting order (Run 2) had negligible effect (99.72%), but switching to a statement-level split (Run 3) dropped accuracy to 77.22% — a 22+ point drop that confirms the model was largely memorizing seen statements. The high test loss in Run 3 (15.69) further indicates the model produces overconfident, miscalibrated probabilities on truly unseen claims.

**Model 2 — Majority-Class Collapse:** At 50.84% with the statement-level split, Model 2 effectively collapsed to a majority-class predictor — it predicted "True" for 98% of the test set (12,992 + 12,644 out of 26,268 samples). Recall for the "False" class was only 2.8%. This confirms that user behavior features carry essentially no signal for distinguishing true from false claims, regardless of the split strategy used.

### 5.3 Feature Selection Analysis
For Model 1, TF-IDF features likely carried the majority of the predictive signal — specific words and phrases in the tweet or statement are strong indicators of truthfulness. Among the numeric linguistic features, POS tag counts (e.g., adjective frequency, pronoun usage) and punctuation patterns (e.g., exclamation marks, question marks) may contribute to detection, as misleading content often uses more emotional or sensational language. Less useful features likely include the NER percentages (ORG, FAC, LANGUAGE) for categories that rarely appear across the dataset.

For Model 2, `BotScore` and `cred` are likely the most informative features since they are derived signals already designed to capture account credibility. Raw counts like `followers_count` and `statuses_count` likely added noise — a highly active account is not inherently more or less truthful.

Features that could improve both models: sentiment scores, source credibility labels, claim publication date, or cross-referencing with known fact-checking databases.

### 5.4 Hyperparameter Tuning Rationale
- **epochs=50:** A reasonable starting point for this dataset size (134K samples). For Model 1, convergence likely occurred before epoch 50 — training beyond that risks overfitting. For Model 2, 50 epochs may have been sufficient to confirm that user behavior features simply lack predictive power rather than needing more training.
- **batch_size=64:** A standard choice that balances gradient noise and memory efficiency. Smaller batches (32) would introduce more noise per update — potentially helping Model 2 escape local minima but at the cost of training speed. Larger batches (128) would smooth updates but may cause the optimizer to settle into sharper minima with worse generalization.
- **Dropout rates (0.3/0.2 for Model 1, 0.3/0.2/0.1 for Model 2):** Chosen to be aggressive on the first layers (where overfit risk is highest) and lighter on deeper layers. If re-tuned, reducing Dropout on Model 1 might improve recall without sacrificing much precision.
- **Learning rate:** Left at Adam's default (0.001). Reducing it to 0.0001 with more epochs could produce smoother convergence curves and potentially better generalization for Model 2.
- **If re-running:** The most impactful change would be fixing the data leakage in Model 1 (fit TF-IDF only on training data) to get a more honest accuracy estimate. For Model 2, adding interaction features (e.g., ratio of followers to following) or combining it with Model 1's features into a unified model would be worth exploring.

---

## 6. Feature Engineering Suggestions
Several feature engineering improvements could substantially boost model performance:

1. **Combined model (text + behavior):** The most impactful change would be merging Model 1 and Model 2 features into a single input vector, allowing the network to learn interactions between linguistic content and account behavior simultaneously. A credible account sharing suspicious language is a different signal than an anonymous bot making the same claim.

2. **Sentiment and subjectivity scores:** Using a library like TextBlob or VADER to compute polarity and subjectivity of the tweet and statement. Misleading content frequently uses emotionally charged, highly subjective language.

3. **Claim age / temporal features:** How old the claim is relative to when it was tweeted could be informative — older debunked claims resurface regularly.

4. **Network features:** Whether the tweet was retweeted from a previously fact-checked source, or whether the original account had prior fact-check flags, would add strong external signal.

5. **Cross-lingual or cross-domain generalization:** The statement-level split (implemented in Run 3) enforces unseen-claim evaluation, but all claims in this dataset originate from a single political fact-checking source. Training on claims from multiple domains (health, science, international news) and evaluating cross-domain generalization would test whether the learned vocabulary patterns transfer beyond U.S. political misinformation.

---

## 7. Conclusion
This lab produced a three-run progression that revealed how evaluation design can make the difference between a misleadingly perfect model and an honest one:

| Run | Split Strategy | Model 1 Accuracy | Model 2 Accuracy |
|:---|:---|:---|:---|
| Run 1 | Row-level, TF-IDF fit on full data | 99.73% | 57.07% |
| Run 2 | Row-level, TF-IDF fit on train only | 99.72% | 57.01% |
| **Run 3** | **Statement-level split (final)** | **77.22%** | **50.84%** |

The final honest result is that linguistic content predicts factual accuracy at 77.22% on genuinely unseen claims — a meaningful but imperfect signal. User behavior features are functionally useless (50.84%, near-random), with the model collapsing to always predicting "True."

The key methodological lesson: fixing a preprocessing error (TF-IDF fitting order) had almost no effect on accuracy, while fixing the *evaluation design* (splitting by unique statement instead of by row) caused a 22-point drop. This demonstrates that understanding your data's structure — specifically, that the Truth Seeker Dataset assigns one fact-check label to many rows — is as critical as any modeling decision.

Misinformation detection remains a fundamentally hard language problem. A basic TF-IDF + feedforward network achieves 77% on unseen claims, which suggests meaningful vocabulary differences between true and false content. Improvements would require richer semantic representations (e.g., BERT embeddings), external knowledge integration (fact-checking databases), or combining linguistic and behavioral signals in a single model rather than evaluating them in isolation.
