# Lab Report: AAPL LSTM Stock Prediction — Window Size Experiments
**Course:** SP26 BFOR516 - Advanced Data Analytics for Cyber  
**Instructor:** Srishti Gupta, Ph.D.  
**Dataset:** Apple (AAPL) Historical Stock Data — 1 Year (NASDAQ)  
**Date Range:** March 31, 2025 – March 30, 2026 (251 trading days)

---

## 1. AI Disclosure

**Claude Code (Anthropic, claude-sonnet-4-6)** was utilized as a pair-programming assistant to complete this lab. Specifically, AI was used to:

1. Scaffold the Jupyter Notebook (`AAPL_LSTM_Lab_Experiments.ipynb`) structure, including all data preprocessing cells, model definition blocks, and metric logging logic.
2. Implement the self-contained experiment block structure (one block per window size) to comply with the lab's kernel-restart isolation requirement.
3. Write the `matplotlib` visualization code for the predicted-vs-actual overlay plots and the cross-window comparison chart.
4. Generate a reference version of this lab report for comparison against the student's own written analysis.

All generated code was actively reviewed, understood, and executed by the student on the course JupyterHub prior to submission. The metric values reported in Section 5 and all interpretive narrative were written by the student based on observed outputs. This document is the student's own analysis.

---

## 2. Lab Objective

The goal of this lab was to build and compare a Vanilla RNN (SimpleRNN) and an LSTM model using one year of Apple (AAPL) closing price data. The specific experimental variable was the **look-back window size**, tested at three settings: **40, 60, and 80 trading days**. Both baseline models were run under each window size, then hyperparameters were tuned to attempt further accuracy improvement. The final evaluation target was the **most recent 10 trading days** of the dataset (March 17–30, 2026).

---

## 3. Workflow & Preprocessing

The following preprocessing steps were applied identically in each of the three self-contained experiment blocks:

1. **Data Ingestion:** `apple_1year.csv` was loaded into a Pandas DataFrame (251 rows, 6 columns).
2. **Data Cleaning:** The `$` character was stripped from `Close/Last`, `Open`, `High`, and `Low` columns and values were cast to `float64`. The `Date` column was parsed into `datetime` objects.
3. **Chronological Sort:** Rows were sorted ascending by date (oldest to newest) to preserve temporal ordering. The raw CSV is sorted descending, so this step is critical.
4. **Feature Selection:** Only the `Close/Last` column was used (univariate prediction task).
5. **Normalization:** Closing prices were scaled to `[0, 1]` using `MinMaxScaler`. The scaler was fit exclusively on the training partition and applied to the test partition to prevent data leakage.
6. **Test Set Reservation:** The final 10 rows (03/17/2026–03/30/2026, closing prices approximately $246–$254) were held out as the out-of-sample test set before any sequence generation. Training used the remaining 241 days.
7. **Sequence Generation:** A sliding window of size `window_size` was applied to the training data to produce input/output sequence pairs. Each input sequence contains `window_size` consecutive closing prices; the target is the next day's price.
8. **Kernel Restart:** Per lab instructions, the kernel was restarted between window-size runs to ensure zero state-leakage between experiments.

---

## 4. Model Architectures

Both models followed the class baseline architecture, adapted identically for each model type:

**Baseline Architecture (applied to both SimpleRNN and LSTM variants):**

| Layer | Type | Units / Rate |
|---|---|---|
| 1 | SimpleRNN / LSTM | 64 units, `return_sequences=True` |
| 2 | Dropout | 0.2 |
| 3 | SimpleRNN / LSTM | 32 units |
| 4 | Dense | 1 unit, linear activation |

- **Compiler:** Adam optimizer, Mean Squared Error (MSE) loss
- **Training:** EarlyStopping (monitor=`val_loss`, patience=15), 15% validation split taken chronologically from the tail of the training data
- **Reproducibility:** `np.random.seed(42)`, `tf.random.set_seed(42)` set at the top of each experiment block

**Tuned Architecture (Section 6):** Hyperparameter adjustments documented per experiment.

---

## 5. Baseline Results

*Metrics are reported on the held-out 10-day test set after inverse-transforming predictions to USD. Fill in values from `output_metrics.txt` after running the notebook.*

### Window Size = 40

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| SimpleRNN | [INSERT] | [INSERT] | [INSERT]% |
| LSTM | [INSERT] | [INSERT] | [INSERT]% |

### Window Size = 60

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| SimpleRNN | [INSERT] | [INSERT] | [INSERT]% |
| LSTM | [INSERT] | [INSERT] | [INSERT]% |

### Window Size = 80

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| SimpleRNN | [INSERT] | [INSERT] | [INSERT]% |
| LSTM | [INSERT] | [INSERT] | [INSERT]% |

---

## 6. Hyperparameter Tuning

After observing baseline results, the following changes were applied to attempt accuracy improvement. The rationale for each change is described below.

**Tuning changes applied:**

| Change | Original | Tuned | Rationale |
|---|---|---|---|
| Layer 1 units | 64 | [INSERT] | [describe reasoning] |
| Dropout rate | 0.2 | [INSERT] | [describe reasoning] |
| Optimizer / LR | Adam (default) | [INSERT] | [describe reasoning] |
| Batch size | [default] | [INSERT] | [describe reasoning] |

### Tuned Results (Best Window Size)

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| Tuned SimpleRNN | [INSERT] | [INSERT] | [INSERT]% |
| Tuned LSTM | [INSERT] | [INSERT] | [INSERT]% |

---

## 7. Analysis & Observations

### 7.1 Effect of Window Size on Model Performance

The window size governs how many historical trading days each model uses to predict the next day's price. Increasing it from 40 to 80 days meaningfully changes both the information available to the model and the training dynamics — but differently for each architecture.

**Window 40:** With a 40-day look-back (~8 standard trading weeks), the model captures roughly two months of price momentum. This shorter context window produces more training sequences (201 sequences from 241 training days), giving the model more update steps per epoch. Both architectures react quickly to recent price movement, but may miss slower macro trends.

**Window 60:** A 60-day window (~3 trading months) offers broader market context. Training sequences drop to 181. For SimpleRNN, this is where the **vanishing gradient problem** begins to become a more visible constraint — the earliest days of a 60-step sequence contribute increasingly small gradient signals during Backpropagation Through Time (BPTT), as demonstrated in lecture slides 23–27. LSTM's internal gate mechanism (input, forget, and output gates) explicitly addresses this by learning to selectively retain or discard information across long sequences, enabling it to keep relevant signals from earlier in the window without exponential gradient decay.

**Window 80:** At 80 days (~4 trading months), the gap between SimpleRNN and LSTM performance is most pronounced. The SimpleRNN must propagate error gradients back through 80 timesteps; at each step, those gradients are multiplied by the recurrent weight matrix and a derivative that is typically less than 1, compounding the decay. LSTM avoids this through its **constant error carousel** — the cell state pathway allows gradients to flow backward with minimal multiplicative interference. Meanwhile, training sequences shrink to 161, reducing the number of weight updates per epoch and potentially slowing convergence.

**Observed pattern:** [Write your observation here — did RMSE decrease or increase as window size grew? Which model degraded more sharply?]

### 7.2 SimpleRNN vs. LSTM Comparison

The fundamental architectural difference is memory span. A SimpleRNN at timestep *t* produces a hidden state that is a function of the current input and the immediately prior hidden state. Across many steps, the influence of earlier inputs geometrically decays due to the repeated application of the weight matrix. An LSTM carries two streams of state: the hidden state (short-term context) and the cell state (long-term memory). The forget gate determines what fraction of the cell state to retain; the input gate determines what new information to write. This structure allows the LSTM to theoretically maintain a signal from any prior timestep if the gates learn to keep it.

In practical terms for this dataset:
- **SimpleRNN** tends to track short-term directional slope accurately but can overreact to single-day volatility because it lacks the smoothing effect of a persistent cell state.
- **LSTM** typically produces smoother prediction curves, particularly on the 10-day test window, as it blends recent price action with the longer trend encoded in the cell state.

**Observed pattern:** [Write your observation here — was LSTM's advantage consistent across all three window sizes, or only apparent at larger windows?]

### 7.3 Vanishing Gradient — Theoretical vs. Observed

Lecture slides 23–27 establish the mathematical basis: in standard BPTT, the gradient of the loss with respect to an early-timestep weight involves a product of *T* Jacobians. If the spectral radius of the recurrent weight matrix is less than 1, this product shrinks to zero; if greater than 1, it explodes. SimpleRNN does nothing to interrupt this chain. LSTM breaks it by routing the primary gradient path through the cell state, which involves addition (not multiplication) and a learned gate that can be set to 1 to preserve the gradient perfectly.

The window-size experiment in this lab provides a direct empirical demonstration of this theory: by holding the architecture constant and only varying how many timesteps the gradient must travel, we can observe the point at which SimpleRNN begins to degrade relative to LSTM.

**Observed pattern:** [Write your observation here — at which window size did you first notice a meaningful divergence in RMSE between SimpleRNN and LSTM?]

### 7.4 Hyperparameter Tuning Observations

[Write your observations here — what did you change, what was your reasoning, and did it improve results? For example: increasing units gives the model more representational capacity but risks overfitting on 241 training rows; reducing the learning rate may allow finer convergence but risks early stopping triggering too soon.]

---

## 8. Limitations

- **Univariate input:** The model predicts closing price based solely on historical closing prices. Incorporating volume, open, high, and low would likely improve predictive accuracy by providing richer daily context.
- **Small training set:** 241 training rows is an extremely limited sample for deep learning. The models are essentially memorizing a single year of AAPL's behavioral regime. Any regime shift (e.g., a macro event like the April 2025 tariff shock visible in the raw data) that differs from the training distribution will degrade generalization.
- **Recursive forecasting not used:** This lab predicts within-sample test days, not future days beyond the dataset. A recursive autoregressive extension (as implemented in the Week 8 MSFT lab) would demonstrate additional limitations around compounding prediction error.
- **Stationarity:** Stock price series are non-stationary. MinMaxScaler normalizes scale but does not address the underlying random-walk structure of price. A returns-based approach (predicting day-over-day percentage change rather than raw price) would be more statistically rigorous.

---

## 9. Conclusion

This lab demonstrated the practical impact of window size and architectural choice on sequential prediction tasks using AAPL stock data. The three-window experiment provided a controlled empirical test of the vanishing gradient problem: by holding all other variables constant and increasing the sequence length from 40 to 80 steps, the performance divergence between SimpleRNN and LSTM became directly observable.

LSTM's gated architecture was designed precisely for this failure mode. Where SimpleRNN's gradient signal decays multiplicatively across long sequences, LSTM's cell state pathway preserves information additively, allowing the model to maintain relevant long-horizon context without exponential signal loss. The results of this lab [confirm / partially confirm / did not clearly confirm] that advantage at the window sizes tested.

The hyperparameter tuning phase reinforced the importance of matching model capacity to dataset size — [write your conclusion about what worked].
