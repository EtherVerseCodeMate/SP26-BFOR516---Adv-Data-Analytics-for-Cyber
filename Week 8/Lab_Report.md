# Lab Report: MSFT RNN Stock Prediction
**Course:** SP26 BFOR516 - Adv Data Analytics for Cyber  
**Instructor:** Srishti Gupta, Ph.D.  
**Dataset:** Microsoft (MSFT) Historical Stock Data (NASDAQ)  

---

## 1. AI Disclosure
**Antigravity AI (Google DeepMind)** was utilized as a pair-programming assistant to complete this lab. Specifically, AI was used to:
1. Scaffold the initial Jupyter Notebook structure and write boilerplate code using `pandas`, `matplotlib`, and `sklearn.preprocessing.MinMaxScaler`.
2. Troubleshoot a Python 3.13 compatibility issue by explicitly upgrading the environment to TensorFlow 2.21, ensuring the notebook could execute.
3. Assist in generating the `matplotlib` visualization code to neatly compare the SimpleRNN and LSTM outputs.
4. Help format this final lab report. 

All generated code and analysis were actively reviewed, understood, and tested by the student prior to submission. The narrative analysis and model implementation strategy accurately represent the student's understanding of sequential data modeling.

---

## 2. Lab Objective
The goal of this lab was to build a Recurrent Neural Network (RNN) from scratch using the `TensorFlow/Keras` framework to predict the closing stock prices of Microsoft (MSFT). The methodology closely mirrored the AAPL in-class demonstration, applying data normalization, sequenced sliding windows, and predictive modeling sequentially to the MSFT dataset.

---

## 3. Workflow & Preprocessing
1. **Data Ingestion:** The MSFT dataset (125 trading days) was loaded into a Pandas DataFrame.
2. **Data Cleaning:** The dollar signs (`$`) were stripped from the pricing data, and the variables were cast to floating-point numbers. Dates were parsed into datetime objects and sorted chronologically (oldest to newest).
3. **Normalization:** Because RNNs rely on gradient descent and `tanh`/`sigmoid` activations, the closing prices were scaled down to a `[0, 1]` range using `MinMaxScaler`.
4. **Sequence Generation:** A sliding window approach was implemented using a `look_back` of 10 days (representing two full standard trading weeks, which balances short-term market momentum without washing out immediate trend reactions). The model was trained to look at the past 10 days to predict the $11^{th}$ day's closing price. Crucially, the data was split **chronologically** (the first 80% as training and the final 20% as testing) rather than randomly, to strictly prevent look-ahead bias and data leakage.

---

## 4. Model Architectures & Training
Two models were constructed to evaluate performance differences:

1. **SimpleRNN Model:** 
   - A sequential architecture featuring a `SimpleRNN` layer (64 units, returning sequences), a `Dropout` layer (20% rate to prevent overfitting), a second `SimpleRNN` layer (32 units), and a final `Dense` linear output node.
2. **LSTM Model:**
   - Designed to address the *vanishing gradient problem* inherent to SimpleRNNs. It used the exact same architecture but substituted standard RNN layers with Long Short-Term Memory (`LSTM`) gated memory cells.

Both models were compiled using the `Adam` optimizer and the `Mean Squared Error (MSE)` loss function. We implemented Keras `EarlyStopping` (monitoring validation loss with a patience of 15 epochs) by taking a strict chronological 15% internal validation split from the tail end of the training data. This ensured the models did not overfit, while keeping the true 20% test set completely out-of-sample and untainted.

---

## 5. Evaluation & Results
Before calculating the final error metrics or visualizing the timeline, the raw `[0, 1]` model predictions were passed through the scaler's `inverse_transform()` method to remap the values back into the real-world USD domain. Both models performed admirably on the test data, successfully mapping the broader volatile trends of MSFT stock prices from January through March 2026. 

**Test Set Evaluation Metrics:**
*Note: While neural networks are naturally stochastic, a random seed (`seed=42`) was explicitly set in the beginning of the notebook to guarantee strict reproducibility of the reported outputs and models.*

* **SimpleRNN:** Captured the short-term directional slope efficiently but sometimes overreacted to sudden day-over-day volatility.
* **LSTM:** Demonstrated superior memory retention. Because SimpleRNNs are trained using Backpropagation Through Time (BPTT), longer sequences are prone to the vanishing gradient problem (as mapped out in lecture slides 23-27). The internal gates of the LSTM variant effectively mitigated this mathematical decay, addressing the gradient vanishing issue and resulting directly in smoothed predictions and superior error metrics (e.g., lower RMSE and MAPE). 

---

## 6. Future Price Forecast
To finalize the pipeline, the best-performing model was used to forecast the next 5 trading days beyond the dataset's maximum date. 
Using a recursive algorithm, the model predicted day $T+1$, appended that prediction to the sliding window, and used the new window to predict $T+2$ up to $T+5$. The results (visible in the notebook's final output cell and the generated `msft_forecast.png` chart) demonstrated a realistic, smoothed continuation of the prevailing stock momentum.

**Autoregressive Challenges:** A core limitation of this recursive autoregressive methodology (noted in lecture slide 7) is that prediction errors compound mathematically. Because each forecasted day feeds forward into the feature window of the subsequent prediction, minute deviations are cascaded, causing confidence limits to degrade significantly over longer horizons.

---

## 7. Analysis & Conclusion
* **Data Scarcity:** The primary limitation in this lab is the small sample size (125 days). Deep learning models typically thrive on thousands of sequence steps.
* **Feature Unidimensionality:** This was a univariate model predicting closing price exclusively based on historical closing prices.
* **Conclusion:** Despite these limitations, the lab successfully proved that even a lightweight RNN structure can capture strong autocorrelation signals in financial time series. As observed during the AAPL class demo, upgrading to an LSTM network directly mitigates the short-term memory constraints of a SimpleRNN, establishing it as a highly capable architecture for sequential regression tasks.
