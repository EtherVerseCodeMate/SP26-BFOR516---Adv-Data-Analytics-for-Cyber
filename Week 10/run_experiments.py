"""
BFOR516 Week 10 Lab - AAPL RNN vs LSTM Experiments
Runs all 4 blocks independently and saves all metrics + plots.
Results are written to metrics.json for automatic report population.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# CONFIGURATION
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = r"C:\Users\intel\OneDrive - University at Albany - SUNY\Documents\MSDF\Spring 2026\SP26 BFOR516 - Adv Data Analytics for Cyber\Week 10\Week 10 Lab attached files Apr 4, 2026 502 PM\apple_1year.csv"
OUT_DIR   = r"C:\Users\intel\OneDrive - University at Albany - SUNY\Documents\MSDF\Spring 2026\SP26 BFOR516 - Adv Data Analytics for Cyber\Week 10"
TEST_DAYS = 10
EPOCHS    = 50
BATCH     = 16

# HELPERS

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
    df = df.sort_values('Date').reset_index(drop=True)
    return df[['Date', 'Close']]


def make_sequences(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


def build_rnn(window, units1=64, units2=32, lr=0.001, dropout=0.2):
    model = Sequential([
        SimpleRNN(units1, return_sequences=True, input_shape=(window, 1)),
        Dropout(dropout),
        SimpleRNN(units2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def build_lstm(window, units1=64, units2=32, lr=0.001, dropout=0.2, bidirectional=False):
    if bidirectional:
        model = Sequential([
            Bidirectional(LSTM(units1, return_sequences=True), input_shape=(window, 1)),
            Dropout(dropout),
            LSTM(units2),
            Dense(1)
        ])
    else:
        model = Sequential([
            LSTM(units1, return_sequences=True, input_shape=(window, 1)),
            Dropout(dropout),
            LSTM(units2),
            Dense(1)
        ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def calc_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return round(float(rmse), 4), round(float(mae), 4), round(float(mape), 4)


def run_block(window, label, rnn_kwargs=None, lstm_kwargs=None,
              epochs=EPOCHS, callbacks=None):
    """Full self-contained block: load → preprocess → split → train → evaluate → plot."""
    print(f"\n{'='*60}")
    print(f"  {label}  |  window={window}")
    print(f"{'='*60}")

    # --- Load & preprocess ---
    df = load_data()
    prices = df['Close'].values
    dates  = df['Date'].values

    # --- Scale ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # --- Train/test split (keep final TEST_DAYS as test) ---
    train_raw = scaled[:-TEST_DAYS]
    test_raw  = scaled[-(window + TEST_DAYS):]   # include look-back window for test seqs

    X_train, y_train = make_sequences(train_raw, window)
    X_test,  y_test  = make_sequences(test_raw, window)

    X_train = X_train.reshape(-1, window, 1)
    X_test  = X_test.reshape(-1, window, 1)

    print(f"  Training sequences : {len(X_train)}")
    print(f"  Test sequences     : {len(X_test)}")

    # --- Build models ---
    rnn_kw  = rnn_kwargs  or {}
    lstm_kw = lstm_kwargs or {}
    cb = callbacks or []

    rnn_model  = build_rnn(window,  **rnn_kw)
    lstm_model = build_lstm(window, **lstm_kw)

    rnn_model.summary()
    lstm_model.summary()

    # --- Train ---
    rnn_hist  = rnn_model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH,
                              validation_split=0.1, callbacks=cb, verbose=0)
    lstm_hist = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=BATCH,
                               validation_split=0.1, callbacks=cb, verbose=0)

    # --- Predict ---
    rnn_pred  = scaler.inverse_transform(rnn_model.predict(X_test,  verbose=0))
    lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test, verbose=0))
    actual    = scaler.inverse_transform(y_test.reshape(-1, 1))

    # --- Metrics ---
    rnn_rmse,  rnn_mae,  rnn_mape  = calc_metrics(actual, rnn_pred)
    lstm_rmse, lstm_mae, lstm_mape = calc_metrics(actual, lstm_pred)

    print(f"\n  RNN  → RMSE={rnn_rmse:.4f}  MAE={rnn_mae:.4f}  MAPE={rnn_mape:.2f}%")
    print(f"  LSTM → RMSE={lstm_rmse:.4f}  MAE={lstm_mae:.4f}  MAPE={lstm_mape:.2f}%")

    # --- Plot: Predicted vs Actual ---
    test_dates = dates[-(TEST_DAYS):]
    _, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test_dates, actual,    label='Actual',   color='black', linewidth=2)
    ax.plot(test_dates, rnn_pred,  label='Vanilla RNN', color='royalblue', linestyle='--', marker='o', markersize=4)
    ax.plot(test_dates, lstm_pred, label='LSTM',     color='crimson',   linestyle='--', marker='s', markersize=4)
    ax.set_title(f'AAPL Closing Price — Predicted vs Actual\n{label} | Window={window}', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    slug = label.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plot_path = os.path.join(OUT_DIR, f'aapl_{slug}_w{window}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")

    # --- Plot: Training loss ---
    _, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(rnn_hist.history['val_loss'],  label='RNN Val Loss',  color='royalblue')
    ax2.plot(lstm_hist.history['val_loss'], label='LSTM Val Loss', color='crimson')
    ax2.set_title(f'Validation Loss — {label} | Window={window}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(OUT_DIR, f'aapl_{slug}_w{window}_loss.png')
    plt.savefig(loss_path, dpi=150)
    plt.close()

    return {
        'rnn':  {'rmse': rnn_rmse,  'mae': rnn_mae,  'mape': rnn_mape},
        'lstm': {'rmse': lstm_rmse, 'mae': lstm_mae, 'mape': lstm_mape},
        'actual': actual.flatten().tolist(),
        'rnn_pred': rnn_pred.flatten().tolist(),
        'lstm_pred': lstm_pred.flatten().tolist(),
        'test_dates': [str(d)[:10] for d in test_dates],
        'n_train_seq': len(X_train),
    }


# ─────────────────────────────────────────────
# BLOCK 1 — Window 40 (Baseline)
# ─────────────────────────────────────────────
b1 = run_block(window=40, label="Block 1 Baseline")

# ─────────────────────────────────────────────
# BLOCK 2 — Window 60 (Baseline)
# ─────────────────────────────────────────────
b2 = run_block(window=60, label="Block 2 Baseline")

# ─────────────────────────────────────────────
# BLOCK 3 — Window 80 (Baseline)
# NOTE: 252 - 10 = 242 training rows.
#       Window 80 → 242 - 80 = 162 training sequences (lean but workable).
#       Expect noisier validation loss curves vs Window 40/60.
# ─────────────────────────────────────────────
b3 = run_block(window=80, label="Block 3 Baseline")

# ─────────────────────────────────────────────
# BLOCK 4 — Tuned Models (best window = 60 by design, user adjusts if needed)
# Tuned RNN:  128→64 units, Dropout(0.3), LR=0.0005
# Tuned LSTM: 128→64 units, Bidirectional, EarlyStopping
# ─────────────────────────────────────────────
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

b4 = run_block(
    window=60,
    label="Block 4 Tuned",
    rnn_kwargs  = {'units1': 128, 'units2': 64, 'lr': 0.0005, 'dropout': 0.3},
    lstm_kwargs = {'units1': 128, 'units2': 64, 'lr': 0.001,  'dropout': 0.2, 'bidirectional': True},
    epochs=100,
    callbacks=[es]
)

# WRITE output_metrics.txt
if b1['rnn']['rmse'] <= b2['rnn']['rmse'] and b1['rnn']['rmse'] <= b3['rnn']['rmse']:
    best_rnn_label = "W=40"
elif b2['rnn']['rmse'] <= b3['rnn']['rmse']:
    best_rnn_label = "W=60"
else:
    best_rnn_label = "W=80"

if b1['lstm']['rmse'] <= b2['lstm']['rmse'] and b1['lstm']['rmse'] <= b3['lstm']['rmse']:
    best_lstm_label = "W=40"
elif b2['lstm']['rmse'] <= b3['lstm']['rmse']:
    best_lstm_label = "W=60"
else:
    best_lstm_label = "W=80"
metrics_txt = f"""
BFOR516 Week 10 Lab — AAPL Stock Prediction
Model Performance Summary (All Runs)
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
================================================================

Model           | Window | RMSE       | MAE        | MAPE
----------------|--------|------------|------------|----------
RNN  (W=40)     |   40   | {b1['rnn']['rmse']:10.4f} | {b1['rnn']['mae']:10.4f} | {b1['rnn']['mape']:8.2f}%
LSTM (W=40)     |   40   | {b1['lstm']['rmse']:10.4f} | {b1['lstm']['mae']:10.4f} | {b1['lstm']['mape']:8.2f}%
RNN  (W=60)     |   60   | {b2['rnn']['rmse']:10.4f} | {b2['rnn']['mae']:10.4f} | {b2['rnn']['mape']:8.2f}%
LSTM (W=60)     |   60   | {b2['lstm']['rmse']:10.4f} | {b2['lstm']['mae']:10.4f} | {b2['lstm']['mape']:8.2f}%
RNN  (W=80)     |   80   | {b3['rnn']['rmse']:10.4f} | {b3['rnn']['mae']:10.4f} | {b3['rnn']['mape']:8.2f}%
LSTM (W=80)     |   80   | {b3['lstm']['rmse']:10.4f} | {b3['lstm']['mae']:10.4f} | {b3['lstm']['mape']:8.2f}%
RNN  (Tuned)    |   60   | {b4['rnn']['rmse']:10.4f} | {b4['rnn']['mae']:10.4f} | {b4['rnn']['mape']:8.2f}%
LSTM (Tuned)    |   60   | {b4['lstm']['rmse']:10.4f} | {b4['lstm']['mae']:10.4f} | {b4['lstm']['mape']:8.2f}%

================================================================
BEST BASELINE RNN  (lowest RMSE): {best_rnn_label}
BEST BASELINE LSTM (lowest RMSE): {best_lstm_label}
"""

metrics_path = os.path.join(OUT_DIR, 'output_metrics.txt')
with open(metrics_path, 'w') as f:
    f.write(metrics_txt)
print(f"\nMetrics written to: {metrics_path}")

# ─────────────────────────────────────────────
# WRITE metrics.json (used by report generator)
# ─────────────────────────────────────────────
results = {
    'w40': b1, 'w60': b2, 'w80': b3, 'tuned': b4,
    'best_rnn_window':  min([('40', b1['rnn']['rmse']), ('60', b2['rnn']['rmse']), ('80', b3['rnn']['rmse'])], key=lambda x: x[1])[0],
    'best_lstm_window': min([('40', b1['lstm']['rmse']), ('60', b2['lstm']['rmse']), ('80', b3['lstm']['rmse'])], key=lambda x: x[1])[0],
}

json_path = os.path.join(OUT_DIR, 'metrics.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"JSON written to:    {json_path}")
print("\nAll experiments complete.")
